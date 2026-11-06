"""Build smooth liquid surface meshes from recorded particle trajectories."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import shutil

# Surface reconstruction is parallelized across processes. Keep native numerical
# libraries single-threaded inside each worker to avoid oversubscribing the host.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import lmdb
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
from tqdm import tqdm

CACHE_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trajectory_root",
        type=Path,
        help="trajectory directory containing episode folders",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--kernel-radius", type=float, default=1.6)
    parser.add_argument("--voxel-scale", type=float, default=0.45)
    parser.add_argument("--iso-ratio", type=float, default=0.35)
    parser.add_argument("--smooth-iterations", type=int, default=4)
    parser.add_argument("--max-voxels", type=int, default=6_000_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_scalar(trajectory, key):
    with (trajectory / "meta_info.pkl").open("rb") as file:
        meta = pickle.load(file)
    key = key.encode()
    if key not in meta["keys"]["scalar_data"]:
        raise KeyError(key.decode())

    env = lmdb.open(
        str(trajectory / "lmdb"),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    with env.begin() as txn:
        value = pickle.loads(txn.get(key))
    env.close()
    return value


def estimate_spacing(points):
    points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    distances = cKDTree(points).query(points, k=2)[0][:, 1]
    distances = distances[np.isfinite(distances) & (distances > 1e-6)]
    return float(np.median(distances))


def splat_particles(points, lower, voxel_size, shape):
    grid = np.zeros(tuple(shape), dtype=np.float32)
    coordinates = (points - lower) / voxel_size
    base = np.floor(coordinates).astype(np.int32)
    fraction = coordinates - base

    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                offset = np.array([x, y, z], dtype=np.int32)
                indices = np.clip(base + offset, 0, shape - 1)
                weights = np.prod(
                    np.where(offset == 1, fraction, 1.0 - fraction), axis=1
                )
                np.add.at(
                    grid,
                    (indices[:, 0], indices[:, 1], indices[:, 2]),
                    weights,
                )
    return grid


def taubin_smooth(vertices, faces, iterations):
    if iterations <= 0 or not len(faces):
        return vertices

    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])

    def pass_once(current, factor):
        totals = np.zeros_like(current)
        counts = np.zeros(len(current), dtype=np.float32)
        np.add.at(totals, edges[:, 0], current[edges[:, 1]])
        np.add.at(counts, edges[:, 0], 1.0)
        average = totals / np.maximum(counts[:, None], 1.0)
        return current + factor * (average - current)

    for _ in range(iterations):
        vertices = pass_once(vertices, 0.4)
        vertices = pass_once(vertices, -0.42)
    return vertices


def vertex_normals(vertices, faces):
    normals = np.zeros_like(vertices)
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-12)


def reconstruct_surface(job):
    frame, points, spacing, output, params = job
    points = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 4:
        np.savez_compressed(
            output,
            vertices=np.empty((0, 3), dtype=np.float32),
            faces=np.empty((0, 3), dtype=np.int32),
            normals=np.empty((0, 3), dtype=np.float32),
        )
        return frame

    voxel_size = spacing * params["voxel_scale"]
    kernel_radius = spacing * params["kernel_radius"]
    padding = 3.0 * kernel_radius
    lower = np.floor((points.min(axis=0) - padding) / voxel_size) * voxel_size
    upper = points.max(axis=0) + padding
    shape = np.ceil((upper - lower) / voxel_size).astype(np.int64) + 2

    voxel_count = int(np.prod(shape))
    if voxel_count > params["max_voxels"]:
        scale = (voxel_count / params["max_voxels"]) ** (1.0 / 3.0)
        voxel_size *= scale
        lower = np.floor((points.min(axis=0) - padding) / voxel_size) * voxel_size
        shape = np.ceil((upper - lower) / voxel_size).astype(np.int64) + 2

    mass = splat_particles(points, lower, voxel_size, shape)
    density = gaussian_filter(
        mass,
        sigma=kernel_radius / voxel_size,
        mode="constant",
        truncate=3.0,
    )
    indices = np.rint((points - lower) / voxel_size).astype(np.int64)
    indices = np.clip(indices, 0, shape - 1)
    particle_density = density[indices[:, 0], indices[:, 1], indices[:, 2]]
    level = float(np.percentile(particle_density, 25)) * params["iso_ratio"]

    vertices, faces, _, _ = marching_cubes(
        density,
        level=level,
        spacing=(voxel_size,) * 3,
        allow_degenerate=False,
    )
    vertices = vertices.astype(np.float32) + lower.astype(np.float32)
    faces = faces.astype(np.int32)
    vertices = taubin_smooth(vertices, faces, params["smooth_iterations"])
    normals = vertex_normals(vertices, faces).astype(np.float32)
    np.savez_compressed(
        output,
        vertices=vertices.astype(np.float32),
        faces=faces,
        normals=normals,
    )
    return frame


def build_episode(trajectory, args, pool=None):
    info = json.loads((trajectory / "lmdb" / "info.json").read_text())
    mappings = info.get("observation/points_mapping", {})
    cache_root = trajectory / "liquid_surface_cache"
    manifest_path = cache_root / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        print(f"skip {trajectory.name}: cache already exists")
        return
    if cache_root.exists() and args.overwrite:
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    params = {
        "kernel_radius": args.kernel_radius,
        "voxel_scale": args.voxel_scale,
        "iso_ratio": args.iso_ratio,
        "smooth_iterations": args.smooth_iterations,
        "max_voxels": args.max_voxels,
    }
    manifest = {"version": CACHE_VERSION, "parameters": params, "points": {}}

    for key, mapping in mappings.items():
        values = read_scalar(trajectory, f"observation/points/{key}/points")
        folder = str(key).replace("/", "_")
        output_dir = cache_root / folder
        output_dir.mkdir(exist_ok=True)
        spacing = estimate_spacing(values[0])
        manifest["points"][str(key)] = {
            "folder": folder,
            "prim_path": mapping["prim_path"],
            "frame_count": len(values),
            "particle_spacing": spacing,
        }

        jobs = [
            (
                frame,
                points,
                spacing,
                output_dir / f"{frame:06d}.npz",
                params,
            )
            for frame, points in enumerate(values)
        ]
        label = f"{trajectory.name}/points {key}"
        if args.workers == 1:
            for job in tqdm(jobs, desc=label):
                reconstruct_surface(job)
        else:
            if pool is None:
                raise RuntimeError("A process pool is required when workers > 1")
            chunksize = max(1, len(jobs) // (args.workers * 8))
            list(
                tqdm(
                    pool.map(reconstruct_surface, jobs, chunksize=chunksize),
                    total=len(jobs),
                    desc=label,
                )
            )

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done {trajectory.name}: {cache_root}")


def find_episodes(trajectory_root):
    trajectory_root = trajectory_root.resolve()
    if (trajectory_root / "lmdb" / "info.json").is_file():
        return [trajectory_root]
    return [
        episode
        for episode in sorted(trajectory_root.iterdir())
        if episode.is_dir()
        and (episode / "lmdb" / "info.json").is_file()
        and (episode / "meta_info.pkl").is_file()
    ]


def build_cache(args):
    episodes = find_episodes(args.trajectory_root)
    if not episodes:
        raise FileNotFoundError(
            f"No trajectory episodes found in {args.trajectory_root}"
        )
    print(f"found {len(episodes)} episode(s)")
    if args.workers == 1:
        for episode in episodes:
            build_episode(episode, args)
        return

    print(
        f"starting spawn process pool with {args.workers} worker(s)",
        flush=True,
    )
    # Reuse one spawn-based pool for all point sets and episodes. Creating a
    # fresh spawn pool for every mapping is expensive, especially at 32+ workers.
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
    ) as pool:
        for episode in episodes:
            build_episode(episode, args, pool)


if __name__ == "__main__":
    build_cache(parse_args())
