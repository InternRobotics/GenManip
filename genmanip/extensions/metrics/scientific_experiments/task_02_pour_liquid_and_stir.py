from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from pxr import UsdGeom  # type: ignore
from scipy.spatial import ConvexHull, QhullError

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory
from genmanip.extensions.metrics.default.sr_based_genmanip_relationship import (
    SRBasedGenmanipRelationship,
    is_inside,
    is_point_in_convex_hull_fast,
)


GC_LIQUID_IN_BEAKER_TIMING_1_THRESHOLD = 0.45
GC_LIQUID_IN_BEAKER_TIMING_2_THRESHOLD = 0.90
GLASS_ROD_IN_BEAKER_TIMING_3_THRESHOLD = 0.1


class PourLiquidAndStirConfig(BaseModel):
    graduated_cylinder_uid: str = Field(
        ..., description="Prim name or path of the graduated cylinder object"
    )
    beaker_uid: str = Field(..., description="Prim name or path of the beaker object")
    glass_rod_uid: str = Field(
        ..., description="Prim name or path of the graduated glass rod object"
    )
    plate_uid: str = Field(
        ..., description="Prim name or path of the graduated plate object"
    )
    gc_liquid_mesh_prim_path: str = Field(
        ..., description="Prim path of the liquid mesh in graduated cylinder object"
    )
    b_liquid_mesh_prim_path: str = Field(
        ..., description="Prim path of the liquid mesh in beaker object"
    )
    condition_type: str = Field(..., description="condition type")


@MetricFactory.register("manip/scientific_experiments/task_02_pour_liquid_and_stir")
class PourLiquidAndStir(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = PourLiquidAndStirConfig(**sub_goal_setting)

    def check_status(self, scene):
        pclist = SRBasedGenmanipRelationship.get_target_pc_list(
            scene, [self.setting.glass_rod_uid, self.setting.beaker_uid]
        )
        glass_rod_vertices = pclist.get(self.setting.glass_rod_uid)
        beaker_vertices = pclist.get(self.setting.beaker_uid)

        gc_liquid_prim = scene.world.stage.GetPrimAtPath(
            self.setting.gc_liquid_mesh_prim_path
        )
        gc_liquid_points = gc_liquid_prim.GetAttribute("points").Get()
        gc_liquid_world_transform = UsdGeom.XformCache().GetLocalToWorldTransform(
            gc_liquid_prim
        )
        gc_liquid_particles = np.asarray(
            [gc_liquid_world_transform.Transform(point) for point in gc_liquid_points],
            dtype=np.float64,
        )

        b_liquid_prim = scene.world.stage.GetPrimAtPath(
            self.setting.b_liquid_mesh_prim_path
        )
        b_liquid_points = b_liquid_prim.GetAttribute("points").Get()
        b_liquid_world_transform = UsdGeom.XformCache().GetLocalToWorldTransform(
            b_liquid_prim
        )
        b_liquid_particles = np.asarray(
            [b_liquid_world_transform.Transform(point) for point in b_liquid_points],
            dtype=np.float64,
        )

        if self.setting.condition_type in ["timing_1", "timing_2"]:
            if beaker_vertices is None:
                return False

            threshold = (
                GC_LIQUID_IN_BEAKER_TIMING_1_THRESHOLD
                if self.setting.condition_type == "timing_1"
                else GC_LIQUID_IN_BEAKER_TIMING_2_THRESHOLD
            )
            return is_inside(
                src_pts=gc_liquid_particles,
                target_pts=beaker_vertices,
                thresh=threshold,
                scene=scene,
            )

        if self.setting.condition_type == "timing_3":
            if (
                glass_rod_vertices is None
                or beaker_vertices is None
                or not len(glass_rod_vertices)
                or not len(beaker_vertices)
            ):
                return False

            if not is_inside(
                src_pts=glass_rod_vertices,
                target_pts=beaker_vertices,
                thresh=GLASS_ROD_IN_BEAKER_TIMING_3_THRESHOLD,
                scene=scene,
            ):
                return False

            # Include both liquid sources, regardless of their original container.
            liquid_particles = np.concatenate(
                (
                    gc_liquid_particles.reshape(-1, 3),
                    b_liquid_particles.reshape(-1, 3),
                )
            )
            if not len(liquid_particles):
                return False

            try:
                beaker_hull = ConvexHull(beaker_vertices)
            except (QhullError, ValueError):
                return False

            particles_in_beaker = liquid_particles[
                is_point_in_convex_hull_fast(liquid_particles, beaker_hull)
            ]
            if not len(particles_in_beaker):
                return False

            return glass_rod_vertices[:, 2].min() < particles_in_beaker[:, 2].max()

        if (
            self.setting.condition_type
            == "timing_4/graduated_cylinder_liquid_in_beaker"
        ):
            if beaker_vertices is None:
                return False

            return is_inside(
                src_pts=gc_liquid_particles,
                target_pts=beaker_vertices,
                thresh=GC_LIQUID_IN_BEAKER_TIMING_2_THRESHOLD,
                scene=scene,
            )

        if self.setting.condition_type == "timing_4/beaker_liquid_in_beaker":
            if beaker_vertices is None:
                return False

            return is_inside(
                src_pts=b_liquid_particles,
                target_pts=beaker_vertices,
                thresh=GC_LIQUID_IN_BEAKER_TIMING_2_THRESHOLD,
                scene=scene,
            )

        return False
