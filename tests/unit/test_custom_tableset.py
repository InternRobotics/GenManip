"""Simulator-free control-flow tests; actual geometry/contact needs Isaac probes."""

import ast
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from shapely.geometry import Polygon


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "genmanip/demogen/random_place/random_place.py"
)


def load_functions(namespace):
    # Isolate the actual functions from imports that require a running SimulationApp.
    tree = ast.parse(SOURCE.read_text())
    selected = {"setup_random_custom_tableset", "randomly_place_object_on_object"}
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ] + [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in selected
    ]
    exec(compile(ast.fix_missing_locations(tree), str(SOURCE), "exec"), namespace)
    return namespace


class Object:
    def __init__(self):
        self.position = np.array([0.12, 0.23, 1.04])
        self.orientation = np.array([0.5, 0.5, 0.5, 0.5])
        self.writes = 0

    def get_world_pose(self):
        return self.position.copy(), self.orientation.copy()

    def set_world_pose(self, position=None, orientation=None):
        if position is not None:
            self.position = np.array(position, copy=True)
        if orientation is not None:
            self.orientation = np.array(orientation, copy=True)
        self.writes += 1


class CustomTablesetTest(unittest.TestCase):
    def setUp(self):
        self.ns = load_functions(
            {"np": np, "random": random.Random(0), "Polygon": Polygon}
        )

    def jitter(self, obj, w, h):
        return self.ns["setup_random_custom_tableset"](
            {"prop": obj},
            {},
            {},
            {"prop": {"type": "local_jitter", "w": w, "h": h}},
            True,
        )

    def test_local_jitter_bounds_and_preserved_pose(self):
        for seed in range(1000):
            self.ns["random"].seed(seed)
            obj = Object()
            initial, orientation = obj.get_world_pose()
            self.assertEqual(self.jitter(obj, 0.02, 0.02), 0)
            self.assertTrue((np.abs(obj.position[:2] - initial[:2]) <= 0.01).all())
            self.assertEqual(obj.position[2], initial[2])
            np.testing.assert_array_equal(obj.orientation, orientation)
            self.assertEqual(obj.writes, 1)

    def test_zero_and_invalid_jitter(self):
        obj = Object()
        initial = obj.position.copy()
        self.jitter(obj, 0, 0)
        np.testing.assert_array_equal(obj.position, initial)
        for w, h in [(-1, 0), (0, -1), (np.nan, 0), (0, np.inf)]:
            with self.subTest(w=w, h=h), self.assertRaises(ValueError):
                self.jitter(Object(), w, h)

    def test_rotation_fallback_is_opt_out(self):
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        fallback = Mock(return_value=[(np.zeros(2), 0.4)])
        rotate = Mock(side_effect=lambda q, angle: q)
        self.ns.update(
            get_xy_contour=Mock(return_value=polygon),
            o3d=SimpleNamespace(
                geometry=SimpleNamespace(PointCloud=SimpleNamespace),
                utility=SimpleNamespace(Vector3dVector=lambda x: x),
            ),
            compute_pcd_bbox=Mock(
                return_value=SimpleNamespace(
                    get_min_bound=lambda: np.zeros(3), get_max_bound=lambda: np.ones(3)
                )
            ),
            find_polygon_placement=Mock(return_value=[]),
            find_fixed_polygon_placement=Mock(return_value=[]),
            find_polygon_placement_with_rotation=fallback,
            rotate_quaternion_z=rotate,
            _dump_polygon_debug=Mock(),
        )
        place = self.ns["randomly_place_object_on_object"]
        pc = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        obj = Object()
        self.assertEqual(
            place(pc, pc, obj, mesh_top_only=True, allow_rotation_fallback=False), -1
        )
        fallback.assert_not_called()
        self.assertEqual(obj.writes, 0)
        self.assertEqual(place(pc, pc, obj, mesh_top_only=True), 0)
        fallback.assert_called_once()
        self.assertEqual(rotate.call_args.args[1], 0.4)
        self.ns["find_polygon_placement"].return_value = [(np.zeros(2), 0)]
        fallback.reset_mock()
        self.assertEqual(
            place(pc, pc, obj, mesh_top_only=True, allow_rotation_fallback=False), 0
        )
        fallback.assert_not_called()
        self.assertEqual(rotate.call_args.args[1], 0)
        self.assertEqual(place(pc, pc, obj, fixed_position=True), -1)


if __name__ == "__main__":
    unittest.main()
