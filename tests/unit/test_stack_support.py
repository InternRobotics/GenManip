"""Simulator-free regression tests using the actual relationship implementation."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy
from scipy.spatial import QhullError
from shapely.geometry import Polygon
from shapely.vectorized import contains
from pydantic import BaseModel, Field, ValidationError
from typing import Literal
from scipy.spatial.transform import Rotation


PATH = (
    Path(__file__).resolve().parents[2]
    / "genmanip/extensions/metrics/default/sr_based_genmanip_relationship.py"
)
tree = ast.parse(PATH.read_text())
tree.body = [
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
] + [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Assign))]
namespace = dict(
    np=np,
    scipy=scipy,
    QhullError=QhullError,
    Polygon=Polygon,
    contains=contains,
    BaseModel=BaseModel,
    Field=Field,
    Literal=Literal,
    BaseMetric=object,
    MetricFactory=SimpleNamespace(register=lambda name: lambda cls: cls),
)
exec(compile(ast.fix_missing_locations(tree), str(PATH), "exec"), namespace)
support = namespace["support_contact_diagnostics"]
Config = namespace["SRBasedGenmanipRelationshipConfig"]
Config.model_rebuild(_types_namespace=namespace)


def box(width, depth, zmin, zmax, x=0, y=0):
    return np.array(
        [
            [xx, yy, zz]
            for xx in np.linspace(x - width / 2, x + width / 2, 41)
            for yy in np.linspace(y - depth / 2, y + depth / 2, 41)
            for zz in (zmin, zmax)
        ]
    )


class StackSupportTest(unittest.TestCase):
    def setUp(self):
        self.steak = box(0.16, 0.12, 0, 0.03)
        self.leaf = box(0.24, 0.20, 0.031, 0.045)

    def test_large_leaf_supported_despite_legacy_area_rejection(self):
        self.assertTrue(support(self.leaf, self.steak)["supported"])
        legacy = namespace["_check_support_relationship"](
            self.leaf.min(0),
            self.leaf.max(0),
            self.steak.min(0),
            self.steak.max(0),
            0.01,
        )
        self.assertNotIn("on", legacy)

    def test_wrong_order_beside_floating_and_interpenetrating_fail(self):
        cases = [
            (self.steak, self.leaf),
            (self.leaf + [0.22, 0, 0], self.steak),
            (self.leaf + [0, 0, 0.08], self.steak),
            (self.leaf - [0, 0, 0.025], self.steak),
        ]
        for upper, lower in cases:
            with self.subTest(center=upper.mean(0).tolist()):
                self.assertFalse(support(upper, lower)["supported"])

    def test_in_plane_rotation_does_not_invalidate_an_overhanging_leaf(self):
        leaf = box(0.20, 0.18, 0.031, 0.045)
        for angle in np.deg2rad([0, 15, 30, 45, 60, 75, 90]):
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            rotated = leaf.copy()
            rotated[:, :2] = rotated[:, :2] @ rotation.T
            with self.subTest(angle=angle):
                self.assertTrue(support(rotated, self.steak)["supported"])

    def test_remote_plate_rim_does_not_define_support_height(self):
        bed = box(0.3, 0.3, 0, 0.01)
        rim = np.array(
            [
                [x, y, z]
                for x in (-0.15, 0.15)
                for y in np.linspace(-0.15, 0.15, 20)
                for z in (0.01, 0.06)
            ]
        )
        self.assertTrue(
            support(box(0.16, 0.12, 0.011, 0.04), np.vstack([bed, rim]))["supported"]
        )

    def test_invalid_and_degenerate_clouds_fail(self):
        for points in (
            np.empty((0, 3)),
            np.zeros((8, 3)),
            np.ones((5, 2)),
            np.full((5, 3), np.nan),
        ):
            self.assertFalse(support(points, self.steak)["supported"])

    def test_tilted_stack_compares_matching_surface_columns(self):
        rotation = Rotation.from_euler("x", 12, degrees=True)
        self.assertTrue(
            support(rotation.apply(self.leaf), rotation.apply(self.steak))["supported"]
        )

    def test_container_rejects_stack_outside_plate(self):
        plate = box(0.32, 0.32, -0.01, 0)
        self.assertTrue(support(self.leaf, self.steak, container=plate)["supported"])
        self.assertFalse(
            support(
                self.leaf + [0.14, 0, 0], self.steak + [0.14, 0, 0], container=plate
            )["supported"]
        )

    def test_opt_in_and_validation(self):
        self.assertEqual(Config(obj1_uid="lettuce").support_mode, "legacy")
        for option in (
            {"support_mode": "unknown"},
            {"support_min_overlap": 0},
            {"support_min_overlap": 1.1},
            {"support_max_gap": np.inf},
        ):
            with self.assertRaises(ValidationError):
                Config(obj1_uid="lettuce", **option)

    def test_contact_goal_dispatch_and_legacy_default(self):
        cls = namespace["SRBasedGenmanipRelationship"]
        metric = object.__new__(cls)
        scene = SimpleNamespace(articulation_list={})
        pcs = {"leaf": self.leaf, "steak": self.steak}
        with patch.object(cls, "get_target_pc_list", return_value=pcs):
            metric.goal_setting = Config(
                obj1_uid="leaf",
                obj2_uid="steak",
                position="top",
                support_mode="contact",
            )
            self.assertTrue(metric.check_status(scene))
            pcs["plate"] = box(0.32, 0.32, -0.01, 0)
            metric.goal_setting = Config(
                obj1_uid="leaf",
                obj2_uid="steak",
                position="top",
                support_mode="contact",
                support_container_uid="plate",
            )
            self.assertTrue(metric.check_status(scene))
            pcs["plate"] += [0.5, 0, 0]
            self.assertFalse(metric.check_status(scene))
            metric.goal_setting = Config(
                obj1_uid="leaf", obj2_uid="steak", position="top"
            )
            with patch.dict(
                namespace, check_subgoal_finished_rigid=lambda *a, **kw: False
            ):
                self.assertFalse(metric.check_status(scene))


if __name__ == "__main__":
    unittest.main()
