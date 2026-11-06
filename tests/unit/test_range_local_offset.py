"""Simulator-free tests of the actual range metric's optional measured point."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator
from scipy.spatial.transform import Rotation as R


class Base:
    def __init__(self, *args, **kwargs):
        pass


PATH = (
    Path(__file__).resolve().parents[2]
    / "genmanip/extensions/metrics/default/sr_based_genmanip_range.py"
)
tree = ast.parse(PATH.read_text())
tree.body = [
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
] + [node for node in tree.body if isinstance(node, ast.ClassDef)]
namespace = dict(
    np=np,
    R=R,
    BaseModel=BaseModel,
    Field=Field,
    field_validator=field_validator,
    BaseMetric=Base,
    MetricFactory=SimpleNamespace(register=lambda name: lambda cls: cls),
)
exec(compile(ast.fix_missing_locations(tree), str(PATH), "exec"), namespace)
Metric = namespace["SRBasedGenmanipRange"]
Config = namespace["SRBasedGenmanipRangeConfig"]


class RangeLocalOffsetTest(unittest.TestCase):
    def setUp(self):
        self.position = np.array([2.0, 3.0, 5.0])
        self.orientation = R.from_euler("y", 90, degrees=True).as_quat()[[3, 0, 1, 2]]
        self.scene = SimpleNamespace(
            object_list={
                "object": SimpleNamespace(
                    get_world_pose=lambda: (self.position, self.orientation)
                ),
                "target": SimpleNamespace(
                    get_world_pose=lambda: (
                        np.array([2.0, 3.0, 4.0]),
                        np.array([1.0, 0.0, 0.0, 0.0]),
                    )
                ),
            },
            articulation_list={},
        )
        self.config = dict(
            obj1_uid="object",
            x_type="relative",
            y_type="relative",
            z_type="relative",
            x_rel_object_uid="target",
            y_rel_object_uid="target",
            z_rel_object_uid="target",
            x_range=[-0.01, 0.01],
            y_range=[-0.01, 0.01],
            z_range=[-0.01, 0.01],
        )

    def test_default_still_measures_origin(self):
        self.assertEqual(Config(**self.config).obj1_local_offset, (0.0, 0.0, 0.0))
        metric = Metric(sub_goal_setting=self.config)
        self.assertFalse(metric.check_status(self.scene))
        self.position[2] = 4
        self.assertTrue(metric.check_status(self.scene))

    def test_rotated_offset_uses_meters_and_does_not_mutate_pose(self):
        metric = Metric(
            sub_goal_setting={**self.config, "obj1_local_offset": [1, 0, 0]}
        )
        self.assertTrue(metric.check_status(self.scene))
        np.testing.assert_array_equal(self.position, [2, 3, 5])
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self.assertFalse(metric.check_status(self.scene))

    def test_invalid_offsets(self):
        for offset in ([1, 2], [1, 2, 3, 4], [0, np.nan, 0], [np.inf, 0, 0]):
            with self.subTest(offset=offset), self.assertRaises(ValidationError):
                Config(**self.config, obj1_local_offset=offset)


if __name__ == "__main__":
    unittest.main()
