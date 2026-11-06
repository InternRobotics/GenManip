"""Verify position-only anchors using the actual custom-motion resolver."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator
from scipy.spatial.transform import Rotation as R

from genmanip.utils.standalone.transform_utils import pose_frame_to_world


PATH = (
    Path(__file__).resolve().parents[2]
    / "genmanip/extensions/skills/default/custom_motion/custom_motion.py"
)
tree = ast.parse(PATH.read_text())
tree.body = [
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
] + [n for n in tree.body if isinstance(n, ast.ClassDef)]
namespace = dict(
    np=np,
    copy=copy,
    BaseModel=BaseModel,
    Field=Field,
    field_validator=field_validator,
    SkillConfig=BaseModel,
    BaseSkill=object,
    pose_frame_to_world=pose_frame_to_world,
    SkillFactory=SimpleNamespace(register=lambda name: lambda cls: cls),
    adjust_target_pose_by_embodiment=lambda pose, robot: {
        k: np.array(v) for k, v in pose.items()
    },
)
exec(compile(ast.fix_missing_locations(tree), str(PATH), "exec"), namespace)
Motion = namespace["SingleArmMotion"]
Motion.model_rebuild(_types_namespace=namespace)
Skill = namespace["CustomMotionSkill"]


class ObjectPositionTest(unittest.TestCase):
    def resolve(self, mode, yaw):
        rotation = R.from_euler("z", yaw, degrees=True)
        motion = Motion(
            name="place",
            type=mode,
            grasp=True,
            rel_object_uid="beef",
            rel_object_point=[0.02, 0, 0],
            translation=[0, 0, 0.1],
            orientation=[1, 0, 0, 0],
        )
        scene = SimpleNamespace(
            robot_list=[object()],
            object_list={
                "beef": SimpleNamespace(
                    get_world_pose=lambda: (
                        np.array([1.0, 2.0, 3.0]),
                        rotation.as_quat()[[3, 0, 1, 2]],
                    )
                )
            },
        )
        skill = object.__new__(Skill)
        skill.config = SimpleNamespace(motion_list={"right": [motion]})
        skill._process_motion_target(scene)
        return motion

    def test_rotated_anchor_world_offset_fixed_orientation(self):
        motion = self.resolve("object_position", 90)
        np.testing.assert_allclose(motion.translation, [1, 2.02, 3.1])
        np.testing.assert_allclose(motion.orientation, [1, 0, 0, 0])

    def test_original_object_frame_still_rotates_orientation(self):
        motion = self.resolve("object_frame", 90)
        np.testing.assert_allclose(motion.translation, [1, 2, 3.1])
        np.testing.assert_allclose(motion.orientation, [2**-0.5, 0, 0, 2**-0.5])

    def test_invalid_anchor(self):
        for point in ([1, 2], [0, np.nan, 0]):
            with self.assertRaises(ValidationError):
                Motion(
                    name="bad",
                    type="object_position",
                    grasp=False,
                    rel_object_point=point,
                )


if __name__ == "__main__":
    unittest.main()
