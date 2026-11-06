"""Base step limits and resulting motion, without an Isaac Sim process."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from pydantic import BaseModel, Field, ValidationError


SOURCE = (
    Path(__file__).resolve().parents[2]
    / "genmanip/extensions/skills/default/move_to/move_to.py"
)
tree = ast.parse(SOURCE.read_text())
tree.body = [
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
] + [node for node in tree.body if isinstance(node, ast.ClassDef)]
namespace = dict(
    np=np,
    os=os,
    Field=Field,
    SkillConfig=BaseModel,
    BaseSkill=object,
    SkillFactory=SimpleNamespace(register=lambda name: lambda cls: cls),
    tqdm=lambda items, **kwargs: items,
)
exec(compile(ast.fix_missing_locations(tree), str(SOURCE), "exec"), namespace)
Config = namespace["MoveToConfig"]
Skill = namespace["MoveToSkill"]


class MoveToTest(unittest.TestCase):
    def run_motion(self, delta, **options):
        skill = Skill(dict(move_type="delta", **options), {})
        embodiment = SimpleNamespace(
            robot=SimpleNamespace(get_joint_positions=lambda: np.zeros(6)),
            default_dof_indices=np.arange(6),
            robot_view=Mock(),
            delta_move_to=Mock(),
        )
        scene, recorder = Mock(), Mock()
        with patch.dict(os.environ, {"GENMANIP_DEBUG": "0"}):
            skill._handle_dual_arm_base_move(scene, recorder, embodiment, "test", delta)
        moves = np.array(
            [call.args for call in embodiment.delta_move_to.call_args_list]
        )
        self.assertEqual(scene.step.call_count, len(moves))
        self.assertEqual(recorder.load_dynamic_info.call_count, len(moves))
        return moves

    def test_default_step_and_total_displacement(self):
        self.assertEqual(Config(move_type="delta").linear_step, 0.01)
        moves = self.run_motion(dict(x=0.025, y=-0.0125, yaw=0))
        np.testing.assert_allclose(moves[0], [0.01, -0.005, 0])
        np.testing.assert_allclose(moves.sum(axis=0), [0.025, -0.0125, 0])
        self.assertLessEqual(np.abs(moves[:, :2]).max(), 0.01)

    def test_slow_step_preserves_direction_and_destination(self):
        moves = self.run_motion(dict(x=-0.025, y=0.0125, yaw=None), linear_step=0.002)
        np.testing.assert_allclose(moves[:, 1], -moves[:, 0] / 2)
        np.testing.assert_allclose(moves.sum(axis=0), [-0.025, 0.0125, 0])
        self.assertLessEqual(np.abs(moves[:, :2]).max(), 0.002)

    def test_yaw_only_is_unchanged(self):
        for step in (0.002, 0.01):
            with self.subTest(step=step):
                moves = self.run_motion(dict(x=0, y=0, yaw=-2.5), linear_step=step)
                np.testing.assert_allclose(
                    moves, [[0, 0, -1], [0, 0, -1], [0, 0, -0.5]]
                )

    def test_zero_motion(self):
        self.assertEqual(len(self.run_motion(dict(x=0, y=0, yaw=0))), 0)

    def test_step_validation(self):
        self.assertEqual(
            Config(move_type="delta", linear_step=0.015).linear_step, 0.015
        )
        for value in (0, -0.001, 0.0151, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                Config(move_type="delta", linear_step=value)


if __name__ == "__main__":
    unittest.main()
