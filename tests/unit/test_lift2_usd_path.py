"""Exercise actual config and robot creation methods without starting Kit."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import yaml

from genmanip.core.scene.scene_config import RobotConfig, SceneConfig

ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_PATH = "robot_usds/lift2_scientific_experiments/robot.usd"


class Lift2UsdPathTest(unittest.TestCase):
    def create(self, **kwargs):
        cfg = RobotConfig(
            type="manip/lift2/R5a",
            position=[0.0, -0.7, 0.0],
            orientation=[1.0, 0.0, 0.0, 0.0],
            **kwargs,
        )
        source = ROOT / "genmanip/extensions/robots/default/r5a/lift2.py"
        body = next(
            n
            for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "Lift2Embodiment"
        )
        method = next(
            n
            for n in body.body
            if isinstance(n, ast.FunctionDef) and n.name == "create_robot"
        )
        robot = Mock(prim_path="/World/test/lift2")
        create = Mock()
        namespace = {
            "os": os,
            "np": np,
            "RobotConfig": RobotConfig,
            "Robot": Mock(return_value=robot),
            "get_prim_at_path": lambda _: SimpleNamespace(IsValid=lambda: False),
            "create_prim": create,
            "Usd": SimpleNamespace(PrimRange=lambda _: []),
        }
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        namespace["create_robot"](
            SimpleNamespace(), "test", {"ASSETS_DIR": "/assets"}, cfg
        )
        return cfg, create.call_args.kwargs, robot

    def test_config_retains_explicit_path(self):
        cfg = RobotConfig(type="manip/lift2/R5a", usd_path=SCIENTIFIC_PATH)
        self.assertEqual(cfg.usd_path, SCIENTIFIC_PATH)
        self.assertEqual(cfg.model_dump()["usd_path"], SCIENTIFIC_PATH)

    def test_scene_config_retains_yaml_path(self):
        source = (
            ROOT
            / "configs/tasks/scientific_experiments/task_11_remove_container_from_oven/validation/unseen_prospect/unseen_prospect_2.yml"
        )
        data = yaml.safe_load(source.read_text())
        config_data = data["demonstration_configs"][0]
        config_data.pop("generation_config", None)
        self.assertEqual(config_data["robots"][0]["usd_path"], SCIENTIFIC_PATH)
        self.assertEqual(
            data["evaluation_configs"][0]["robots"][0]["usd_path"],
            SCIENTIFIC_PATH,
        )
        cfg = SceneConfig(**config_data)
        self.assertEqual(cfg.robots[0].usd_path, SCIENTIFIC_PATH)

    def test_explicit_scientific_asset_reaches_create_prim(self):
        _, args, _ = self.create(usd_path=SCIENTIFIC_PATH)
        self.assertEqual(args["usd_path"], f"/assets/{SCIENTIFIC_PATH}")

    def test_legacy_none_and_empty_preserve_default(self):
        for kwargs in ({}, {"usd_path": None}, {"usd_path": ""}):
            with self.subTest(kwargs=kwargs):
                _, args, _ = self.create(**kwargs)
                self.assertEqual(args["usd_path"], "/assets/robot_usds/lift2/robot.usd")

    def test_pose_and_solver_requests_are_preserved(self):
        _, args, robot = self.create(usd_path=SCIENTIFIC_PATH)
        np.testing.assert_array_equal(args["position"], [0.0, -0.7, 0.0])
        np.testing.assert_array_equal(args["orientation"], [1.0, 0.0, 0.0, 0.0])
        robot.set_solver_position_iteration_count.assert_called_once_with(128)
        robot.set_solver_velocity_iteration_count.assert_called_once_with(4)


if __name__ == "__main__":
    unittest.main()
