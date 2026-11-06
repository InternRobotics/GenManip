"""Check collider target selection without starting Isaac Sim."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class ColliderPreprocessTest(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[2] / "genmanip/utils/loader/scene.py"
        tree = ast.parse(source.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "preprocess_scene"
        )
        self.collider = Mock()
        namespace = {"SceneConfig": object, "set_colliders": self.collider}
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        self.preprocess = namespace["preprocess_scene"]
        self.scene = SimpleNamespace(
            object_list={
                name: SimpleNamespace(prim_path=f"/World/{name}")
                for name in (
                    "bowl",
                    "beef",
                    "defaultGroundPlane",
                    "articulation",
                    "part",
                    "00000000000000000000000000000000",
                )
            },
            articulation_list={"articulation": object()},
            articulation_part_list={"part": object()},
        )

    def run_config(self, **kwargs):
        self.preprocess(
            self.scene,
            SimpleNamespace(
                preprocess_config=[
                    {"type": "collider", "config": {"type": "none", **kwargs}}
                ]
            ),
        )

    def test_targeted_bowl_leaves_other_objects_untouched(self):
        self.run_config(uids=["bowl"])
        self.collider.assert_called_once_with(
            "/World/bowl", collision_approximation="none"
        )

    def test_omitted_targets_preserve_global_behavior(self):
        self.run_config()
        self.assertEqual(self.collider.call_count, 3)
        self.collider.assert_any_call(
            "/World/00000000000000000000000000000000",
            collision_approximation="none",
            convex_hulls=2048,
        )

    def test_empty_targets_touch_nothing(self):
        self.run_config(uids=[])
        self.collider.assert_not_called()

    def test_unknown_target_fails_before_mutation(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            self.run_config(uids=["bowl", "missing"])
        self.collider.assert_not_called()

    def test_invalid_targets_fail_before_mutation(self):
        for targets in ("bowl", [1], {}):
            with self.subTest(targets=targets):
                with self.assertRaisesRegex(ValueError, "list of strings"):
                    self.run_config(uids=targets)
                self.collider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
