"""Regression for PhysX quaternion hemisphere flips in placement validation."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

path = (
    Path(__file__).resolve().parents[2]
    / "genmanip/demogen/random_place/random_place.py"
)
tree = ast.parse(path.read_text())
tree.body = [
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
] + [
    n
    for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "verify_placement"
]
namespace = {"np": np}
exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)


class PlacementTest(unittest.TestCase):
    def verify(self, before, after, displacement=(0, 0, 0)):
        poses = iter(
            [(np.zeros(3), np.array(before)), (np.array(displacement), np.array(after))]
        )
        return namespace["verify_placement"](
            SimpleNamespace(get_world_pose=lambda: next(poses)),
            SimpleNamespace(step=lambda **kw: None),
        )

    def test_same_rotation_with_opposite_sign(self):
        self.assertTrue(self.verify([0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, 0.5]))

    def test_small_rotation_in_both_hemispheres(self):
        for sign in (-1, 1):
            self.assertTrue(
                self.verify(
                    [1, 0, 0, 0], np.array([np.cos(0.02), np.sin(0.02), 0, 0]) * sign
                )
            )

    def test_large_rotation_still_rejected(self):
        self.assertFalse(self.verify([1, 0, 0, 0], [0, 1, 0, 0]))

    def test_translation_still_rejected(self):
        self.assertFalse(self.verify([1, 0, 0, 0], [-1, 0, 0, 0], (0.11, 0, 0)))


if __name__ == "__main__":
    unittest.main()
