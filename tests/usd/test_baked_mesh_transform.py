"""USD-only regression tests; see tests/README.md for setup."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from pxr import Gf, Usd, UsdGeom


PATH = (
    Path(__file__).resolve().parents[2] / "genmanip/utils/usd_utils/transform_utils.py"
)
SPEC = importlib.util.spec_from_file_location("salad_transform_test", PATH)
TRANSFORMS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRANSFORMS
SPEC.loader.exec_module(TRANSFORMS)


class BakedMeshTransformTest(unittest.TestCase):
    def test_empty_order_ignores_inherited_ops(self):
        source = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(source, "/Mesh")
        mesh.AddTranslateOp().Set(Gf.Vec3d(3, 4, 5))
        mesh.AddScaleOp().Set(Gf.Vec3f(0.01))
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().subLayerPaths.append(source.GetRootLayer().identifier)
        baked = UsdGeom.Mesh(stage.GetPrimAtPath("/Mesh"))
        baked.ClearXformOpOrder()
        self.assertIsNotNone(baked.GetPrim().GetAttribute("xformOp:scale").Get())
        actual = TRANSFORMS.resolve_prim_local_transform(baked.GetPrim())
        points = [np.array([0.02, 0.03, 0.04])]
        np.testing.assert_allclose(
            TRANSFORMS.apply_local_transform_to_points(points, actual), points
        )
        np.testing.assert_allclose(baked.GetLocalTransformation(), np.eye(4))

    def test_active_trs_unchanged(self):
        stage = Usd.Stage.CreateInMemory()
        prim = UsdGeom.Xform.Define(stage, "/Object")
        prim.AddTranslateOp().Set(Gf.Vec3d(3, 4, 5))
        prim.AddScaleOp().Set(Gf.Vec3f(0.01))
        actual = TRANSFORMS.resolve_prim_local_transform(prim.GetPrim())
        np.testing.assert_allclose(actual.translation, [3, 4, 5])
        np.testing.assert_allclose(actual.scale, [0.01, 0.01, 0.01])


if __name__ == "__main__":
    unittest.main()
