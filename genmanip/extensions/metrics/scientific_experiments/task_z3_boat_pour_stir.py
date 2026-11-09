from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from pxr import UsdGeom  # type: ignore

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory
from genmanip.extensions.metrics.default.sr_based_genmanip_relationship import (
    SRBasedGenmanipRelationship,
    is_inside,
)


LIQUID_IN_BEAKER_THRESHOLD = 0.80
GLASS_ROD_IN_BEAKER_THRESHOLD = 0.1


class BoatPourStirConfig(BaseModel):
    beaker_uid: str = Field(..., description="Prim name or path of the beaker object")
    glass_rod_uid: str = Field(
        ..., description="Prim name or path of the graduated glass rod object"
    )
    gc_liquid_mesh_prim_path: str = Field(
        ..., description="Prim path of the liquid mesh in graduated cylinder object"
    )
    condition_type: str = Field(..., description="condition type")


@MetricFactory.register("manip/scientific_experiments/task_z3_boat_pour_stir")
class BoatPourStir(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = BoatPourStirConfig(**sub_goal_setting)

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

        if self.setting.condition_type == "graduated_cylinder_liquid_in_beaker":
            if beaker_vertices is None:
                return False

            return is_inside(
                src_pts=gc_liquid_particles,
                target_pts=beaker_vertices,
                thresh=LIQUID_IN_BEAKER_THRESHOLD,
                scene=scene,
            )

        if self.setting.condition_type == "glass_rod_in_beaker":
            if glass_rod_vertices is None or beaker_vertices is None:
                return False

            return is_inside(
                src_pts=glass_rod_vertices,
                target_pts=beaker_vertices,
                thresh=GLASS_ROD_IN_BEAKER_THRESHOLD,
                scene=scene,
            )

        return False
