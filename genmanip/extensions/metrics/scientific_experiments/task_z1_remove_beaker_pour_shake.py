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


GC_LIQUID_IN_CONICAL_FLASK_THRESHOLD = 0.6


class RemoveBeakerPourShakeConfig(BaseModel):
    oven_body_uid: str = Field(..., description="UID of the oven's body")
    oven_door_uid: str = Field(..., description="UID of the oven's door")
    beaker_uid: str = Field(..., description="UID of the beaker")
    gc_1_liquid_mesh_prim_path: str = Field(
        ..., description="Prim path of the liquid mesh in graduated_cylinder_01 object"
    )
    gc_2_liquid_mesh_prim_path: str = Field(
        ..., description="Prim path of the liquid mesh in graduated_cylinder_02 object"
    )
    shake_detector_prim_path: str = Field(
        ..., description="Prim path of the shake_detector"
    )
    condition_type: str = Field(..., description="condition type")


@MetricFactory.register("manip/scientific_experiments/task_z1_remove_beaker_pour_shake")
class RemoveBeakerPourShake(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = RemoveBeakerPourShakeConfig(**sub_goal_setting)

    def check_status(self, scene):
        if self.setting.condition_type == "beaker_not_in_oven":
            target_uids = [
                self.setting.oven_body_uid,
                self.setting.oven_door_uid,
                self.setting.beaker_uid,
            ]
            pclist = SRBasedGenmanipRelationship.get_target_pc_list(scene, target_uids)

            oven_vertices = np.concatenate(
                [
                    pclist[self.setting.oven_body_uid],
                    pclist[self.setting.oven_door_uid],
                ],
                axis=0,
            )
            beaker_vertices = pclist.get(self.setting.beaker_uid)

            return not is_inside(
                src_pts=beaker_vertices,
                target_pts=oven_vertices,
                thresh=0.9,
                scene=scene,
            )

        if self.setting.condition_type in [
            "gc_1_liquid_in_beaker",
            "gc_2_liquid_in_beaker",
        ]:
            pclist = SRBasedGenmanipRelationship.get_target_pc_list(
                scene, [self.setting.beaker_uid]
            )
            beaker_vertices = pclist.get(self.setting.beaker_uid)

            if beaker_vertices is None:
                return False

            gc_liquid_mesh_prim_path = (
                self.setting.gc_1_liquid_mesh_prim_path
                if self.setting.condition_type == "gc_1_liquid_in_beaker"
                else self.setting.gc_2_liquid_mesh_prim_path
            )

            gc_liquid_prim = scene.world.stage.GetPrimAtPath(gc_liquid_mesh_prim_path)
            gc_liquid_points = gc_liquid_prim.GetAttribute("points").Get()
            gc_liquid_world_transform = UsdGeom.XformCache().GetLocalToWorldTransform(
                gc_liquid_prim
            )
            gc_liquid_particles = np.asarray(
                [
                    gc_liquid_world_transform.Transform(point)
                    for point in gc_liquid_points
                ],
                dtype=np.float64,
            )

            return is_inside(
                src_pts=gc_liquid_particles,
                target_pts=beaker_vertices,
                thresh=GC_LIQUID_IN_CONICAL_FLASK_THRESHOLD,
                scene=scene,
            )

        if self.setting.condition_type == "has_shaken":
            shake_detector_prim = scene.world.stage.GetPrimAtPath(
                self.setting.shake_detector_prim_path
            )
            valve_state = shake_detector_prim.GetAttribute("mixing:hasShaken").Get()
            return valve_state

        return False
