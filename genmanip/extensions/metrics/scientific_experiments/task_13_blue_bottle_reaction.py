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
BLUE_FRACTION_THRESHOLD = 0.9
COLORLESS_THRESHOLD = 0.1


class BlueBottleReactionConfig(BaseModel):
    conical_flask_uid: str = Field(
        ..., description="Prim name or path of the conical flask object"
    )
    gc_1_liquid_mesh_prim_path: str = Field(
        ..., description="Prim path of the liquid mesh in graduated_cylinder_01 object"
    )
    gc_2_liquid_mesh_prim_path: str = Field(
        ..., description="Prim path of the liquid mesh in graduated_cylinder_02 object"
    )
    flow_controller_prim_path: str = Field(
        ..., description="Prim path of the flow_controller"
    )
    condition_type: str = Field(..., description="condition type")


@MetricFactory.register("manip/scientific_experiments/task_13_blue_bottle_reaction")
class BlueBottleReaction(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = BlueBottleReactionConfig(**sub_goal_setting)

    def check_status(self, scene):
        if self.setting.condition_type in [
            "gc_1_liquid_in_conical_flask",
            "gc_2_liquid_in_conical_flask",
        ]:
            pclist = SRBasedGenmanipRelationship.get_target_pc_list(
                scene, [self.setting.conical_flask_uid]
            )
            conical_flask_vertices = pclist.get(self.setting.conical_flask_uid)

            if conical_flask_vertices is None:
                return False

            gc_liquid_mesh_prim_path = (
                self.setting.gc_1_liquid_mesh_prim_path
                if self.setting.condition_type == "gc_1_liquid_in_conical_flask"
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
                target_pts=conical_flask_vertices,
                thresh=GC_LIQUID_IN_CONICAL_FLASK_THRESHOLD,
                scene=scene,
            )

        flow_controller_station_prim = scene.world.stage.GetPrimAtPath(
            self.setting.flow_controller_prim_path
        )

        if self.setting.condition_type == "glass_rod_get_dye":
            valve_state = flow_controller_station_prim.GetAttribute(
                "blueBottle:rodLoaded"
            ).Get()
            return valve_state

        if self.setting.condition_type == "is_blue":
            valve_state = flow_controller_station_prim.GetAttribute(
                "blueBottle:blueFraction"
            ).Get()
            return valve_state >= BLUE_FRACTION_THRESHOLD

        if self.setting.condition_type == "is_colorless":
            valve_state = flow_controller_station_prim.GetAttribute(
                "blueBottle:blueFraction"
            ).Get()
            return valve_state <= COLORLESS_THRESHOLD

        return False
