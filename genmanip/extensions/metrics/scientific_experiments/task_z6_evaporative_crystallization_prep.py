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


LIQUID_IN_CONICAL_FLASK_THRESHOLD = 0.8


class EvaporativeCrystallizationPrepConfig(BaseModel):
    evaporating_dish_deep_uid: str = Field(
        ..., description="Prim name or path of the evaporating_dish_deep object"
    )
    b_liquid_mesh_prim_path: str = Field(
        ..., description="Prim path of the liquid mesh in beaker object"
    )
    status_prim_path: str = Field(..., description="Prim path of the status")
    condition_type: str = Field(..., description="condition type")


@MetricFactory.register(
    "manip/scientific_experiments/task_z6_evaporative_crystallization_prep"
)
class EvaporativeCrystallizationPrep(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = EvaporativeCrystallizationPrepConfig(**sub_goal_setting)

    def check_status(self, scene):
        if self.setting.condition_type == "align_alcohol_lamp":
            status_prim = scene.world.stage.GetPrimAtPath(self.setting.status_prim_path)
            state = status_prim.GetAttribute("heating:wickUnderRing").Get()
            return state

        if self.setting.condition_type == "place_evaporating_dish_deep":
            status_prim = scene.world.stage.GetPrimAtPath(self.setting.status_prim_path)
            state = status_prim.GetAttribute("heating:dishOnRing").Get()
            return state

        if self.setting.condition_type == "b_liquid_in_dish":
            pclist = SRBasedGenmanipRelationship.get_target_pc_list(
                scene, [self.setting.evaporating_dish_deep_uid]
            )
            dish_vertices = pclist.get(self.setting.evaporating_dish_deep_uid)

            if dish_vertices is None:
                return False

            liquid_prim = scene.world.stage.GetPrimAtPath(
                self.setting.b_liquid_mesh_prim_path
            )
            liquid_points = liquid_prim.GetAttribute("points").Get()
            liquid_world_transform = UsdGeom.XformCache().GetLocalToWorldTransform(
                liquid_prim
            )
            liquid_particles = np.asarray(
                [liquid_world_transform.Transform(point) for point in liquid_points],
                dtype=np.float64,
            )

            return is_inside(
                src_pts=liquid_particles,
                target_pts=dish_vertices,
                thresh=LIQUID_IN_CONICAL_FLASK_THRESHOLD,
                scene=scene,
            )

        return False
