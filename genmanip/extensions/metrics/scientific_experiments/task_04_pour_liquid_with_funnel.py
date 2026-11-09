from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from pxr import UsdGeom  # type: ignore

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory
from genmanip.extensions.metrics.default.sr_based_genmanip_relationship import (
    SRBasedGenmanipRelationship,
    calculate_distance_between_two_point_clouds,
    is_inside,
)
from genmanip.extensions.metrics.default.match_local_world_orientation import (
    MatchLocalWorldOrientation,
    OrientationConstraintSpec,
)


CONTACT_TOLERANCE = 0.005
FUNNEL_UPRIGHT_MAX_ANGLE = 20
FUNNEL_BOTTLE_CENTER_TOLERANCE = 0.01
LIQUID_IN_BOTTLE_THRESHOLD = 0.80


class PourLiquidWithFunnelConfig(BaseModel):
    funnel_uid: str = Field(..., description="Prim name or path of the funnel object")
    funnel_rack_uid: str = Field(
        ..., description="Prim name or path of the funnel rack object"
    )
    target_reagent_bottle_uid: str = Field(
        ..., description="Prim name or path of the target reagent bottle object"
    )
    gc_liquid_mesh_prim_path: str = Field(
        ..., description="Prim path of the liquid mesh in graduated cylinder object"
    )
    funnel_constraint_spec: OrientationConstraintSpec = Field(
        ..., description="Orientation matching constraints"
    )
    condition_type: str = Field(..., description="condition type")


@MetricFactory.register("manip/scientific_experiments/task_04_pour_liquid_with_funnel")
class PourLiquidWithFunnel(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = PourLiquidWithFunnelConfig(**sub_goal_setting)
        self.funnel_orientation = MatchLocalWorldOrientation(
            sub_goal_setting={
                "obj_uid": self.setting.funnel_uid,
                "constraint_spec": self.setting.funnel_constraint_spec,
            }
        )

    def check_status(self, scene):
        ids = [
            self.setting.funnel_uid,
            self.setting.funnel_rack_uid,
            self.setting.target_reagent_bottle_uid,
        ]
        pcs = SRBasedGenmanipRelationship.get_target_pc_list(scene, ids)
        funnel, rack, bottle = (pcs.get(i) for i in ids)

        if self.setting.condition_type == "timing_1":
            if funnel is None or rack is None or not len(funnel) or not len(rack):
                return False
            distance = calculate_distance_between_two_point_clouds(
                funnel, rack, scene=scene
            )
            # Leaving the rack only counts while the funnel stays near the
            # upright pose it had at spawn (local +Z within the configured angle).
            return (
                distance > CONTACT_TOLERANCE
                and self.funnel_orientation.check_status(scene)
            )

        if self.setting.condition_type == "timing_2":
            if (
                funnel is None
                or bottle is None
                or not len(funnel)
                or not len(bottle)
                or not self.funnel_orientation.check_status(scene)
            ):
                return False
            if not bottle[:, 2].min() <= funnel[:, 2].min() <= bottle[:, 2].max():
                return False
            center_distance = np.linalg.norm(
                funnel[:, :2].mean(0) - bottle[:, :2].mean(0)
            )
            if center_distance >= FUNNEL_BOTTLE_CENTER_TOLERANCE:
                return False
            distance = calculate_distance_between_two_point_clouds(
                funnel, bottle, scene=scene
            )
            return distance < CONTACT_TOLERANCE

        if self.setting.condition_type == "timing_3":
            if bottle is None or not len(bottle):
                return False
            prim = scene.world.stage.GetPrimAtPath(
                self.setting.gc_liquid_mesh_prim_path
            )
            if not prim or not prim.IsValid():
                return False
            points = prim.GetAttribute("points").Get()
            transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            liquid = np.asarray(
                [transform.Transform(point) for point in points], dtype=np.float64
            )
            return len(liquid) > 0 and is_inside(
                liquid,
                bottle,
                thresh=LIQUID_IN_BOTTLE_THRESHOLD,
                scene=scene,
            )

        return False
