from typing import Any

import math
import re
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


CONTACT_TOLERANCE = 0.05
WEIGHING_BOAT_THRESHOLD = 8.0
TARE_THRESHOLD = 0.02
SPOON_IN_BEAKER_THRESHOLD = 0.1


class FixedWeightWeighingConfig(BaseModel):
    sampling_spoon_uid: str = Field(
        ..., description="Prim name or path of the spoon object"
    )
    beaker_uid: str = Field(..., description="Prim name or path of the beaker object")
    weighing_boat_uid: str = Field(
        ..., description="Prim name or path of the weighing boat object"
    )
    target_weighting: float = Field(..., description="target weighting")
    weighting_error: float = Field(..., description="weighting error")
    analytical_balance_WeighingPan_uid: str = Field(
        ...,
        description="Prim name or path of the analytical_balance_WeighingPan object",
    )
    analytical_balance_prim_path: str = Field(
        ..., description="Prim path of the analytical balance"
    )
    condition_type: str = Field(..., description="condition type")


@MetricFactory.register("manip/scientific_experiments/task_z5_fixed_weight_weighing")
class FixedWeightWeighing(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = FixedWeightWeighingConfig(**sub_goal_setting)

    def check_status(self, scene):
        target_uids = [
            self.setting.sampling_spoon_uid,
            self.setting.beaker_uid,
            self.setting.weighing_boat_uid,
            self.setting.analytical_balance_WeighingPan_uid,
        ]
        pclist = SRBasedGenmanipRelationship.get_target_pc_list(scene, target_uids)
        sampling_spoon_vertices = pclist.get(self.setting.sampling_spoon_uid)
        beaker_vertices = pclist.get(self.setting.beaker_uid)
        weighing_boat_vertices = pclist.get(self.setting.weighing_boat_uid)
        analytical_balance_vertices = pclist.get(
            self.setting.analytical_balance_WeighingPan_uid
        )

        analytical_balance_prim = scene.world.stage.GetPrimAtPath(
            self.setting.analytical_balance_prim_path
        )
        lcd_readout = analytical_balance_prim.GetAttribute("balance:lcd_readout").Get()
        balance_g = self.parse_balance_g(lcd_readout)

        if self.setting.condition_type in [
            "weighing_weighing_boat",
            "check_tare",
            "weighting",
            "numerical_check",
        ]:
            if (
                weighing_boat_vertices is None
                or analytical_balance_vertices is None
                or len(weighing_boat_vertices) == 0
                or len(analytical_balance_vertices) == 0
                or balance_g is None
            ):
                return False

            is_above_balance = (
                weighing_boat_vertices[:, 2].min()
                >= analytical_balance_vertices[:, 2].max() - CONTACT_TOLERANCE
            )
            distance = calculate_distance_between_two_point_clouds(
                weighing_boat_vertices,
                analytical_balance_vertices,
                scene=scene,
            )

            if not (is_above_balance and distance <= CONTACT_TOLERANCE):
                return False

            if self.setting.condition_type == "weighing_weighing_boat":
                return balance_g > WEIGHING_BOAT_THRESHOLD

            if self.setting.condition_type == "check_tare":
                return balance_g < TARE_THRESHOLD

            if self.setting.condition_type == "weighting":
                return balance_g >= TARE_THRESHOLD

            if self.setting.condition_type == "numerical_check":
                return (
                    (self.setting.target_weighting - self.setting.weighting_error)
                    <= balance_g
                    <= (self.setting.target_weighting + self.setting.weighting_error)
                )

        if self.setting.condition_type == "pick_spoon":
            if sampling_spoon_vertices is None or beaker_vertices is None:
                return False

            return not is_inside(
                src_pts=sampling_spoon_vertices,
                target_pts=beaker_vertices,
                thresh=SPOON_IN_BEAKER_THRESHOLD,
                scene=scene,
            )

        if self.setting.condition_type == "place_spoon":
            if sampling_spoon_vertices is None or beaker_vertices is None:
                return False

            return is_inside(
                src_pts=sampling_spoon_vertices,
                target_pts=beaker_vertices,
                thresh=SPOON_IN_BEAKER_THRESHOLD,
                scene=scene,
            )

        return False

    def parse_balance_g(self, lcd_readout):
        if not isinstance(lcd_readout, str):
            return None

        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*g\s*", lcd_readout)
        if match is None:
            return None

        value = float(match.group(1))
        return value if math.isfinite(value) else None
