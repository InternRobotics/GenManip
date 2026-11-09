from typing import Any
from pydantic import BaseModel, Field
import numpy as np

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory
from genmanip.extensions.metrics.default.sr_based_genmanip_relationship import (
    SRBasedGenmanipRelationship,
    is_inside,
)
from genmanip.extensions.metrics.default.match_local_world_orientation import (
    MatchLocalWorldOrientation,
)


TUBE_IN_MIXED_RACK_THRESHOLD = 0.40


class SilverMirrorReactionConfig(BaseModel):
    require_slot_geometry: bool = False
    tube_uid: str = Field(..., description="Prim name or path of the tube object")
    mixed_rack_uid: str = Field(
        ..., description="Prim name or path of the mixed rack object"
    )
    tube_prim_path: str = Field(..., description="Prim path of the tube object")
    progress_thre: float = Field(
        ..., description="Progress threshold of the fehling test"
    )
    condition_type: str = Field(..., description="condition type")


@MetricFactory.register("manip/scientific_experiments/task_z4_silver_mirror_reaction")
class SilverMirrorReaction(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = SilverMirrorReactionConfig(**sub_goal_setting)

    def check_status(self, scene):
        if self.setting.require_slot_geometry:
            from genmanip.extensions.metrics.scientific_experiments.repair_checks import (
                runtime_ok,
            )

            if not runtime_ok(scene):
                return False
            from genmanip.extensions.metrics.scientific_experiments.repair_checks import (
                tube_inserted_in_rack,
            )

            if self.setting.condition_type in (
                "tube_insert_rack",
                "tube_not_insert_rack",
            ):
                inserted = tube_inserted_in_rack(
                    scene, self.setting.tube_uid, self.setting.mixed_rack_uid
                )
                return (
                    inserted
                    if self.setting.condition_type == "tube_insert_rack"
                    else not inserted
                )
            if self.setting.condition_type == "check_progress":
                tube = scene.world.stage.GetPrimAtPath(self.setting.tube_prim_path)
                from genmanip.extensions.metrics.scientific_experiments.bath_representation import (
                    bath_progress_valid,
                )

                if not bath_progress_valid(tube, "tollens"):
                    return False

        target_uids = [self.setting.tube_uid, self.setting.mixed_rack_uid]
        pclist = SRBasedGenmanipRelationship.get_target_pc_list(scene, target_uids)
        tube_vertices = pclist.get(self.setting.tube_uid)
        mixed_rack_vertices = pclist.get(self.setting.mixed_rack_uid)

        if self.setting.condition_type == "tube_not_insert_rack":
            if tube_vertices is None or mixed_rack_vertices is None:
                return False
            if len(tube_vertices) == 0 or len(mixed_rack_vertices) == 0:
                return False

            return not is_inside(
                src_pts=tube_vertices,
                target_pts=mixed_rack_vertices,
                thresh=TUBE_IN_MIXED_RACK_THRESHOLD,
                scene=scene,
            )

        if self.setting.condition_type == "tube_insert_rack":
            if tube_vertices is None or mixed_rack_vertices is None:
                return False
            if len(tube_vertices) == 0 or len(mixed_rack_vertices) == 0:
                return False

            if not is_inside(
                src_pts=tube_vertices,
                target_pts=mixed_rack_vertices,
                thresh=TUBE_IN_MIXED_RACK_THRESHOLD,
                scene=scene,
            ):
                return False

            tube_orientation = MatchLocalWorldOrientation(
                sub_goal_setting={
                    "obj_uid": self.setting.tube_uid,
                    "constraint_spec": {
                        "type": "AND",
                        "constraints": [
                            {
                                "local_axis": "Z",
                                "world_axis": "Z",
                                "angle_range": [0, 20],
                            }
                        ],
                    },
                }
            )

            return tube_orientation.check_status(scene)

        if self.setting.condition_type == "check_progress":
            tube_prim = scene.world.stage.GetPrimAtPath(self.setting.tube_prim_path)
            valve_state = tube_prim.GetAttribute("tollens:deposition_progress").Get()
            return valve_state > self.setting.progress_thre

        return False
