from typing import Any
from pydantic import BaseModel, Field

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory
from genmanip.extensions.metrics.default.sr_based_genmanip_relationship import (
    SRBasedGenmanipRelationship,
    is_inside,
)


TUBE_IN_MIXED_RACK_THRESHOLD = 0.40


class RemoveCentrifugeTubesConfig(BaseModel):
    primary_tube_uid: str = Field(
        ..., description="Prim name or path of the primary tube object"
    )
    balance_tube_uid: str = Field(
        ..., description="Prim name or path of the balance tube object"
    )
    mixed_rack_uid: str = Field(
        ..., description="Prim name or path of the mixed rack object"
    )
    centrifuge_rotor_link_uid: str = Field(
        ..., description="Prim name or path of the centrifuge/rotor_link object"
    )
    condition_type: str = Field(..., description="condition type")


@MetricFactory.register("manip/scientific_experiments/task_06_remove_centrifuge_tubes")
class RemoveCentrifugeTubes(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = RemoveCentrifugeTubesConfig(**sub_goal_setting)

    def check_status(self, scene):
        target_uids = [
            self.setting.primary_tube_uid,
            self.setting.balance_tube_uid,
            self.setting.mixed_rack_uid,
            self.setting.centrifuge_rotor_link_uid,
        ]
        pclist = SRBasedGenmanipRelationship.get_target_pc_list(scene, target_uids)
        primary_tube_vertices = pclist.get(self.setting.primary_tube_uid)
        balance_tube_vertices = pclist.get(self.setting.balance_tube_uid)
        mixed_rack_vertices = pclist.get(self.setting.mixed_rack_uid)
        centrifuge_vertices = pclist.get(self.setting.centrifuge_rotor_link_uid)

        if self.setting.condition_type == "primary_tube_out_centrifuge":
            if primary_tube_vertices is None or centrifuge_vertices is None:
                return False
            if len(primary_tube_vertices) == 0 or len(centrifuge_vertices) == 0:
                return False

            return not is_inside(
                src_pts=primary_tube_vertices,
                target_pts=centrifuge_vertices,
                scene=scene,
            )

        if self.setting.condition_type == "balance_tube_out_centrifuge":
            if balance_tube_vertices is None or centrifuge_vertices is None:
                return False
            if len(balance_tube_vertices) == 0 or len(centrifuge_vertices) == 0:
                return False

            return not is_inside(
                src_pts=balance_tube_vertices,
                target_pts=centrifuge_vertices,
                scene=scene,
            )

        if self.setting.condition_type == "primary_tube_insert_mixed_rack":
            if primary_tube_vertices is None or mixed_rack_vertices is None:
                return False
            if len(primary_tube_vertices) == 0 or len(mixed_rack_vertices) == 0:
                return False

            return is_inside(
                src_pts=primary_tube_vertices,
                target_pts=mixed_rack_vertices,
                thresh=TUBE_IN_MIXED_RACK_THRESHOLD,
                scene=scene,
            )

        if self.setting.condition_type == "balance_tube_insert_mixed_rack":
            if balance_tube_vertices is None or mixed_rack_vertices is None:
                return False
            if len(balance_tube_vertices) == 0 or len(mixed_rack_vertices) == 0:
                return False

            return is_inside(
                src_pts=balance_tube_vertices,
                target_pts=mixed_rack_vertices,
                thresh=TUBE_IN_MIXED_RACK_THRESHOLD,
                scene=scene,
            )

        return False
