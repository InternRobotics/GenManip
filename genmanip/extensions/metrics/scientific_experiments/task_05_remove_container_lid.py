from typing import Any
from pydantic import BaseModel, Field
from scipy.spatial import ConvexHull, QhullError

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory
from genmanip.extensions.metrics.default.sr_based_genmanip_relationship import (
    SRBasedGenmanipRelationship,
    calculate_distance_between_two_point_clouds,
    is_point_in_convex_hull_fast,
)


CONTACT_TOLERANCE = 0.001
PLUG_PLATE_CONTACT_TOLERANCE = 0.01


class RemoveContainerLidConfig(BaseModel):
    bottle_uid: str = Field(..., description="Prim name or path of the bottle object")
    bottle_plug_uid: str = Field(
        ..., description="Prim name or path of the bottle_plug object"
    )
    plate_uid: str = Field(..., description="Prim name or path of the plate object")
    condition_type: str = Field(..., description="condition type")


@MetricFactory.register("manip/scientific_experiments/task_05_remove_container_lid")
class RemoveContainerLid(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = RemoveContainerLidConfig(**sub_goal_setting)

    def check_status(self, scene):
        target_uids = [
            self.setting.bottle_uid,
            self.setting.bottle_plug_uid,
            self.setting.plate_uid,
        ]
        pclist = SRBasedGenmanipRelationship.get_target_pc_list(scene, target_uids)
        bottle_vertices = pclist.get(self.setting.bottle_uid)
        bottle_plug_vertices = pclist.get(self.setting.bottle_plug_uid)
        plate_vertices = pclist.get(self.setting.plate_uid)

        if bottle_plug_vertices is None or not len(bottle_plug_vertices):
            return False

        if self.setting.condition_type == "plug_remove_from_bottle":
            if bottle_vertices is None or not len(bottle_vertices):
                return False
            dist = calculate_distance_between_two_point_clouds(
                bottle_vertices, bottle_plug_vertices
            )

            return dist > CONTACT_TOLERANCE

        if self.setting.condition_type == "plug_above_the_plate":
            if plate_vertices is None or not len(plate_vertices):
                return False

            if not (bottle_plug_vertices[:, 2].min() > plate_vertices[:, 2].min()):
                return False

            # Approximate the plate footprint by its convex XY projection.
            try:
                plate_xy_hull = ConvexHull(plate_vertices[:, :2])
            except (QhullError, ValueError):
                return False

            # Require every sampled lid point to lie inside or on the boundary.
            if not bool(
                is_point_in_convex_hull_fast(
                    bottle_plug_vertices[:, :2], plate_xy_hull
                ).all()
            ):
                return False

            dist = calculate_distance_between_two_point_clouds(
                bottle_plug_vertices, plate_vertices
            )
            return dist <= PLUG_PLATE_CONTACT_TOLERANCE

        return False
