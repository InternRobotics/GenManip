"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory


class SRBasedGenmanipRangeConfig(BaseModel):
    obj1_uid: str = Field(..., description="UID of the object to measure the range")
    obj1_local_offset: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Measured point offset in meters, rotated by obj1 pose without USD scale",
    )

    @field_validator("obj1_local_offset")
    @classmethod
    def validate_local_offset(cls, value):
        if not np.isfinite(value).all():
            raise ValueError("obj1_local_offset must contain finite values")
        return value

    x_type: str = Field(..., description="Type of the x axis")
    y_type: str = Field(..., description="Type of the y axis")
    z_type: str = Field(..., description="Type of the z axis")
    x_range: list[float] | None = Field(default=None, description="Range of the x axis")
    y_range: list[float] | None = Field(default=None, description="Range of the y axis")
    z_range: list[float] | None = Field(default=None, description="Range of the z axis")
    x_rel_object_uid: str | None = Field(
        default=None, description="UID of the relative object for the x axis"
    )
    y_rel_object_uid: str | None = Field(
        default=None, description="UID of the relative object for the y axis"
    )
    z_rel_object_uid: str | None = Field(
        default=None, description="UID of the relative object for the z axis"
    )


@MetricFactory.register("manip/default/sr_based_genmanip_range")
class SRBasedGenmanipRange(BaseMetric):
    def __init__(
        self,
        skip_steps: int = 1,
        succ_cnts: int = 0,
        sub_goal_setting: dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.goal_setting = SRBasedGenmanipRangeConfig(**sub_goal_setting)

    def _get_axis_range(self, scene) -> list[list[float]]:
        x_range = [-np.inf, np.inf]
        y_range = [-np.inf, np.inf]
        z_range = [-np.inf, np.inf]
        if self.goal_setting.x_type == "none":
            x_range = [-np.inf, np.inf]
        elif self.goal_setting.x_type == "relative":
            if self.goal_setting.x_rel_object_uid in scene.object_list:
                rel_object = scene.object_list[self.goal_setting.x_rel_object_uid]
            elif self.goal_setting.x_rel_object_uid in scene.articulation_list:
                rel_object = scene.articulation_list[self.goal_setting.x_rel_object_uid]
            else:
                raise ValueError(
                    f"Relative object {self.goal_setting.x_rel_object_uid} not found"
                )
            x_range = [
                (
                    rel_object.get_world_pose()[0][0] + self.goal_setting.x_range[0]
                    if self.goal_setting.x_range is not None
                    else -np.inf
                ),
                (
                    rel_object.get_world_pose()[0][0] + self.goal_setting.x_range[1]
                    if self.goal_setting.x_range is not None
                    else np.inf
                ),
            ]
        elif self.goal_setting.x_type == "absolute":
            x_range = [
                (
                    self.goal_setting.x_range[0]
                    if self.goal_setting.x_range is not None
                    else -np.inf
                ),
                (
                    self.goal_setting.x_range[1]
                    if self.goal_setting.x_range is not None
                    else np.inf
                ),
            ]
        if self.goal_setting.y_type == "none":
            y_range = [-np.inf, np.inf]
        elif self.goal_setting.y_type == "relative":
            if self.goal_setting.y_rel_object_uid in scene.object_list:
                rel_object = scene.object_list[self.goal_setting.y_rel_object_uid]
            elif self.goal_setting.y_rel_object_uid in scene.articulation_list:
                rel_object = scene.articulation_list[self.goal_setting.y_rel_object_uid]
            else:
                raise ValueError(
                    f"Relative object {self.goal_setting.y_rel_object_uid} not found"
                )
            y_range = [
                (
                    rel_object.get_world_pose()[0][1] + self.goal_setting.y_range[0]
                    if self.goal_setting.y_range is not None
                    else -np.inf
                ),
                (
                    rel_object.get_world_pose()[0][1] + self.goal_setting.y_range[1]
                    if self.goal_setting.y_range is not None
                    else np.inf
                ),
            ]
        elif self.goal_setting.y_type == "absolute":
            y_range = [
                (
                    self.goal_setting.y_range[0]
                    if self.goal_setting.y_range is not None
                    else -np.inf
                ),
                (
                    self.goal_setting.y_range[1]
                    if self.goal_setting.y_range is not None
                    else np.inf
                ),
            ]
        if self.goal_setting.z_type == "none":
            z_range = [-np.inf, np.inf]
        elif self.goal_setting.z_type == "relative":
            if self.goal_setting.z_rel_object_uid in scene.object_list:
                rel_object = scene.object_list[self.goal_setting.z_rel_object_uid]
            elif self.goal_setting.z_rel_object_uid in scene.articulation_list:
                rel_object = scene.articulation_list[self.goal_setting.z_rel_object_uid]
            else:
                raise ValueError(
                    f"Relative object {self.goal_setting.z_rel_object_uid} not found"
                )
            z_range = [
                (
                    rel_object.get_world_pose()[0][2] + self.goal_setting.z_range[0]
                    if self.goal_setting.z_range is not None
                    else -np.inf
                ),
                (
                    rel_object.get_world_pose()[0][2] + self.goal_setting.z_range[1]
                    if self.goal_setting.z_range is not None
                    else np.inf
                ),
            ]
        elif self.goal_setting.z_type == "absolute":
            z_range = [
                (
                    self.goal_setting.z_range[0]
                    if self.goal_setting.z_range is not None
                    else -np.inf
                ),
                (
                    self.goal_setting.z_range[1]
                    if self.goal_setting.z_range is not None
                    else np.inf
                ),
            ]
        return [x_range, y_range, z_range]

    def check_status(self, scene) -> bool:
        """
        Check if the alignment condition is met.

        Args:
            scene: The simulation scene object.

        Returns:
            bool: True if the condition is met, False otherwise.
        """
        object_1 = scene.object_list[self.goal_setting.obj1_uid]
        object_1_pose = object_1.get_world_pose()
        x_range, y_range, z_range = self._get_axis_range(scene)
        position = np.asarray(object_1_pose[0], dtype=float)
        if any(self.goal_setting.obj1_local_offset):
            rotation = R.from_quat(np.asarray(object_1_pose[1])[[1, 2, 3, 0]])
            position = position + rotation.apply(self.goal_setting.obj1_local_offset)
        if position[0] < x_range[0] or position[0] > x_range[1]:
            return False
        if position[1] < y_range[0] or position[1] > y_range[1]:
            return False
        if position[2] < z_range[0] or position[2] > z_range[1]:
            return False
        return True

    def get_info(self):
        return f"put {self.goal_setting.obj1_uid} to the range of {self.goal_setting.x_range}{f' relative to {self.goal_setting.x_rel_object_uid}' if self.goal_setting.x_rel_object_uid is not None else ''} {self.goal_setting.y_range}{f' relative to {self.goal_setting.y_rel_object_uid}' if self.goal_setting.y_rel_object_uid is not None else ''} {self.goal_setting.z_range}{f' relative to {self.goal_setting.z_rel_object_uid}' if self.goal_setting.z_rel_object_uid is not None else ''}"
