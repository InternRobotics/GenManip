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


BEAKER_LIQUID_IN_BEAKER_THRESHOLD = 0.95
BEAKER_STIRRER_CENTER_XY_DISTANCE_THRESHOLD = 0.018
STIR_BAR_ABOVE_STEEL_PLATE_HEIGHT_RATIO = 0.25


class MagneticBeakerStirrerConfig(BaseModel):
    stir_bar_uid: str = Field(
        ..., description="Prim name or path of the stir bar object"
    )
    beaker_uid: str = Field(..., description="Prim name or path of the beaker object")
    magnetic_stirrer_uid: str = Field(
        ..., description="Prim name or path of the graduated magnetic stirrer object"
    )
    steel_plate_uid: str = Field(
        ..., description="Prim name or path of the graduated plate object"
    )
    b_liquid_mesh_prim_path: str = Field(
        ..., description="Prim path of the liquid mesh in beaker object"
    )
    condition_type: str = Field(..., description="condition type")


@MetricFactory.register("manip/scientific_experiments/task_03_magnetic_beaker_stirrer")
class MagneticBeakerStirrer(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = MagneticBeakerStirrerConfig(**sub_goal_setting)

    def check_status(self, scene):
        pclist = SRBasedGenmanipRelationship.get_target_pc_list(
            scene,
            [
                self.setting.stir_bar_uid,
                self.setting.beaker_uid,
                self.setting.magnetic_stirrer_uid,
                self.setting.steel_plate_uid,
            ],
        )
        stir_bar_vertices = pclist.get(self.setting.stir_bar_uid)
        beaker_vertices = pclist.get(self.setting.beaker_uid)
        magnetic_stirrer_vertices = pclist.get(self.setting.magnetic_stirrer_uid)
        steel_plate_vertices = pclist.get(self.setting.steel_plate_uid)

        if self.setting.condition_type == "timing_1":
            if stir_bar_vertices is None or steel_plate_vertices is None:
                return False
            if len(stir_bar_vertices) == 0 or len(steel_plate_vertices) == 0:
                return False

            stir_bar_xy_center = stir_bar_vertices[:, :2].mean(axis=0)
            steel_plate_min_z = steel_plate_vertices[:, 2].min()
            steel_plate_height = steel_plate_vertices[:, 2].max() - steel_plate_min_z
            stir_bar_height_threshold = (
                steel_plate_min_z
                + steel_plate_height * STIR_BAR_ABOVE_STEEL_PLATE_HEIGHT_RATIO
            )
            is_above_steel_plate = (
                stir_bar_vertices[:, 2].min() > stir_bar_height_threshold
            )
            is_in_steel_plate_x_projection = (
                steel_plate_vertices[:, 0].min()
                <= stir_bar_xy_center[0]
                <= steel_plate_vertices[:, 0].max()
            )
            is_in_steel_plate_y_projection = (
                steel_plate_vertices[:, 1].min()
                <= stir_bar_xy_center[1]
                <= steel_plate_vertices[:, 1].max()
            )
            return (
                is_above_steel_plate
                and is_in_steel_plate_x_projection
                and is_in_steel_plate_y_projection
            )

        if self.setting.condition_type == "timing_2/beaker_in_the_center_of_stirrer":
            if beaker_vertices is None or magnetic_stirrer_vertices is None:
                return False
            if len(beaker_vertices) == 0 or len(magnetic_stirrer_vertices) == 0:
                return False

            beaker_xy_center = beaker_vertices[:, :2].mean(axis=0)
            magnetic_stirrer_xy_center = magnetic_stirrer_vertices[:, :2].mean(axis=0)
            magnetic_stirrer_xy_center = magnetic_stirrer_xy_center + np.array(
                [0.0, 0.012]
            )
            xy_distance = np.linalg.norm(beaker_xy_center - magnetic_stirrer_xy_center)
            return bool(xy_distance < BEAKER_STIRRER_CENTER_XY_DISTANCE_THRESHOLD)

        if self.setting.condition_type == "timing_2/beaker_liquid_in_beaker":
            if beaker_vertices is None or len(beaker_vertices) == 0:
                return False

            b_liquid_prim = scene.world.stage.GetPrimAtPath(
                self.setting.b_liquid_mesh_prim_path
            )
            b_liquid_points = b_liquid_prim.GetAttribute("points").Get()
            b_liquid_world_transform = UsdGeom.XformCache().GetLocalToWorldTransform(
                b_liquid_prim
            )
            b_liquid_particles = np.asarray(
                [
                    b_liquid_world_transform.Transform(point)
                    for point in b_liquid_points
                ],
                dtype=np.float64,
            )
            if len(b_liquid_particles) == 0:
                return False

            return is_inside(
                src_pts=b_liquid_particles,
                target_pts=beaker_vertices,
                thresh=BEAKER_LIQUID_IN_BEAKER_THRESHOLD,
                scene=scene,
            )

        return False
