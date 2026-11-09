import math
from typing import Any

import numpy as np
from pxr import Usd, UsdGeom
from pydantic import BaseModel, Field

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
TUBE_INSERT_TILT_RANGE = (15, 35)

# Symmetry is the angle between the two tubes, measured in the rotor frame.
# Do not snap each tube to a sleeve center first. A pair sitting on the
# boundary between slots is 180 deg apart, but independent rounding assigns
# centers 165 deg apart. Episode 010 of the 500-episode render is that case:
# azimuths 7.76 deg and -172.74 deg (179.5 deg apart, 0.5 deg from opposite)
# round to slots 15 deg and -180 deg.
NEST_PRIM_PREFIX = "CavitySleeve_"
NEST_OPPOSITE_ANGLE_DEG = 180.0
# Half the 15 deg slot pitch. An adjacent slot is a 15 deg error, so 7.5 deg
# accepts an opposite pair and rejects a one-slot miss.
NEST_ANGLE_TOL_DEG = 7.5
# Radial sanity band for the tube centroid in the rotor frame. Seated tubes sit
# at r ~= 0.128 (sleeve centre radius 0.1285); the band only rejects a tube that
# is nowhere near the rotor ring.
NEST_RADIUS_RANGE = (0.09, 0.18)


class InsertCentrifugeTubesConfig(BaseModel):
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
    nest_prim_prefix: str = Field(
        NEST_PRIM_PREFIX,
        description="Name prefix of the rotor slot prims (children of rotor_link)",
    )
    nest_angle_tolerance_deg: float = Field(
        NEST_ANGLE_TOL_DEG,
        description="Allowed deviation from 180 deg between the two occupied slots",
    )


@MetricFactory.register("manip/scientific_experiments/task_07_insert_centrifuge_tubes")
class InsertCentrifugeTubes(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = InsertCentrifugeTubesConfig(**sub_goal_setting)
        self._nest_azimuths = None
        self._nest_source_path = None

    def _check_tube_orientation(self, scene, tube_uid):
        tube_orientation = MatchLocalWorldOrientation(
            sub_goal_setting={
                "obj_uid": tube_uid,
                "constraint_spec": {
                    "type": "AND",
                    "constraints": [
                        {
                            "local_axis": "Z",
                            "world_axis": "Z",
                            "angle_range": TUBE_INSERT_TILT_RANGE,
                        }
                    ],
                },
            }
        )
        return tube_orientation.check_status(scene)

    def _tube_inserted(self, scene, tube_vertices, centrifuge_vertices, tube_uid):
        if tube_vertices is None or centrifuge_vertices is None:
            return False
        if len(tube_vertices) == 0 or len(centrifuge_vertices) == 0:
            return False
        return is_inside(
            src_pts=tube_vertices,
            target_pts=centrifuge_vertices,
            scene=scene,
        ) and self._check_tube_orientation(scene, tube_uid)

    # ---------- rotor slot (nest) identification ----------

    @staticmethod
    def _matrix_to_numpy(m) -> np.ndarray:
        return np.array(
            [[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64
        )

    @staticmethod
    def _wrap_deg(angle: float) -> float:
        return (angle + 180.0) % 360.0 - 180.0

    def _rotor_world_matrix(self, scene):
        rotor = scene.object_list.get(self.setting.centrifuge_rotor_link_uid)
        if rotor is None:
            return None, None
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        world = xform_cache.GetLocalToWorldTransform(rotor.prim)
        return self._matrix_to_numpy(world), rotor.prim

    def _get_nest_azimuths(self, scene):
        """Azimuth (deg) of every rotor slot, in the rotor frame (cached per episode).

        Not used to decide symmetry. Nearest-slot rounding rejects pairs that
        sit on a slot boundary even when the tubes are opposite.
        """
        rotor_m, rotor_prim = self._rotor_world_matrix(scene)
        if rotor_m is None:
            return None

        source = str(rotor_prim.GetPath())
        if self._nest_azimuths is not None and self._nest_source_path == source:
            return self._nest_azimuths

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        rotor_m_inv = np.linalg.inv(rotor_m)
        azimuths = []
        for child in rotor_prim.GetChildren():
            if not child.GetName().startswith(self.setting.nest_prim_prefix):
                continue
            bound = bbox_cache.ComputeWorldBound(child).ComputeAlignedRange()
            if bound.IsEmpty():
                continue
            center_world = (bound.GetMin() + bound.GetMax()) / 2.0
            center_local = (
                np.array([center_world[0], center_world[1], center_world[2], 1.0])
                @ rotor_m_inv
            )
            azimuths.append(math.degrees(math.atan2(center_local[1], center_local[0])))

        self._nest_azimuths = (
            np.asarray(azimuths, dtype=np.float64) if azimuths else None
        )
        self._nest_source_path = source
        return self._nest_azimuths

    def _tube_slot_azimuth(self, tube_vertices, rotor_m):
        """Azimuth (deg) of the tube in the rotor frame, or None if off the ring."""
        # Row-vector frame: p_local = [p_world, 1] @ M^-1
        hom = np.concatenate(
            [
                np.asarray(tube_vertices, dtype=np.float64),
                np.ones((len(tube_vertices), 1)),
            ],
            axis=1,
        )
        tube_local = hom @ np.linalg.inv(rotor_m)

        centroid = tube_local[:, :3].mean(axis=0)
        radius = float(np.hypot(centroid[0], centroid[1]))
        if not (NEST_RADIUS_RANGE[0] <= radius <= NEST_RADIUS_RANGE[1]):
            return None
        return math.degrees(math.atan2(centroid[1], centroid[0]))

    def _check_nests_symmetric(self, scene, primary_vertices, balance_vertices) -> bool:
        rotor_m, _ = self._rotor_world_matrix(scene)
        if rotor_m is None:
            return False

        azimuths = []
        for tube_vertices in (primary_vertices, balance_vertices):
            azimuth = self._tube_slot_azimuth(tube_vertices, rotor_m)
            if azimuth is None:
                return False
            azimuths.append(azimuth)

        # Angle between the tubes, not between the nearest sleeve centers.
        separation = abs(self._wrap_deg(azimuths[0] - azimuths[1]))
        return (
            abs(separation - NEST_OPPOSITE_ANGLE_DEG)
            <= self.setting.nest_angle_tolerance_deg
        )

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

        if self.setting.condition_type == "primary_tube_insert_centrifuge":
            return self._tube_inserted(
                scene,
                primary_tube_vertices,
                centrifuge_vertices,
                self.setting.primary_tube_uid,
            )

        if self.setting.condition_type == "balance_tube_insert_centrifuge":
            return self._tube_inserted(
                scene,
                balance_tube_vertices,
                centrifuge_vertices,
                self.setting.balance_tube_uid,
            )

        if self.setting.condition_type == "tubes_insert_symmetric":
            # Only meaningful once both tubes are actually seated in the rotor.
            if not self._tube_inserted(
                scene,
                primary_tube_vertices,
                centrifuge_vertices,
                self.setting.primary_tube_uid,
            ):
                return False
            if not self._tube_inserted(
                scene,
                balance_tube_vertices,
                centrifuge_vertices,
                self.setting.balance_tube_uid,
            ):
                return False
            return self._check_nests_symmetric(
                scene, primary_tube_vertices, balance_tube_vertices
            )

        if self.setting.condition_type == "primary_tube_out_mixed_rack":
            if primary_tube_vertices is None or mixed_rack_vertices is None:
                return False
            if len(primary_tube_vertices) == 0 or len(mixed_rack_vertices) == 0:
                return False

            return not is_inside(
                src_pts=primary_tube_vertices,
                target_pts=mixed_rack_vertices,
                thresh=TUBE_IN_MIXED_RACK_THRESHOLD,
                scene=scene,
            )

        if self.setting.condition_type == "balance_tube_out_mixed_rack":
            if balance_tube_vertices is None or mixed_rack_vertices is None:
                return False
            if len(balance_tube_vertices) == 0 or len(mixed_rack_vertices) == 0:
                return False

            return not is_inside(
                src_pts=balance_tube_vertices,
                target_pts=mixed_rack_vertices,
                thresh=TUBE_IN_MIXED_RACK_THRESHOLD,
                scene=scene,
            )

        return False
