from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from pxr import UsdGeom  # type: ignore
from scipy.spatial import ConvexHull, QhullError

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory
from genmanip.extensions.metrics.default.sr_based_genmanip_relationship import (
    SRBasedGenmanipRelationship,
    is_inside,
    is_point_in_convex_hull_fast,
)

CF_LIQUID_IN_GC_TIMING_2_THRESHOLD = 0.01
CF_LIQUID_IN_GC_OR_CF_TIMING_3_THRESHOLD = 0.95


class PourFlaskToCylinderConfig(BaseModel):
    graduated_cylinder_uid: str = Field(
        ..., description="Prim name or path of the graduated cylinder object"
    )
    conical_flask_uid: str = Field(
        ..., description="Prim name or path of the conical flask object"
    )
    cf_liquid_mesh_prim_path: str = Field(
        ..., description="Prim path of the liquid mesh in conical flask object"
    )
    condition_type: str = Field(..., description="condition type")


@MetricFactory.register("manip/scientific_experiments/task_01_pour_flask_to_cylinder")
class PourFlaskToCylinder(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.setting = PourFlaskToCylinderConfig(**sub_goal_setting)

    def check_status(self, scene):
        pclist = SRBasedGenmanipRelationship.get_target_pc_list(
            scene, [self.setting.graduated_cylinder_uid, self.setting.conical_flask_uid]
        )
        graduated_cylinder_vertices = pclist.get(self.setting.graduated_cylinder_uid)
        conical_flask_vertices = pclist.get(self.setting.conical_flask_uid)

        cf_liquid_prim = scene.world.stage.GetPrimAtPath(
            self.setting.cf_liquid_mesh_prim_path
        )
        cf_liquid_points = cf_liquid_prim.GetAttribute("points").Get()
        cf_liquid_world_transform = UsdGeom.XformCache().GetLocalToWorldTransform(
            cf_liquid_prim
        )
        cf_liquid_particles = np.asarray(
            [cf_liquid_world_transform.Transform(point) for point in cf_liquid_points],
            dtype=np.float64,
        )

        if self.setting.condition_type == "timing_2":
            if graduated_cylinder_vertices is None:
                return False

            return is_inside(
                src_pts=cf_liquid_particles,
                target_pts=graduated_cylinder_vertices,
                thresh=CF_LIQUID_IN_GC_TIMING_2_THRESHOLD,
                scene=scene,
            )

        if self.setting.condition_type == "timing_3/liquid_only_in_gc_and_cf":
            if graduated_cylinder_vertices is None or conical_flask_vertices is None:
                return False

            if len(cf_liquid_particles) == 0:
                return False

            try:
                graduated_cylinder_hull = ConvexHull(graduated_cylinder_vertices)
                conical_flask_hull = ConvexHull(conical_flask_vertices)
            except (QhullError, ValueError):
                return False

            particles_in_graduated_cylinder = is_point_in_convex_hull_fast(
                cf_liquid_particles, graduated_cylinder_hull
            )
            particles_in_conical_flask = is_point_in_convex_hull_fast(
                cf_liquid_particles, conical_flask_hull
            )
            particles_in_either_container = np.logical_or(
                particles_in_graduated_cylinder, particles_in_conical_flask
            )
            required_particle_count = int(
                np.ceil(
                    CF_LIQUID_IN_GC_OR_CF_TIMING_3_THRESHOLD * len(cf_liquid_particles)
                )
            )
            return int(particles_in_either_container.sum()) >= required_particle_count

        return False
