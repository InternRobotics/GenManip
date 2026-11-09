from typing import Any, Literal
from pydantic import BaseModel, Field
import numpy as np

from .task_11_surface_support import MeshSupportCache, surface_support_diagnostics

from genmanip.core.metrics.base import BaseMetric
from genmanip.core.metrics.utils import MetricFactory
from genmanip.extensions.metrics.default.sr_based_genmanip_relationship import (
    SRBasedGenmanipRelationship,
    is_inside,
)


class RemoveContainerFromOvenConfig(BaseModel):
    oven_body_uid: str = Field(..., description="UID of the oven's body")
    oven_door_uid: str = Field(..., description="UID of the oven's door")
    beaker_uid: str = Field(..., description="UID of the beaker")
    conical_flask_uid: str = Field(..., description="UID of the conical flask")
    condition_type: str = Field(..., description="condition type")


class TraySurfaceSupportConfig(BaseModel):
    condition_type: Literal["vessel_on_tray_surface"]
    obj1_uid: str
    obj2_uid: str
    support_grid_spacing_m: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    support_gap_tolerance_m: float = Field(default=0.003, ge=0, allow_inf_nan=False)
    support_penetration_tolerance_m: float = Field(
        default=0.003, ge=0, allow_inf_nan=False
    )
    support_min_projection_coverage: float = Field(
        default=0.95, gt=0, le=1, allow_inf_nan=False
    )
    support_min_patch_area_m2: float = Field(default=3e-6, gt=0, allow_inf_nan=False)
    support_max_grid_rays: int = Field(default=250000, gt=0)


@MetricFactory.register(
    "manip/scientific_experiments/task_11_remove_container_from_oven"
)
class RemoveContainerFromOven(BaseMetric):
    def __init__(
        self, skip_steps=1, succ_cnts=0, sub_goal_setting: dict[str, Any] = {}, **kwargs
    ):
        super().__init__(skip_steps, succ_cnts, sub_goal_setting, **kwargs)
        self.surface_setting = None
        self.last_support_diagnostics = None
        self._surface_cache = MeshSupportCache()
        if sub_goal_setting.get("condition_type") == "vessel_on_tray_surface":
            self.surface_setting = TraySurfaceSupportConfig(**sub_goal_setting)
        else:
            self.setting = RemoveContainerFromOvenConfig(**sub_goal_setting)

    def _check_tray_surface(self, scene):
        setting = self.surface_setting
        try:
            # A fresh MetricsManager builds fresh metric instances on episode reset.
            # Stage/world identity also clears local BVHs if an instance is reused.
            epoch = (
                id(scene),
                id(scene.world),
                scene.uuid,
                scene.world.stage.GetRootLayer().identifier,
            )
            self._surface_cache.bind_epoch(epoch)
            surfaces = []
            for uid in (setting.obj1_uid, setting.obj2_uid):
                surfaces.append(
                    self._surface_cache.get(
                        uid, scene.object_list[uid], scene.cache_library.mesh_dict[uid]
                    )
                )
            self.last_support_diagnostics = surface_support_diagnostics(
                surfaces[0],
                surfaces[1],
                spacing=setting.support_grid_spacing_m,
                gap_tolerance=setting.support_gap_tolerance_m,
                penetration_tolerance=setting.support_penetration_tolerance_m,
                min_coverage=setting.support_min_projection_coverage,
                min_patch_area=setting.support_min_patch_area_m2,
                max_grid_rays=setting.support_max_grid_rays,
            )
        except (KeyError, ValueError, RuntimeError, AttributeError) as exc:
            # Missing/stale geometry must not grant success. Diagnostics belong
            # to this metric instance, never to physical scene state.
            self.last_support_diagnostics = {
                "supported": False,
                "reason": "geometry_adapter_error",
                "error": str(exc),
            }
        return bool(self.last_support_diagnostics["supported"])

    def check_status(self, scene):
        if self.surface_setting is not None:
            return self._check_tray_surface(scene)
        target_uids = [
            self.setting.oven_body_uid,
            self.setting.oven_door_uid,
            self.setting.beaker_uid,
            self.setting.conical_flask_uid,
        ]
        pclist = SRBasedGenmanipRelationship.get_target_pc_list(scene, target_uids)

        oven_vertices = np.concatenate(
            [
                pclist[self.setting.oven_body_uid],
                pclist[self.setting.oven_door_uid],
            ],
            axis=0,
        )
        beaker_vertices = pclist.get(self.setting.beaker_uid)
        conical_flask_vertices = pclist.get(self.setting.conical_flask_uid)

        if self.setting.condition_type == "beaker_not_in_oven":
            return not is_inside(
                src_pts=beaker_vertices,
                target_pts=oven_vertices,
                thresh=0.9,
                scene=scene,
            )

        if self.setting.condition_type == "conical_flask_not_in_oven":
            return not is_inside(
                src_pts=conical_flask_vertices,
                target_pts=oven_vertices,
                thresh=0.9,
                scene=scene,
            )

        return False
