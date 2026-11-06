"""
We recommend using a independent conda environment to run this script.

conda create -n usd python=3.10
conda activate usd
pip install usd-core==24.11
pip install open3d numpy shapely
"""

import os
from pxr import Usd, Sdf  # type: ignore


def move_prim(stage: Usd.Stage, old_path: str, new_path: str):
    old_path = Sdf.Path(old_path)
    new_path = Sdf.Path(new_path)
    old_prim = stage.GetPrimAtPath(old_path)
    if not old_prim or not old_prim.IsValid():
        print(f"[Error] Prim at {old_path} does not exist.")
        return
    if stage.GetPrimAtPath(new_path).IsValid():
        print(f"[Error] Prim at {new_path} already exists.")
        return
    layer = stage.GetEditTarget().GetLayer()
    if not Sdf.CopySpec(layer, old_path, layer, new_path):
        print(f"[Error] Failed to copy spec from {old_path} to {new_path}")
        return
    stage.RemovePrim(old_path)


def add_prim_from_asset(stage: Usd.Stage, prim_path: str, asset_path: str):
    prim_path = Sdf.Path(prim_path)
    if stage.GetPrimAtPath(prim_path).IsValid():
        print(f"[Error] Prim at {prim_path} already exists.")
        return
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(asset_path)


def save_stage_as(stage: Usd.Stage, new_path: str):
    new_layer = Sdf.Layer.CreateNew(new_path)
    if not new_layer:
        return
    stage.Export(new_path)


def delete_prim(stage: Usd.Stage, prim_path: str):
    prim_path = Sdf.Path(prim_path)
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim_path)


def print_prim_tree(prim, prefix="", is_last=True):
    RESET = "\033[0m"
    COLORS = {
        "Xform": "\033[94m",
        "Mesh": "\033[95m",
        "Camera": "\033[93m",
        "Light": "\033[96m",
        "Default": "\033[92m",
        "Type": "\033[91m",
    }
    connector = "└── " if is_last else "├── "
    prim_name = prim.GetName()
    prim_type = prim.GetTypeName()
    color = COLORS.get(prim_type, COLORS["Default"])
    color_type = COLORS["Type"]
    colored_name = f"{color}{prim_name}{RESET}"
    colored_type = f"{color_type} ({prim_type}){RESET}" if prim_type else ""
    print(prefix + connector + colored_name + colored_type)
    children = prim.GetChildren()
    count = len(children)
    for i, child in enumerate(children):
        is_last_child = i == count - 1
        extension = "    " if is_last else "│   "
        print_prim_tree(child, prefix + extension, is_last_child)


if __name__ == "__main__":
    stage = Usd.Stage.Open(
        os.environ.get("GENMANIP_SCENE_USD", "saved/assets/scene_usds/test_mona_lisa/scene.usd")
    )
    print_prim_tree(stage.GetPseudoRoot())
