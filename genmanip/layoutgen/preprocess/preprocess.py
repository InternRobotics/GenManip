from genmanip.utils.file_utils import load_json
from collections import defaultdict
import random

TABLE_UID = "00000000000000000000000000000000"
TABLE_OBJ = "table"


class ObjectSampler:
    def __init__(self, json_file):
        self.json_file = json_file
        self.data = load_json(json_file)

    def get_object_info(self, uid):
        return self.data.get(uid, None)


import os


def flip_position(position):
    flips = {
        "left": "right",
        "right": "left",
        "front": "back",
        "back": "front",
        "top": "bottom",
        "bottom": "top",
    }
    return flips.get(position, position)


def solve_task_graph_conflict(data):
    graph_list = []
    for x in data["graph"]:
        if not (
            (
                (
                    x["obj1_uid"] == data["goal"][0][0]["obj1_uid"]
                    and x["obj2_uid"] == data["goal"][0][0]["obj2_uid"]
                )
                or (
                    x["obj1_uid"] == data["goal"][0][0]["obj2_uid"]
                    and x["obj2_uid"] == data["goal"][0][0]["obj1_uid"]
                )
            )
            and x["position"] == data["goal"][0][0]["position"]
        ):
            graph_list.append(x)
        else:
            graph_list.append(
                {
                    "obj1": x["obj1"],
                    "obj1_uid": x["obj1_uid"],
                    "position": "top",
                    "obj2": TABLE_OBJ,
                    "obj2_uid": TABLE_UID,
                }
            )
    data["graph"] = graph_list


def collect_uids(graph):
    uids = set()
    for relation in graph:
        uids.add(relation["obj1_uid"])
        uids.add(relation["obj2_uid"])
    uids.add(TABLE_UID)  # 添加桌子的 UID
    return list(uids)


def add_table_relations(data, uids):
    obj1_uids = set(
        relation["obj1_uid"]
        for relation in data["graph"]
        if relation.get("position") == "top"
    )
    for uid in uids:
        if uid == TABLE_UID:
            continue
        if uid not in obj1_uids:
            obj_name = None
            for relation in data["graph"]:
                if relation["obj1_uid"] == uid:
                    obj_name = relation["obj1"]
                    break
                elif relation["obj2_uid"] == uid:
                    obj_name = relation["obj2"]
                    break
                new_relation = {
                    "obj1": obj_name,
                    "obj1_uid": uid,
                    "position": "top",
                    "obj2": TABLE_OBJ,
                    "obj2_uid": TABLE_UID,
                }
                data["graph"].append(new_relation)


def process_task(file_path, uid_num=3, uid=None):
    data = load_json(file_path)
    if uid and uid != -1:
        data = data[uid]
        scene_uid = uid
    elif uid and uid == -1:
        data = data
    else:
        scene_uid, data = random.choice(list(data.items()))

    solve_task_graph_conflict(data)

    not_in = True
    for graph in data["graph"]:
        if (
            data["goal"][0][0]["obj1_uid"] == graph["obj1_uid"]
            or data["goal"][0][0]["obj1_uid"] == graph["obj2_uid"]
        ):
            not_in = False
    if not_in:
        data["graph"].append(
            {
                "obj1": data["goal"][0][0]["obj1"],
                "obj1_uid": data["goal"][0][0]["obj1_uid"],
                "position": "top",
                "obj2": TABLE_OBJ,
                "obj2_uid": TABLE_UID,
            }
        )
    not_in = True
    for graph in data["graph"]:
        if (
            data["goal"][0][0]["obj2_uid"] == graph["obj1_uid"]
            or data["goal"][0][0]["obj2_uid"] == graph["obj2_uid"]
        ):
            not_in = False
    if not_in:
        data["graph"].append(
            {
                "obj1": data["goal"][0][0]["obj2"],
                "obj1_uid": data["goal"][0][0]["obj2_uid"],
                "position": "top",
                "obj2": TABLE_OBJ,
                "obj2_uid": TABLE_UID,
            }
        )
    uids = collect_uids(data["graph"])
    sampler = ObjectSampler(
        os.path.join(
            "/home/gaoning/isaac/AutoBenchmark/scripts/object_filter/results/results.json"
        )
    )
    for uid in uids:
        is_bottom = False
        for graph in data["graph"]:
            if graph["obj2_uid"] == uid and graph["position"] == "top":
                is_bottom = True
                break
        try:
            if (
                (not is_bottom)
                and uid != TABLE_UID
                and sampler.get_object_info(uid)["size"] > 60
            ):
                uids.remove(uid)
        except:
            print(uid)
    black_list = ["table"]
    filtered_uids = []
    for uid in uids:
        if uid == TABLE_UID:
            continue
        try:
            name = sampler.get_object_info(uid)["name"].lower()
        except:
            print(uid)
        if not any(word in name for word in black_list):
            filtered_uids.append(uid)
    uids[:] = filtered_uids
    while len(uids) > uid_num:
        removed = random.choice(uids)
        if (
            removed != data["goal"][0][0]["obj1_uid"]
            and removed != data["goal"][0][0]["obj2_uid"]
            and removed != TABLE_UID
        ):
            uids.remove(removed)
    for graph in data["graph"]:
        if not graph["obj1_uid"] in uids:
            if graph["obj2_uid"] in uids:
                data["graph"].append(
                    {
                        "obj1": graph["obj2"],
                        "obj1_uid": graph["obj2_uid"],
                        "position": "top",
                        "obj2": TABLE_OBJ,
                        "obj2_uid": TABLE_UID,
                    }
                )
            data["graph"].remove(graph)
            IS_OK = False
        elif not graph["obj2_uid"] in uids:
            if graph["obj1_uid"] in uids:
                data["graph"].append(
                    {
                        "obj1": graph["obj1"],
                        "obj1_uid": graph["obj1_uid"],
                        "position": "top",
                        "obj2": TABLE_OBJ,
                        "obj2_uid": TABLE_UID,
                    }
                )
            data["graph"].remove(graph)
    add_table_relations(data, uids)
    uids = collect_uids(data["graph"])
    sorted_uids = sort_uids(data)
    reconstruct_graph(data, sorted_uids)
    return data, sorted_uids, scene_uid


def sort_uids(data):
    placement_graph = defaultdict(list)
    for relation in data["graph"]:
        if relation["position"] == "top":
            placement_graph[relation["obj2_uid"]].append(relation["obj1_uid"])
    sorted_uids = []
    visited = set()

    def dfs(uid):
        if uid in visited:
            return
        visited.add(uid)
        for child_uid in placement_graph.get(uid, []):
            dfs(child_uid)
        sorted_uids.append(uid)

    dfs(TABLE_UID)
    sorted_uids = sorted_uids[::-1]
    return sorted_uids


def reconstruct_graph(data, sorted_uids):
    original_graph = data["graph"][:]
    new_graph = []
    for uid in sorted_uids:
        if uid == TABLE_UID:
            continue
        top_relations = [
            rel
            for rel in original_graph
            if rel["obj1_uid"] == uid and rel["position"] == "top"
        ]
        non_top_relations = [
            rel
            for rel in original_graph
            if (rel["obj1_uid"] == uid or rel["obj2_uid"] == uid)
            and rel["position"] != "top"
        ]
        notop = True
        for bottom_uid in reversed(sorted_uids):
            for rel in top_relations:
                if rel["obj2_uid"] == bottom_uid:
                    if notop:
                        notop = not notop
                        new_graph.append(rel)
                    original_graph.remove(rel)
        for rel in non_top_relations:
            original_graph.remove(rel)
            if rel["obj1_uid"] == uid:
                flipped_rel = {
                    "obj1": rel["obj2"],
                    "obj1_uid": rel["obj2_uid"],
                    "position": flip_position(rel["position"]),
                    "obj2": rel["obj1"],
                    "obj2_uid": rel["obj1_uid"],
                }
                new_graph.append(flipped_rel)
            elif rel["obj2_uid"] == uid:
                # 保持不变
                new_graph.append(rel)
    data["graph"] = new_graph
