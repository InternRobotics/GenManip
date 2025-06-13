import os
import pickle
from typing import Dict, List, Optional
import random
import json
import uuid


def generate_hash():
    return uuid.uuid4().hex


def check_hash_valid(new_hash: str, previous_hash_list: list) -> bool:
    return not (new_hash in set(previous_hash_list))


class ObjectPool:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.object_info = {}
        self.uids = []
        self.set_infos()

    def set_infos(self, path: Optional[str] = None) -> None:
        if path is not None:
            self.path = path
        if self.path is None:
            return
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        self.object_info = data
        self.uids = list(self.object_info.keys())

    def get_object_info(self, uid: str) -> Dict[str, any]:
        return self.object_info.get(uid, None)

    def get_object_infos(self, uids: List[str]) -> Dict[str, any]:
        return {uid: self.get_object_info(uid) for uid in uids}

    def get_uids(self) -> List[str]:
        return self.uids

    def is_valid(self, uid: str) -> bool:
        return uid in self.object_info

    def get_valid_uids(self) -> List[str]:
        return [uid for uid in self.uids if self.is_valid(uid)]

    def sample_object_info_by_keywords_(self, keywords: List[str]) -> Dict[str, any]:
        object_list = {}
        for uid, info in self.object_info.items():
            if not any(
                [
                    keyword.lower() not in info["caption"].lower()
                    or keyword.lower() not in info["description"].lower()
                    for keyword in keywords
                ]
            ) and self.is_valid(uid):
                object_list[uid] = self.get_object_info(uid)
        return object_list

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.object_info, f)

    def update_pickle(self, update_object_info, is_cover=False):
        attributes_list = [
            "general_category",
            "category",
            "mass",
            "scale",
            "is_container",
            "caption",
            "category_path",
            "description",
            "short_description",
            "raw_data_path",
            "can_grasp",
            "materials",
            "color",
            "shape",
            "width",
            "length",
            "height",
            "volume",
            "is_articulated",
            "is_valid",
        ]
        for uid, info in update_object_info.items():
            invalid_flag = False
            for attr in attributes_list:
                if attr not in info:
                    print(f"{uid} info is not valid")
                    invalid_flag = True
                    break
            if invalid_flag:
                continue
            if uid in self.object_info:
                print(f"{uid} is already in object_info")
                continue
            self.object_info[uid] = info
        if is_cover:
            self.save()

    def sample_object_info_by_keywords(self, keywords: List[str]) -> Dict[str, any]:
        object_list = {}
        for uid, info in self.object_info.items():
            if (
                self.is_valid(uid)
                and any(
                    [
                        keyword.lower() in info["caption"].lower()
                        or keyword.lower() in info["description"].lower()
                        or keyword.lower() in info["general_category"].lower()
                        for keyword in keywords
                    ]
                )
                and info["general_category"] != "toy&model"
                and info["general_category"] != "animal model"
            ):
                object_list[uid] = self.get_object_info(uid)
        return object_list


if __name__ == "__main__":
    object_pool = ObjectPool("assets/objects/object_v7.pickle")
    previous_uids = object_pool.uids
    for uid in previous_uids:
        print(object_pool.get_object_info(uid))
