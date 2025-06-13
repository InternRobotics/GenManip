import random


def apply_rule(rule_name, usd_list, object_pool):
    if rule_name == "can_grasp":
        can_grasp_list = [
            usd for usd in usd_list if check_can_grasp(usd.split(".")[0], object_pool)
        ]
        return can_grasp_list
    elif rule_name == "is_container":
        return [
            usd
            for usd in usd_list
            if check_is_container(usd.split(".")[0], object_pool)
        ]
    elif "retrieve_category" in rule_name:
        return [
            usd
            for usd in usd_list
            if retrieve_category(
                usd.split(".")[0],
                object_pool,
                rule_name.replace("retrieve_category_", ""),
            )
        ]
    elif "retrieve_not_category" in rule_name:
        return [
            usd
            for usd in usd_list
            if not retrieve_category(
                usd.split(".")[0],
                object_pool,
                rule_name.replace("retrieve_not_category_", ""),
            )
        ]
    elif "retrieve_shape" in rule_name:
        return [
            usd
            for usd in usd_list
            if retrieve_shape(
                usd.split(".")[0], object_pool, rule_name.replace("retrieve_shape_", "")
            )
        ]
    elif "retrieve_not_shape" in rule_name:
        return [
            usd
            for usd in usd_list
            if not retrieve_shape(
                usd.split(".")[0],
                object_pool,
                rule_name.replace("retrieve_not_shape_", ""),
            )
        ]
    elif "retrieve_materials" in rule_name:
        return [
            usd
            for usd in usd_list
            if retrieve_materials(
                usd.split(".")[0],
                object_pool,
                rule_name.replace("retrieve_materials_", ""),
            )
        ]
    elif "retrieve_not_materials" in rule_name:
        return [
            usd
            for usd in usd_list
            if not retrieve_materials(
                usd.split(".")[0],
                object_pool,
                rule_name.replace("retrieve_not_materials_", ""),
            )
        ]
    elif "retrieve_color" in rule_name:
        return [
            usd
            for usd in usd_list
            if retrieve_color(
                usd.split(".")[0], object_pool, rule_name.replace("retrieve_color_", "")
            )
        ]
    elif "retrieve_not_color" in rule_name:
        return [
            usd
            for usd in usd_list
            if not retrieve_color(
                usd.split(".")[0],
                object_pool,
                rule_name.replace("retrieve_not_color_", ""),
            )
        ]
    elif "retrieve_scale_less_than" in rule_name:
        return [
            usd
            for usd in usd_list
            if retrieve_scale_less_than(
                usd.split(".")[0],
                object_pool,
                rule_name.replace("retrieve_scale_less_than_", ""),
            )
        ]
    elif "retrieve_scale_greater_than" in rule_name:
        return [
            usd
            for usd in usd_list
            if retrieve_scale_greater_than(
                usd.split(".")[0],
                object_pool,
                rule_name.replace("retrieve_scale_greater_than_", ""),
            )
        ]
    else:
        return usd_list


def retrieve_scale_greater_than(uid, object_pool, rule_name):
    object_info = object_pool.get_object_info(uid)
    if object_info is None:
        return False
    if float(object_info["scale"][-1]) >= float(rule_name):
        return True
    else:
        return False


def retrieve_scale_less_than(uid, object_pool, rule_name):
    object_info = object_pool.get_object_info(uid)
    if object_info is None:
        return False
    if float(object_info["scale"][-1]) <= float(rule_name):
        return True
    else:
        return False


def retrieve_materials(uid, object_pool, rule_name):
    object_info = object_pool.get_object_info(uid)
    if object_info is None:
        return False
    if rule_name in object_info["materials"]:
        return True
    else:
        return False


def retrieve_color(uid, object_pool, rule_name):
    object_info = object_pool.get_object_info(uid)
    if object_info is None:
        return False
    if rule_name in object_info["color"]:
        return True
    else:
        return False


def retrieve_category(uid, object_pool, rule_name):
    object_info = object_pool.get_object_info(uid)
    if object_info is None:
        return False
    if rule_name in object_info["category_path"]:
        return True
    else:
        return False


def retrieve_shape(uid, object_pool, rule_name):
    object_info = object_pool.get_object_info(uid)
    if object_info is None:
        return False
    if rule_name in object_info["shape"]:
        return True
    else:
        return False


def check_mass(mass, threshold):
    if isinstance(mass, list):
        if len(mass) == 0:
            return False
        return float(mass[-1]) < threshold
    else:
        return float(mass) < threshold


def check_can_grasp(uid, object_pool):
    object_info = object_pool.get_object_info(uid)
    if object_info is None:
        return False
    return object_info["can_grasp"]


def check_is_container(uid, object_pool):
    object_info = object_pool.get_object_info(uid)
    if object_info is None:
        return False
    return object_info["is_container"]


def find_parent_category(usd_list, category, object_pool):
    parent_category_list = []
    for usd in usd_list:
        uid = usd.split(".")[0]
        object_info = object_pool.get_object_info(uid)
        if object_info is None:
            continue
        if category in object_info["category_path"]:
            index = object_info["category_path"].index(category)
            if index > 0:
                parent_category_list.append(object_info["category_path"][index - 1])
    return parent_category_list


def collect_shapes_by_category(usd_list, category, object_pool):
    shape_list = []
    for usd in usd_list:
        uid = usd.split(".")[0]
        object_info = object_pool.get_object_info(uid)
        if object_info is None:
            continue
        if category in object_info["category_path"]:
            shape_list.extend(object_info["shape"])
    return list(set(shape_list))


def collect_materials_by_category(usd_list, category, object_pool):
    material_list = []
    for usd in usd_list:
        uid = usd.split(".")[0]
        object_info = object_pool.get_object_info(uid)
        if object_info is None:
            continue
        if category in object_info["category_path"]:
            material_list.extend(object_info["materials"])
    return list(set(material_list))


def collect_colors_by_category(usd_list, category, object_pool):
    color_list = []
    for usd in usd_list:
        uid = usd.split(".")[0]
        object_info = object_pool.get_object_info(uid)
        if object_info is None:
            continue
        if category in object_info["category_path"]:
            color_list.extend(object_info["color"])
    return list(set(color_list))


def collect_all_categories(usd_list, object_pool):
    category_list = []
    for usd in usd_list:
        uid = usd.split(".")[0]
        object_info = object_pool.get_object_info(uid)
        if object_info is None:
            continue
        category_list.extend(object_info["category_path"])
    return list(set(category_list))


def collect_all_shapes(object_pool):
    uid_list = object_pool.uids
    shape_list = []
    for uid in uid_list:
        object_info = object_pool.get_object_info(uid)
        if object_info is None:
            continue
        shape_list.extend(object_info["shape"])
    return list(set(shape_list))


def collect_all_materials(object_pool):
    uid_list = object_pool.uids
    material_list = []
    for uid in uid_list:
        object_info = object_pool.get_object_info(uid)
        if object_info is None:
            continue
        material_list.extend(object_info["materials"])
    return list(set(material_list))


def collect_all_colors(object_pool):
    uid_list = object_pool.uids
    color_list = []
    for uid in uid_list:
        object_info = object_pool.get_object_info(uid)
        if object_info is None:
            continue
        color_list.extend(object_info["color"])
    return list(set(color_list))


def generate_long_horizon_by_shape(scene, usd_list, folder_path):
    category_list = collect_all_categories(
        apply_rule("can_grasp", usd_list, scene["object_pool"]), scene["object_pool"]
    )
    choose_category = random.choice(category_list)
    shape_list = collect_shapes_by_category(
        apply_rule("can_grasp", usd_list, scene["object_pool"]),
        choose_category,
        scene["object_pool"],
    )
    if len(shape_list) == 0:
        return None, None
    choose_shape = random.choice(shape_list)
    container_category_list = collect_all_categories(
        apply_rule("is_container", usd_list, scene["object_pool"]), scene["object_pool"]
    )
    container_choose_category = random.choice(container_category_list)
    replacement_config = {
        "obj1": {
            "max_cached_num": 10,
            "clip_range": {
                "min": 0.2,
                "max": 0.3,
            },
            "folder_path": folder_path,
            "option": [
                "adjust_thickness",
            ],
            "type": "rule",
            "rule": [
                f"retrieve_category_{choose_category}",
                f"retrieve_shape_{choose_shape}",
                "can_grasp",
            ],
        },
        "obj2": {
            "max_cached_num": 10,
            "clip_range": {
                "min": 0.35,
                "max": 0.45,
            },
            "folder_path": folder_path,
            "option": [
                "force_top",
            ],
            "type": "rule",
            "rule": [
                f"retrieve_category_{container_choose_category}",
                "is_container",
            ],
        },
        "background": {
            "add_num": 1,
            "max_cached_num": 10,
            "clip_range": {
                "min": 0.05,
                "max": 0.15,
            },
            "folder_path": folder_path,
            "type": "rule",
            "rule": [
                f"retrieve_not_shape_{choose_shape}",
                f"retrieve_category_{choose_category}",
            ],
        },
    }
    meta_info = {
        "obj1": choose_category,
        "obj2": container_choose_category,
        "type": "shape",
        "specific_info": choose_shape,
    }
    return replacement_config, meta_info


def generate_long_horizon_by_materials(scene, usd_list, folder_path):
    category_list = collect_all_categories(
        apply_rule("can_grasp", usd_list, scene["object_pool"]), scene["object_pool"]
    )
    choose_category = random.choice(category_list)
    material_list = collect_materials_by_category(
        apply_rule("can_grasp", usd_list, scene["object_pool"]),
        choose_category,
        scene["object_pool"],
    )
    if len(material_list) == 0:
        return None
    choose_material = random.choice(material_list)
    container_category_list = collect_all_categories(
        apply_rule("is_container", usd_list, scene["object_pool"]), scene["object_pool"]
    )
    container_choose_category = random.choice(container_category_list)
    replacement_config = {
        "obj1": {
            "max_cached_num": 10,
            "clip_range": {
                "min": 0.2,
                "max": 0.3,
            },
            "folder_path": folder_path,
            "option": [
                "adjust_thickness",
            ],
            "type": "rule",
            "rule": [
                f"retrieve_category_{choose_category}",
                f"retrieve_materials_{choose_material}",
                "can_grasp",
            ],
        },
        "obj2": {
            "max_cached_num": 10,
            "clip_range": {
                "min": 0.35,
                "max": 0.45,
            },
            "folder_path": folder_path,
            "option": [
                "force_top",
            ],
            "type": "rule",
            "rule": [
                f"retrieve_category_{container_choose_category}",
                "is_container",
            ],
        },
        "background": {
            "add_num": 1,
            "max_cached_num": 10,
            "clip_range": {
                "min": 0.05,
                "max": 0.15,
            },
            "folder_path": folder_path,
            "type": "rule",
            "rule": [
                f"retrieve_not_materials_{choose_material}",
                f"retrieve_category_{choose_category}",
            ],
        },
    }
    meta_info = {
        "obj1": choose_category,
        "obj2": container_choose_category,
        "type": "materials",
        "specific_info": choose_material,
    }
    return replacement_config, meta_info


def generate_long_horizon_by_color(scene, usd_list, folder_path):
    category_list = collect_all_categories(
        apply_rule("can_grasp", usd_list, scene["object_pool"]), scene["object_pool"]
    )
    choose_category = random.choice(category_list)
    color_list = collect_colors_by_category(
        apply_rule("can_grasp", usd_list, scene["object_pool"]),
        choose_category,
        scene["object_pool"],
    )
    if len(color_list) == 0:
        return None
    choose_color = random.choice(color_list)
    container_category_list = collect_all_categories(
        apply_rule("is_container", usd_list, scene["object_pool"]), scene["object_pool"]
    )
    container_choose_category = random.choice(container_category_list)
    replacement_config = {
        "obj1": {
            "max_cached_num": 10,
            "clip_range": {
                "min": 0.2,
                "max": 0.3,
            },
            "folder_path": folder_path,
            "option": [
                "adjust_thickness",
            ],
            "type": "rule",
            "rule": [
                f"retrieve_category_{choose_category}",
                f"retrieve_color_{choose_color}",
                "can_grasp",
            ],
        },
        "obj2": {
            "max_cached_num": 10,
            "clip_range": {
                "min": 0.35,
                "max": 0.45,
            },
            "folder_path": folder_path,
            "option": [
                "force_top",
            ],
            "type": "rule",
            "rule": [
                f"retrieve_category_{container_choose_category}",
                "is_container",
            ],
        },
        "background": {
            "add_num": 1,
            "max_cached_num": 10,
            "clip_range": {
                "min": 0.05,
                "max": 0.15,
            },
            "folder_path": folder_path,
            "type": "rule",
            "rule": [
                f"retrieve_not_color_{choose_color}",
                f"retrieve_category_{choose_category}",
            ],
        },
    }
    meta_info = {
        "obj1": choose_category,
        "obj2": container_choose_category,
        "type": "color",
        "specific_info": choose_color,
    }
    return replacement_config, meta_info


def generate_long_horizon_by_category(scene, usd_list, folder_path):
    category_list = collect_all_categories(
        apply_rule("can_grasp", usd_list, scene["object_pool"]), scene["object_pool"]
    )
    choose_category = random.choice(category_list)
    parent_category_list = find_parent_category(
        usd_list, choose_category, scene["object_pool"]
    )
    if len(parent_category_list) == 0:
        return None
    parent_choose_category = random.choice(parent_category_list)
    container_category_list = collect_all_categories(
        apply_rule("is_container", usd_list, scene["object_pool"]), scene["object_pool"]
    )
    container_choose_category = random.choice(container_category_list)
    replacement_config = {
        "obj1": {
            "max_cached_num": 10,
            "clip_range": {
                "min": 0.2,
                "max": 0.3,
            },
            "folder_path": folder_path,
            "option": [
                "adjust_thickness",
            ],
            "type": "rule",
            "rule": [
                f"retrieve_category_{choose_category}",
                f"retrieve_category_{parent_choose_category}",
                "can_grasp",
            ],
        },
        "obj2": {
            "max_cached_num": 10,
            "clip_range": {
                "min": 0.35,
                "max": 0.45,
            },
            "folder_path": folder_path,
            "option": [
                "force_top",
            ],
            "type": "rule",
            "rule": [
                f"retrieve_category_{container_choose_category}",
                "is_container",
            ],
        },
        "background": {
            "add_num": 1,
            "max_cached_num": 10,
            "clip_range": {
                "min": 0.05,
                "max": 0.15,
            },
            "folder_path": folder_path,
            "type": "rule",
            "rule": [
                f"retrieve_not_category_{choose_category}",
                f"retrieve_category_{parent_choose_category}",
            ],
        },
    }
    meta_info = {
        "obj1": choose_category,
        "obj2": container_choose_category,
        "type": "category",
        "specific_info": parent_choose_category,
    }
    return replacement_config, meta_info
