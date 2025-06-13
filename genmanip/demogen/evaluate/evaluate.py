import numpy as np
import scipy
from shapely.geometry import Point
from shapely.vectorized import contains
from sklearn.neighbors import NearestNeighbors

from genmanip.core.usd_utils.prim_utils import get_pcd_from_mesh
from genmanip.utils.object_utils.shelf import extract_shelf_planes
from genmanip.utils.pc_utils import get_xy_contour
from genmanip.utils.utils import compare_articulation_status

XY_DISTANCE_CLOSE_THRESHOLD = 0.15
MAX_TO_BE_TOUCHING_DISTANCE = 0.1
MIN_ABOVE_BELOW_DISTANCE = 0.05
MAX_TO_BE_SUPPORTING_AREA_RATIO = 1.5
MIN_TO_BE_SUPPORTED_AREA_RATIO = 0.7
MIN_TO_BE_ABOVE_BELOW_AREA_RATIO = 0.1
INSIDE_PROPORTION_THRESH = 0.5
ANGLE_THRESHOLD = 45


def assign_point_to_shelf(point, shelves):
    point_xyz = np.array(point)
    point_x, point_y, point_z = point_xyz[:3]
    for i, shelf in enumerate(shelves):
        if shelf.z_min < point_z <= shelf.z_max:
            point_xy = (point_x, point_y)
            if shelf.boundary_polygon.contains(Point(point_xy)):
                return i + 1
    return 0


def calculate_distance_between_two_point_clouds(point_cloud_a, point_cloud_b):
    nn = NearestNeighbors(n_neighbors=1).fit(point_cloud_a)
    distances, _ = nn.kneighbors(point_cloud_b)
    res = np.min(distances)
    return res


def calculate_xy_distance_between_two_point_clouds(point_cloud_a, point_cloud_b):
    point_cloud_a = point_cloud_a[:, :2]
    point_cloud_b = point_cloud_b[:, :2]
    nn = NearestNeighbors(n_neighbors=1).fit(point_cloud_a)
    distances, _ = nn.kneighbors(point_cloud_b)
    res = np.min(distances)
    return res


def check_finished(goals, pclist, articulation_list=[]):
    max_sr = 0
    for goal in goals:
        sr = 0
        for subgoal in goal:
            if "position" in subgoal:
                if "another_obj2_uid" in subgoal:
                    pcd3 = pclist[subgoal["another_obj2_uid"]]
                else:
                    pcd3 = None
                if check_subgoal_finished_rigid(
                    subgoal, pclist[subgoal["obj1_uid"]], pclist[subgoal["obj2_uid"]], pcd3
                ):
                    sr += 1 / len(goal)
            elif "status" in subgoal:
                if check_subgoal_finished_articulation(
                    subgoal, articulation_list[subgoal["obj1_uid"]]
                ):
                    sr += 1 / len(goal)
        max_sr = max(max_sr, sr)
    return max_sr


def check_subgoal_finished_articulation(subgoal, articulation):
    subgoal_status = subgoal["status"]
    articulation_status = articulation._articulation_view.get_joints_state().positions
    for status in subgoal_status:
        if compare_articulation_status(articulation_status.tolist(), status):
            return True
    return False


def crop_pcd(pcd1, pcd2):
    contour1 = get_xy_contour(pcd1, contour_type="concave_hull").buffer(0.05)
    xy_points = pcd2[:, :2]
    mask = contains(contour1, xy_points[:, 0], xy_points[:, 1])
    return pcd2[mask]


def check_subgoal_finished_rigid(subgoal, pcd1, pcd2, pcd3=None):
    relation_list = get_related_position(pcd1, pcd2, pcd3)
    if subgoal["position"] == "top" or subgoal["position"] == "on":
        croped_pcd2 = crop_pcd(pcd1, pcd2)
        if len(croped_pcd2) > 0:
            relation_list_2 = get_related_position(pcd1, croped_pcd2)
            if "on" in relation_list_2:
                return True
    if subgoal["position"] == "top" or subgoal["position"] == "on":
        if "on" not in relation_list and "in" not in relation_list:
            return False
    else:
        if subgoal["position"] not in relation_list:
            return False
    return True


def get_related_position(pcd1, pcd2, pcd3=None):
    max_pcd1 = np.max(pcd1, axis=0)
    min_pcd1 = np.min(pcd1, axis=0)
    max_pcd2 = np.max(pcd2, axis=0)
    min_pcd2 = np.min(pcd2, axis=0)
    return infer_spatial_relationship(
        pcd1, pcd2, min_pcd1, max_pcd1, min_pcd2, max_pcd2, pcd3
    )


def infer_spatial_relationship(
    point_cloud_a,
    point_cloud_b,
    min_points_a,
    max_points_a,
    min_points_b,
    max_points_b,
    point_cloud_c=None,
    error_margin_percentage=0.01,
):
    relation_list = []
    if point_cloud_c is None:
        xy_dist = calculate_xy_distance_between_two_point_clouds(
            point_cloud_a, point_cloud_b
        )
        if xy_dist > XY_DISTANCE_CLOSE_THRESHOLD * (1 + error_margin_percentage):
            return []
        dist = calculate_distance_between_two_point_clouds(point_cloud_a, point_cloud_b)
        a_bottom_b_top_dist = min_points_b[2] - max_points_a[2]
        a_top_b_bottom_dist = min_points_a[2] - max_points_b[2]
        if dist < MAX_TO_BE_TOUCHING_DISTANCE * (1 + error_margin_percentage):
            if is_inside(
                src_pts=point_cloud_a,
                target_pts=point_cloud_b,
                thresh=INSIDE_PROPORTION_THRESH,
            ):
                relation_list.append("in")
            elif is_inside(
                src_pts=point_cloud_b,
                target_pts=point_cloud_a,
                thresh=INSIDE_PROPORTION_THRESH,
            ):
                relation_list.append("out of")
            # on, below
            iou_2d, i_ratios, a_ratios = iou_2d_via_boundaries(
                min_points_a, max_points_a, min_points_b, max_points_b
            )
            i_target_ratio, i_anchor_ratio = i_ratios
            target_anchor_area_ratio, anchor_target_area_ratio = a_ratios
            # Target/a supported-by the anchor/b
            a_supported_by_b = False
            if (
                i_target_ratio > MIN_TO_BE_SUPPORTED_AREA_RATIO
                and abs(a_top_b_bottom_dist)
                <= MAX_TO_BE_TOUCHING_DISTANCE * (1 + error_margin_percentage)
                and target_anchor_area_ratio < MAX_TO_BE_SUPPORTING_AREA_RATIO
            ):
                a_supported_by_b = True
            # Target/a supporting the anchor/b
            a_supporting_b = False
            if (
                i_anchor_ratio > MIN_TO_BE_SUPPORTED_AREA_RATIO
                and abs(a_bottom_b_top_dist)
                <= MAX_TO_BE_TOUCHING_DISTANCE * (1 + error_margin_percentage)
                and anchor_target_area_ratio < MAX_TO_BE_SUPPORTING_AREA_RATIO
            ):
                a_supporting_b = True
            if a_supported_by_b:
                relation_list.append("on")
            elif a_supporting_b:
                relation_list.append("below")
            else:
                relation_list.append("near")

        if xy_dist <= XY_DISTANCE_CLOSE_THRESHOLD * (1 + error_margin_percentage):
            x_overlap = (
                (min_points_a[0] <= max_points_b[0] <= max_points_a[0])
                or (min_points_a[0] <= min_points_b[0] <= max_points_a[0])
                or (min_points_b[0] <= min_points_a[0] <= max_points_b[0])
                or (min_points_b[0] <= max_points_a[0] <= max_points_b[0])
            )
            y_overlap = (
                (min_points_a[1] <= max_points_b[1] <= max_points_a[1])
                or (min_points_a[1] <= min_points_b[1] <= max_points_a[1])
                or (min_points_b[1] <= min_points_a[1] <= max_points_b[1])
                or (min_points_b[1] <= max_points_a[1] <= max_points_b[1])
            )
            if x_overlap and y_overlap:
                # If there is overlap on both X and Y axes, classify as "near"
                if "near" not in relation_list:
                    relation_list.append("near")
            elif x_overlap:
                # Objects are close in the X axis; determine Left-Right relationship
                if max_points_a[1] < min_points_b[1]:
                    relation_list.append("left")
                elif max_points_b[1] < min_points_a[1]:
                    relation_list.append("right")
            elif y_overlap:
                # Objects are close in the Y axis; determine Front-Back relationship
                if max_points_a[0] < min_points_b[0]:
                    relation_list.append("front")
                elif max_points_b[0] < min_points_a[0]:
                    relation_list.append("back")
    else:

        def compute_centroid(point_cloud):
            return np.mean(point_cloud, axis=0)

        anchor1_center = compute_centroid(point_cloud_b)
        anchor2_center = compute_centroid(point_cloud_c)
        target_center = compute_centroid(point_cloud_a)
        vector1 = target_center - anchor1_center
        vector2 = anchor2_center - target_center
        vector1_norm = vector1 / np.linalg.norm(vector1)
        vector2_norm = vector2 / np.linalg.norm(vector2)
        cosine_angle = np.dot(vector1_norm, vector2_norm)
        angle = np.degrees(np.arccos(cosine_angle))
        if angle < ANGLE_THRESHOLD:
            relation_list.append("between")
    return relation_list


def initialize_shelves(mesh):
    pcd = get_pcd_from_mesh(mesh, num_points=100000)
    shelves = extract_shelf_planes(pcd)
    return shelves


def iou_2d_via_boundaries(min_points_a, max_points_a, min_points_b, max_points_b):
    a_xmin, a_xmax, a_ymin, a_ymax = (
        min_points_a[0],
        max_points_a[0],
        min_points_a[1],
        max_points_a[1],
    )
    b_xmin, b_xmax, b_ymin, b_ymax = (
        min_points_b[0],
        max_points_b[0],
        min_points_b[1],
        max_points_b[1],
    )

    box_a = [a_xmin, a_ymin, a_xmax, a_ymax]
    box_b = [b_xmin, b_ymin, b_xmax, b_ymax]
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    if box_a_area + box_b_area - inter_area == 0:
        iou = 0
    else:
        iou = inter_area / float(box_a_area + box_b_area - inter_area)
    if box_a_area == 0 or box_b_area == 0:
        i_ratios = [0, 0]
        a_ratios = [0, 0]
    else:
        i_ratios = [inter_area / float(box_a_area), inter_area / float(box_b_area)]
        a_ratios = [box_a_area / box_b_area, box_b_area / box_a_area]

    return iou, i_ratios, a_ratios


def is_point_in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, scipy.spatial.Delaunay):
        hull = scipy.spatial.Delaunay(hull)

    return hull.find_simplex(p) >= 0


def is_inside(src_pts, target_pts, thresh=0.5):
    try:
        hull = scipy.spatial.ConvexHull(target_pts)
    except:
        return False
    # print("vertices of hull: ", np.array(hull.vertices).shape)
    hull_vertices = np.array([[0, 0, 0]])
    for v in hull.vertices:
        try:
            hull_vertices = np.vstack(
                (
                    hull_vertices,
                    np.array([target_pts[v, 0], target_pts[v, 1], target_pts[v, 2]]),
                )
            )
        except:
            import pdb

            pdb.set_trace()
    hull_vertices = hull_vertices[1:]

    num_src_pts = len(src_pts)
    # Don't want threshold to be too large (specially with more objects, like 4, 0.9*thresh becomes too large)
    thresh_obj_particles = thresh * num_src_pts
    src_points_in_hull = is_point_in_hull(src_pts, hull_vertices)
    # print("src pts in target, thresh: ", src_points_in_hull.sum(), thresh_obj_particles)
    if src_points_in_hull.sum() > thresh_obj_particles:
        return True
    else:
        return False
