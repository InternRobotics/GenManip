import numpy as np


def iou(mask1, mask2):
    """
    Calculate the Intersection over Union (IoU) of two binary masks.

    Args:
        mask1 (np.ndarray): Mask 1
        mask2 (np.ndarray): Mask 2

    Returns:
        float: The IoU value, which is the ratio of the intersection area to the union area of the two masks.
               If the union area is zero, the function returns 0.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def nms(masks, iou_threshold=0.5, CtoF=False):
    """
    Perform Non-Maximum Suppression (NMS) on a list of masks to filter out overlapping masks. Additional area filter is added.

    Args:
        masks (list of dict): A list of dictionaries where each dictionary represents a mask.
                              Each dictionary should have at least two keys:
                              - 'segmentation': a binary mask (2D array) where the object of interest is represented by 1s.
                              - 'area': the area of the mask (number of 1s in the segmentation).
        iou_threshold (float, optional): The IoU threshold for suppressing overlapping masks.
                                         Masks with IoU greater than this threshold will be suppressed.
                                         Defaults to 0.5.
        CtoF (bool, optional): Whether it is CtoF mode. Defaults to False.

    Returns:
        list of dict: A list of dictionaries representing the masks that are kept after NMS.
    """
    if not masks:
        return []

    if not CtoF:
        masks = [mask for mask in masks if mask["area"] <= 200000]
    masks = sorted(masks, key=lambda x: x["area"], reverse=True)
    keep = []

    while masks:
        current = masks.pop(0)
        keep.append(current)

        masks = [
            mask
            for mask in masks
            if iou(
                np.array(current["segmentation"], dtype=bool),
                np.array(mask["segmentation"], dtype=bool),
            )
            < iou_threshold
        ]

    return keep
