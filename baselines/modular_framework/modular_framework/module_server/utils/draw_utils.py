import numpy as np
import cv2
import logging

GRID_ROWS = 9
GRID_COLS = 16
GRID_ALPHA = 0.3


def show_anns(image, anns, borders=True, label_masks=True):
    """
    Overlay segmentation masks on an image, optionally drawing borders and labels on the masks.

    Args:
        image (np.ndarray): The input image on which masks will be overlaid. Expected to be a 3-channel image (RGB).
        anns (list of dict): A list of annotations where each annotation is a dictionary containing:
                             - 'segmentation': a binary mask (2D array) representing the object.
        borders (bool, optional): Whether to draw borders around the masks. Defaults to True.
        label_masks (bool, optional): Whether to label the masks with their index. Defaults to True.
    Returns:
        np.ndarray: The image with overlaid masks, borders, and labels (if specified).
    """
    image = image.astype(np.float32) / 255.0  # 归一化
    overlay = np.zeros_like(image)
    labels_to_draw = []

    for idx, ann in enumerate(anns):
        m = np.array(ann["segmentation"], dtype=bool)
        color_mask = np.random.random(3)
        overlay[m] = color_mask

        if borders:
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(overlay, contours, -1, (0, 0, 1), thickness=1)

        if label_masks:
            M = cv2.moments(m.astype(np.uint8))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                labels_to_draw.append((str(idx + 1), (cX, cY)))

    combined = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    combined_uint8 = (combined * 255).astype(np.uint8)

    for label, position in labels_to_draw:
        bg_color = image[position[1], position[0]]
        inverse_color = (
            255 - int(bg_color[0]),
            255 - int(bg_color[1]),
            255 - int(bg_color[2]),
        )
        cv2.putText(
            combined_uint8,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            inverse_color,
            2,
            cv2.LINE_AA,
        )
    return combined_uint8


def draw_grid(image, rows=GRID_ROWS, cols=GRID_COLS, alpha=GRID_ALPHA):
    """
    Draw a grid overlay on the given image and label each cell.

    Args:
        image (np.ndarray): The input image on which the grid will be drawn.
        rows (int, optional): The number of rows in the grid. Defaults to GRID_ROWS.
        cols (int, optional): The number of columns in the grid. Defaults to GRID_COLS.
        alpha (float, optional): The transparency factor for the grid overlay. Defaults to GRID_ALPHA.

    Returns:
        image (np.ndarray): The output image on which the grid is drawn.
    """
    overlay = image.copy()
    height, width = image.shape[:2]
    cell_width = width // cols
    cell_height = height // rows

    # 绘制水平线
    for r in range(1, rows):
        cv2.line(overlay, (0, r * cell_height), (width, r * cell_height), (0, 0, 0), 1)

    # 绘制垂直线
    for c in range(1, cols):
        cv2.line(overlay, (c * cell_width, 0), (c * cell_width, height), (0, 0, 0), 1)

    # 将叠加层与原始图像混合
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # 标注网格单元格
    rows_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    cols_labels = list(range(1, 17))
    for r in range(rows):
        for c in range(cols):
            label = f"{rows_labels[r]}{int(c)+1}"
            center_x = c * cell_width + cell_width // 2
            center_y = r * cell_height + cell_height // 2
            cv2.putText(
                image,
                label,
                (center_x - 10, center_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
    return image


def get_selected_grid_label(grab_point, image_rgb):
    absolute_x = int(grab_point[0])
    absolute_y = int(grab_point[1])
    height, width = image_rgb.shape[:2]
    cell_width = width // GRID_COLS
    cell_height = height // GRID_ROWS
    col = min(int(absolute_x) // cell_width, GRID_COLS - 1)
    row = min(int(absolute_y) // cell_height, GRID_ROWS - 1)
    selected_grid_label = (
        f"{['A','B','C','D','E','F','G','H','I'][int(row)]}{int(col)+1}"
    )
    logging.info(f"Selected Grid Label: {selected_grid_label}")
    return selected_grid_label
