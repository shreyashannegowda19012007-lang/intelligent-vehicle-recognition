"""
Bounding box coordinate transformation and utility functions.
Handles MATLAB 1-based indexing, YOLO normalized format, and IoU computation.
"""


def matlab_to_pixel_box(
    bbox_x1: int,
    bbox_y1: int,
    bbox_x2: int,
    bbox_y2: int,
    img_width: int,
    img_height: int,
    inclusive: bool = True,
) -> tuple[int, int, int, int]:
    """
    Convert 1-based MATLAB bounding box [bbox_x1, bbox_y1, bbox_x2, bbox_y2]
    to 0-based Python pixel coordinates.

    Args:
        bbox_x1: 1-based min x
        bbox_y1: 1-based min y
        bbox_x2: 1-based max x
        bbox_y2: 1-based max y
        img_width: Width of image in pixels
        img_height: Height of image in pixels
        inclusive: If True, returns inclusive [x1, y1, x2, y2] (e.g. for OpenCV).
                   If False, returns half-open [x1, y1, x2, y2] (e.g. for PIL crop).

    Returns:
        Tuple of (x1, y1, x2, y2) clamped within image dimensions.
    """
    x1 = max(0, int(bbox_x1) - 1)
    y1 = max(0, int(bbox_y1) - 1)

    if inclusive:
        x2 = min(img_width - 1, int(bbox_x2) - 1)
        y2 = min(img_height - 1, int(bbox_y2) - 1)
        x2 = max(x1, x2)
        y2 = max(y1, y2)
    else:
        x2 = min(img_width, int(bbox_x2))
        y2 = min(img_height, int(bbox_y2))
        x2 = max(x1 + 1, x2)
        y2 = max(y1 + 1, y2)

    return (x1, y1, x2, y2)


def box_to_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_width: int,
    img_height: int,
    inclusive: bool = True,
) -> tuple[float, float, float, float]:
    """
    Convert pixel bounding box [x1, y1, x2, y2] to YOLO normalized format:
    (x_center, y_center, width, height) in range [0, 1].

    In continuous coordinate space, pixels [x1..x2] inclusive span [x1, x2 + 1],
    so width is (x2 - x1 + 1) and center is x1 + width / 2.0.
    For half-open coordinates, width is (x2 - x1) and center is x1 + width / 2.0.

    Args:
        x1, y1, x2, y2: Pixel coordinates.
        img_width, img_height: Image dimensions.
        inclusive: Whether coordinates are inclusive.

    Returns:
        (x_center, y_center, width, height) normalized to [0, 1].
    """
    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"Invalid image dimensions: {img_width}x{img_height}")

    if inclusive:
        w = max(1.0, float(x2 - x1 + 1))
        h = max(1.0, float(y2 - y1 + 1))
    else:
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))

    xc = float(x1) + w / 2.0
    yc = float(y1) + h / 2.0

    xc_norm = min(1.0, max(0.0, xc / float(img_width)))
    yc_norm = min(1.0, max(0.0, yc / float(img_height)))
    w_norm = min(1.0, max(0.0, w / float(img_width)))
    h_norm = min(1.0, max(0.0, h / float(img_height)))

    return (round(xc_norm, 6), round(yc_norm, 6), round(w_norm, 6), round(h_norm, 6))


def yolo_to_box(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_width: int,
    img_height: int,
    inclusive: bool = True,
) -> tuple[int, int, int, int]:
    """
    Convert YOLO normalized format (x_center, y_center, width, height)
    to pixel bounding box [x1, y1, x2, y2].
    """
    w_px = width * img_width
    h_px = height * img_height
    xc_px = x_center * img_width
    yc_px = y_center * img_height

    x1_c = xc_px - w_px / 2.0
    y1_c = yc_px - h_px / 2.0
    x2_c = xc_px + w_px / 2.0
    y2_c = yc_px + h_px / 2.0

    if inclusive:
        x1 = max(0, min(img_width - 1, round(x1_c)))
        y1 = max(0, min(img_height - 1, round(y1_c)))
        x2 = max(x1, min(img_width - 1, round(x2_c - 1.0)))
        y2 = max(y1, min(img_height - 1, round(y2_c - 1.0)))
    else:
        x1 = max(0, min(img_width, round(x1_c)))
        y1 = max(0, min(img_height, round(y1_c)))
        x2 = max(x1 + 1, min(img_width, round(x2_c)))
        y2 = max(y1 + 1, min(img_height, round(y2_c)))

    return (x1, y1, x2, y2)


def compute_iou(
    box_a: list[float] | tuple[float, ...], box_b: list[float] | tuple[float, ...]
) -> float:
    """
    Compute Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2].
    """
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1 + 1.0)
    inter_h = max(0.0, inter_y2 - inter_y1 + 1.0)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1 + 1.0) * max(0.0, ay2 - ay1 + 1.0)
    area_b = max(0.0, bx2 - bx1 + 1.0) * max(0.0, by2 - by1 + 1.0)

    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)
