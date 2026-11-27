"""3D Geometry and Point Cloud Processing Utilities for LTC Mapping.

Handles ROS image encoding/decoding, ASUS Xtion depth parsing, pinhole back-projection,
voxel downsampling, statistical outlier removal, 2D convex hull footprint generation,
and 3D bounding box wireframe construction.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from sklearn.neighbors import NearestNeighbors


def ros_image_to_numpy(msg: Image) -> np.ndarray:
    """Convert sensor_msgs/Image to numpy array without external cv_bridge bugs."""
    enc = msg.encoding.lower()

    if enc in ('bgr8', 'rgb8'):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        if enc == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    elif enc in ('16uc1', 'mono16'):
        return np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
    elif enc in ('32fc1',):
        return np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
    elif enc in ('mono8', '8uc1'):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
    else:
        # Fallback raw byte interpretation
        return np.frombuffer(msg.data, dtype=np.uint8)


def numpy_to_ros_image(
    cv_img: np.ndarray, encoding: str = 'bgr8', header: Optional[Header] = None
) -> Image:
    """Convert numpy array to sensor_msgs/Image."""
    img_msg = Image()
    img_msg.height = int(cv_img.shape[0])
    img_msg.width = int(cv_img.shape[1])
    img_msg.encoding = encoding
    if header is not None:
        img_msg.header = header
    if cv_img.dtype.byteorder == '>':
        img_msg.is_bigendian = True

    contiguous = np.ascontiguousarray(cv_img)
    img_msg.data.frombytes(contiguous.tobytes())
    img_msg.step = len(img_msg.data) // img_msg.height
    return img_msg


def parse_depth_meters(depth_raw: np.ndarray, encoding: str) -> np.ndarray:
    """Convert raw ASUS Xtion depth frame into float32 meters.
    
    ASUS Xtion with OpenNI2 outputs 16UC1 (millimeters) or 32FC1 (meters).
    """
    enc = encoding.lower()
    if enc in ('16uc1', 'mono16') or depth_raw.dtype == np.uint16:
        depth_m = depth_raw.astype(np.float32) / 1000.0
    elif enc in ('32fc1',) or depth_raw.dtype == np.float32:
        depth_m = depth_raw.astype(np.float32)
    else:
        depth_m = depth_raw.astype(np.float32)

    # Invalidate 0, negative, and non-finite values (NaN, Inf)
    invalid = ~np.isfinite(depth_m) | (depth_m <= 0.0)
    depth_m[invalid] = np.nan
    return depth_m


def pinhole_backproject(
    depth_m: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_depth: float = 0.4,
    max_depth: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project 2D mask pixels to 3D camera optical coordinates.
    
    Formula:
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth

    Returns:
        points_xyz: (N, 3) float32 coordinates
        pixels_vu: (N, 2) int32 (row, col) coordinates
    """
    if fx <= 0.0 or fy <= 0.0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)

    if mask.ndim > 2:
        mask = mask.squeeze()

    if mask.shape[:2] != depth_m.shape[:2]:
        mask = cv2.resize(mask, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)

    valid_mask = (mask > 0) & np.isfinite(depth_m) & (depth_m >= min_depth) & (depth_m <= max_depth)
    v_idx, u_idx = np.where(valid_mask)

    if len(v_idx) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)

    z = depth_m[v_idx, u_idx]
    x = (u_idx - cx) * z / fx
    y = (v_idx - cy) * z / fy

    points_xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    pixels_vu = np.stack([v_idx, u_idx], axis=1).astype(np.int32)
    return points_xyz, pixels_vu


def pinhole_project_3d_to_2d(
    points_xyz: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D camera optical coordinates back to 2D pixel coordinates.

    Formula:
        u = (X * fx / Z) + cx
        v = (Y * fy / Z) + cy

    Returns:
        pixels_vu: (N, 2) float32 (row v, col u) coordinates
        valid_indices: (M,) int32 indices of points with Z > 0
    """
    if len(points_xyz) == 0 or fx <= 0.0 or fy <= 0.0:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int32)

    z = points_xyz[:, 2]
    valid = np.isfinite(z) & (z > 0.0)
    valid_indices = np.where(valid)[0]

    if len(valid_indices) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int32)

    z_val = z[valid_indices]
    x_val = points_xyz[valid_indices, 0]
    y_val = points_xyz[valid_indices, 1]

    u = (x_val * fx / z_val) + cx
    v = (y_val * fy / z_val) + cy

    pixels_vu = np.stack([v, u], axis=1).astype(np.float32)
    return pixels_vu, valid_indices


def voxel_downsample(
    points_xyz: np.ndarray, voxel_size: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample point cloud via uniform 3D voxel grid hashing."""
    if len(points_xyz) <= 1 or voxel_size <= 0.0:
        return points_xyz, np.arange(len(points_xyz))

    voxel_coords = np.floor(points_xyz / voxel_size).astype(np.int32)
    _, keep_indices = np.unique(voxel_coords, axis=0, return_index=True)
    return points_xyz[keep_indices], keep_indices


def statistical_outlier_removal(
    points_xyz: np.ndarray, k: int = 10, std_ratio: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter sensor noise and flying pixels using Statistical Outlier Removal."""
    if len(points_xyz) <= k:
        return points_xyz, np.arange(len(points_xyz))

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(points_xyz)
    distances, _ = nbrs.kneighbors(points_xyz)
    mean_dist = np.mean(distances, axis=1)

    mu = float(np.mean(mean_dist))
    sigma = float(np.std(mean_dist))
    threshold = mu + std_ratio * sigma

    inliers = np.where(mean_dist <= threshold)[0]
    return points_xyz[inliers], inliers


def compute_2d_footprint(
    points_xyz: np.ndarray, frame_type: str = 'optical', min_area: float = 1e-4
) -> Tuple[np.ndarray, float, float]:
    """Extract 2D convex hull polygon footprint from 3D object points.
    
    frame_type:
      'optical': (X, Z) represents horizontal floor plane (Y is gravity/downwards).
      'standard': (X, Y) represents horizontal floor plane (Z is gravity/upwards).
    
    Returns:
      vertices_2d: (P, 2) ordered polygon vertices
      area: surface area in m^2
      perimeter: perimeter length in meters
    """
    if len(points_xyz) < 3:
        return np.empty((0, 2), dtype=np.float32), 0.0, 0.0

    valid_finite = np.all(np.isfinite(points_xyz), axis=1)
    pts_clean = points_xyz[valid_finite]
    if len(pts_clean) < 3:
        return np.empty((0, 2), dtype=np.float32), 0.0, 0.0

    if frame_type == 'optical':
        pts_2d = pts_clean[:, [0, 2]]  # X (lateral), Z (depth)
    else:
        pts_2d = pts_clean[:, :2]      # X (forward), Y (lateral)

    pts_2d_f32 = np.ascontiguousarray(pts_2d, dtype=np.float32)
    hull = cv2.convexHull(pts_2d_f32)
    if hull is None or len(hull) < 3:
        return np.empty((0, 2), dtype=np.float32), 0.0, 0.0

    verts = hull.reshape(-1, 2)
    area = float(cv2.contourArea(verts))
    if area < min_area:
        return np.empty((0, 2), dtype=np.float32), 0.0, 0.0

    perimeter = float(cv2.arcLength(verts, closed=True))
    return verts, area, perimeter


def create_rounded_rectangle_footprint(
    cx: float,
    cy: float,
    w: float = 0.50,
    h: float = 0.50,
    angle_deg: float = 0.0,
    corner_r: float = 0.08,
    num_pts_per_corner: int = 3
) -> np.ndarray:
    """Generate a 2D convex polygon representing a smooth rounded rectangle footprint (e.g. for chairs).
    
    Used for objects like chairs to provide a realistic, compact physical 2D footprint (~0.50m x 0.50m)
    with smooth rounded corners matching Dengler et al. semantic map representations.
    """
    w_clean = float(np.clip(w, 0.48, 0.55))
    h_clean = float(np.clip(h, 0.48, 0.55))
    r = min(corner_r, w_clean / 2.0 - 0.01, h_clean / 2.0 - 0.01)

    centers = [
        (w_clean / 2.0 - r, h_clean / 2.0 - r, 0.0, np.pi / 2.0),
        (-w_clean / 2.0 + r, h_clean / 2.0 - r, np.pi / 2.0, np.pi),
        (-w_clean / 2.0 + r, -h_clean / 2.0 + r, np.pi, 3.0 * np.pi / 2.0),
        (w_clean / 2.0 - r, -h_clean / 2.0 + r, 3.0 * np.pi / 2.0, 2.0 * np.pi)
    ]
    local_pts = []
    for c_x, c_y, a_start, a_end in centers:
        angles = np.linspace(a_start, a_end, num_pts_per_corner, endpoint=True)
        for a in angles:
            local_pts.append([c_x + r * np.cos(a), c_y + r * np.sin(a)])

    local_pts_arr = np.array(local_pts, dtype=np.float32)
    rad = np.radians(angle_deg)
    cos_a, sin_a = float(np.cos(rad)), float(np.sin(rad))
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    rotated = local_pts_arr @ R.T
    world_pts = rotated + np.array([cx, cy], dtype=np.float32)

    hull = cv2.convexHull(np.ascontiguousarray(world_pts, dtype=np.float32), clockwise=False)
    if hull is not None and len(hull) >= 3:
        return hull.reshape(-1, 2)
    return np.ascontiguousarray(world_pts, dtype=np.float32)


def compute_footprint_3d(
    points_xyz: np.ndarray, frame_type: str = 'optical', min_area: float = 1e-4
) -> np.ndarray:
    """Extract corresponding 3D boundary vertices in sensor metric space.
    
    Extracts the 3D points corresponding to the 2D footprint convex boundary
    to ensure projection to map frame represents true ground boundary.
    
    Returns:
        vertices_3d: (P, 3) 3D coordinates of boundary vertices in sensor metric space
    """
    if len(points_xyz) < 3:
        return np.empty((0, 3), dtype=np.float32)

    valid_finite = np.all(np.isfinite(points_xyz), axis=1)
    pts_clean = points_xyz[valid_finite]
    if len(pts_clean) < 3:
        return np.empty((0, 3), dtype=np.float32)

    if frame_type == 'optical':
        pts_2d = pts_clean[:, [0, 2]]
    else:
        pts_2d = pts_clean[:, :2]

    pts_2d_f32 = np.ascontiguousarray(pts_2d, dtype=np.float32)
    hull_idx = cv2.convexHull(pts_2d_f32, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return np.empty((0, 3), dtype=np.float32)

    hull_idx = hull_idx.flatten()
    return pts_clean[hull_idx].astype(np.float32)


def compute_2d_obb(pts_2d: np.ndarray) -> Dict[str, Any]:
    """Compute 2D Oriented Bounding Box (OBB) using cv2.minAreaRect.
    
    Returns:
        Dict with center [cx, cy], size [w, h], yaw_rad, and corners (4, 2).
    """
    pts = np.ascontiguousarray(pts_2d, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return {
            'center': [0.0, 0.0],
            'size': [0.0, 0.0],
            'yaw_rad': 0.0,
            'corners': np.empty((0, 2), dtype=np.float32),
        }

    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle_deg = rect
    if w < h:
        w, h = h, w
        angle_deg += 90.0
    angle_deg = float(angle_deg % 180.0)
    yaw_rad = float(np.radians(angle_deg))
    box = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(angle_deg)))

    return {
        'center': [float(cx), float(cy)],
        'size': [float(w), float(h)],
        'yaw_deg': float(angle_deg),
        'yaw_rad': yaw_rad,
        'corners': box.astype(np.float32),
    }


def is_valid_polygon(vertices_2d: np.ndarray, min_area: float = 1e-4) -> bool:
    """Check if 2D polygon vertices form a non-degenerate closed polygon."""
    if vertices_2d is None or len(vertices_2d) < 3:
        return False
    if not np.all(np.isfinite(vertices_2d)):
        return False
    area = float(cv2.contourArea(vertices_2d.astype(np.float32)))
    return area >= min_area


def compute_3d_bbox(points_xyz: np.ndarray) -> Dict[str, Any]:
    """Compute 3D axis-aligned bounding box and wireframe edge segments."""
    if len(points_xyz) == 0:
        return {
            'center': [0.0, 0.0, 0.0],
            'dimensions': [0.0, 0.0, 0.0],
            'min_bound': [0.0, 0.0, 0.0],
            'max_bound': [0.0, 0.0, 0.0],
            'wireframe_lines': [],
        }

    min_b = np.min(points_xyz, axis=0)
    max_b = np.max(points_xyz, axis=0)
    center = (min_b + max_b) / 2.0
    dims = np.maximum(max_b - min_b, 0.02)  # Avoid zero-thickness visualization artifacts

    # 8 bounding box corners
    c = [
        [float(min_b[0]), float(min_b[1]), float(min_b[2])],  # 0
        [float(max_b[0]), float(min_b[1]), float(min_b[2])],  # 1
        [float(max_b[0]), float(max_b[1]), float(min_b[2])],  # 2
        [float(min_b[0]), float(max_b[1]), float(min_b[2])],  # 3
        [float(min_b[0]), float(min_b[1]), float(max_b[2])],  # 4
        [float(max_b[0]), float(min_b[1]), float(max_b[2])],  # 5
        [float(max_b[0]), float(max_b[1]), float(max_b[2])],  # 6
        [float(min_b[0]), float(max_b[1]), float(max_b[2])],  # 7
    ]

    # 12 wireframe edges (24 vertices pairs)
    edge_pairs = [
        (c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0]),  # Face min_z
        (c[4], c[5]), (c[5], c[6]), (c[6], c[7]), (c[7], c[4]),  # Face max_z
        (c[0], c[4]), (c[1], c[5]), (c[2], c[6]), (c[3], c[7]),  # Pillars
    ]

    return {
        'center': [round(float(center[0]), 4), round(float(center[1]), 4), round(float(center[2]), 4)],
        'dimensions': [round(float(dims[0]), 4), round(float(dims[1]), 4), round(float(dims[2]), 4)],
        'min_bound': [round(float(min_b[0]), 4), round(float(min_b[1]), 4), round(float(min_b[2]), 4)],
        'max_bound': [round(float(max_b[0]), 4), round(float(max_b[1]), 4), round(float(max_b[2]), 4)],
        'wireframe_lines': edge_pairs,
    }


def build_pointcloud2_msg(
    header: Header,
    points_xyz: np.ndarray,
    rgb_floats: np.ndarray,
    instance_ids: np.ndarray,
    class_ids: np.ndarray,
) -> PointCloud2:
    """Create structured PointCloud2 message with color, instance, and class fields."""
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='instance_id', offset=16, datatype=PointField.FLOAT32, count=1),
        PointField(name='class_id', offset=20, datatype=PointField.FLOAT32, count=1),
    ]

    n = len(points_xyz)
    if n == 0:
        return pc2.create_cloud(header, fields, np.empty((0, 6), dtype=np.float32))

    data_matrix = np.hstack([
        points_xyz.astype(np.float32),
        rgb_floats.reshape(-1, 1).astype(np.float32),
        instance_ids.reshape(-1, 1).astype(np.float32),
        class_ids.reshape(-1, 1).astype(np.float32),
    ])

    return pc2.create_cloud(header, fields, data_matrix)


def compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection-over-Union (IoU) between two [x1, y1, x2, y2] bounding boxes."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area1 = max(0.0, float(box1[2] - box1[0])) * max(0.0, float(box1[3] - box1[1]))
    area2 = max(0.0, float(box2[2] - box2[0])) * max(0.0, float(box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def clean_and_smooth_mask(m_bin: np.ndarray) -> np.ndarray:
    """Clean and smooth a binary mask (0/1 or 0/255 uint8).

    1. Removes isolated tiny speckles/islands.
    2. Fills internal pinholes and gaps.
    3. Regularizes boundaries using morphological smoothing and contour polygon approximation.
    """
    if m_bin is None or not np.any(m_bin):
        return m_bin

    mask_u8 = (m_bin > 0).astype(np.uint8)
    if np.sum(mask_u8) < 25:
        return mask_u8

    # 1. Fill holes and filter out small stray speckles using contours
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_u8

    areas = [cv2.contourArea(c) for c in contours]
    max_area = max(areas) if areas else 0
    if max_area < 25:
        return mask_u8

    cleaned = np.zeros_like(mask_u8)
    # Keep components that are at least 8% of max component area (minimum 25 pixels)
    min_keep_area = max(25.0, max_area * 0.08)
    for c in contours:
        if cv2.contourArea(c) >= min_keep_area:
            cv2.drawContours(cleaned, [c], -1, 1, thickness=cv2.FILLED)

    # 2. Morphological open (removes tiny spurs/hairs) and close (seals notches/pinholes)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

    # 3. Gaussian edge anti-jitter filter to prevent single-pixel edge flickering
    blurred = cv2.GaussianBlur(closed.astype(np.float32), (5, 5), sigmaX=1.0)
    thresh = (blurred >= 0.5).astype(np.uint8)

    # 4. Contour smoothing using polygon approximation
    smooth_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smoothed = np.zeros_like(mask_u8)
    for c in smooth_contours:
        if cv2.contourArea(c) < min_keep_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.003 * peri, True)
        cv2.drawContours(smoothed, [approx], -1, 1, thickness=cv2.FILLED)

    return smoothed if np.any(smoothed) else closed
