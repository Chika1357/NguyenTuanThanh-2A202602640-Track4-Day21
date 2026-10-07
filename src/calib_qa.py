"""Các metric QA cho calibration LiDAR-camera (topic A).

1. object_inbox_ratio: lấy điểm LiDAR nằm trong 3D box GT (gán bằng calib GỐC), chiếu bằng calib
   bị perturb, đo % điểm còn rơi vào 2D box của chính object đó. Metric cần label -> dùng offline.
2. edge_alignment_score: không cần label. Lấy điểm LiDAR nằm ở biên độ sâu (depth discontinuity,
   ví dụ mép xe, mép cột) và đo xem chúng có rơi gần cạnh ảnh (Canny) không. Ý tưởng theo
   Levinson & Thrun, "Automatic Online Calibration of Cameras and Lasers", RSS 2013.
3. estimate_yaw_drift: quét lưới yaw quanh calib hiện tại, lấy yaw cho edge score lớn nhất.
   Nếu calib đúng thì argmax ~ 0; nếu LiDAR bị lệch +d thì argmax ~ -d.
"""
from __future__ import annotations

import cv2
import numpy as np

from starter.kitti_io import KittiCalib, KittiObject
from starter.projection import cam_to_image, perturb_extrinsic, velo_to_cam


def points_in_box3d(points_cam: np.ndarray, obj: KittiObject) -> np.ndarray:
    """Mask (N,) các điểm (camera frame) nằm trong 3D box KITTI (location = tâm đáy, y hướng xuống)."""
    h, w, l = obj.dimensions
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])      # cùng quy ước với box3d_corners_cam
    local = (points_cam - obj.location) @ R                 # = R^T (p - loc) cho từng hàng
    return ((np.abs(local[:, 0]) <= l / 2) & (np.abs(local[:, 2]) <= w / 2)
            & (local[:, 1] <= 0) & (local[:, 1] >= -h))


def in_bbox(uv: np.ndarray, bbox) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2)


def object_point_sets(points: np.ndarray, calib: KittiCalib, labels: list[KittiObject], image_shape,
                      min_points: int = 10, require_in_bbox: bool = True) -> list[tuple[KittiObject, np.ndarray]]:
    """Với mỗi object: tập điểm velodyne (M, 3) nằm trong 3D box GT, dùng calib gốc (không perturb).

    require_in_bbox=True: chỉ giữ điểm mà với calib gốc đã rơi vào 2D box (tập tham chiếu).
    Nếu False, object bị cắt ở mép ảnh (truncated) có nhiều điểm ngoài ảnh và ratio thấp ngay cả
    khi calib đúng -> metric báo nhầm drift (xem failure case fail_02 trong REPORT)."""
    xyz = points[np.isfinite(points[:, :3]).all(axis=1), :3]
    cam = velo_to_cam(xyz, calib)
    uv_all = np.full((len(xyz), 2), -1.0)
    uv, _, mask = cam_to_image(cam, calib.P2, image_shape)
    uv_all[mask] = uv
    out = []
    for obj in labels:
        m = points_in_box3d(cam, obj)
        if require_in_bbox:
            m &= mask & in_bbox(uv_all, obj.bbox)
        if m.sum() >= min_points:
            out.append((obj, xyz[m]))
    return out


def object_inbox_ratio(obj_pts_velo: np.ndarray, obj: KittiObject, calib: KittiCalib,
                       image_shape) -> float:
    """% điểm của object (sau khi chiếu bằng `calib`) rơi vào 2D box của object.
    Điểm bị chiếu ra ngoài ảnh hoặc ra sau camera được tính là rơi ra ngoài box."""
    uv, _, _ = cam_to_image(velo_to_cam(obj_pts_velo, calib), calib.P2, image_shape)
    return float(in_bbox(uv, obj.bbox).sum() / len(obj_pts_velo))


def fov_ratio(points: np.ndarray, calib: KittiCalib, image_shape) -> float:
    """% điểm phía trước (x_velo > 0 với KITTI) rơi vào ảnh. Dùng toàn bộ điểm hữu hạn làm mẫu số."""
    xyz = points[np.isfinite(points[:, :3]).all(axis=1), :3]
    _, _, mask = cam_to_image(velo_to_cam(xyz, calib), calib.P2, image_shape)
    return float(mask.mean())


# ----------------------------- edge alignment (không cần label) -----------------------------

def image_edge_distance(image: np.ndarray, canny_lo: int = 50, canny_hi: int = 150) -> np.ndarray:
    """Ảnh (H, W) float: khoảng cách (pixel) từ mỗi pixel tới cạnh Canny gần nhất."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    return cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)


def lidar_depth_edges(points: np.ndarray, jump_m: float = 1.0, ring: np.ndarray | None = None) -> np.ndarray:
    """Điểm (K, 3) nằm ở mép TRƯỚC của một bước nhảy độ sâu.
    Giả định: hai điểm liên tiếp là hai tia kề nhau theo phương ngang trên cùng ring, nên range chênh
    lớn = biên vật thể. Chỉ giữ điểm gần hơn (mép vật, không phải nền phía sau).

    KITTI lưu điểm theo ring nên giả định đúng sẵn. nuScenes lưu theo cột (32 ring của cùng một azimuth
    liền nhau) -> phải truyền `ring` để sắp xếp lại theo (ring, azimuth), nếu không score vô nghĩa."""
    finite = np.isfinite(points[:, :3]).all(axis=1)
    xyz = points[finite, :3]
    if ring is not None:
        az = np.arctan2(xyz[:, 1], xyz[:, 0])
        xyz = xyz[np.lexsort((az, ring[finite]))]
    r = np.linalg.norm(xyz, axis=1)
    prev_gap = np.r_[0.0, r[:-1] - r[1:]]   # r[i-1] - r[i] > 0: điểm i gần hơn điểm trước nó
    next_gap = np.r_[r[1:] - r[:-1], 0.0]   # r[i+1] - r[i] > 0: điểm i gần hơn điểm sau nó
    return xyz[np.maximum(prev_gap, next_gap) > jump_m]


def edge_alignment_score(edge_pts_velo: np.ndarray, dist_map: np.ndarray, calib: KittiCalib,
                         sigma_px: float = 2.0) -> float:
    """Trung bình exp(-d / sigma) với d = khoảng cách tới cạnh ảnh gần nhất. 1 = khớp hoàn hảo."""
    uv, _, _ = cam_to_image(velo_to_cam(edge_pts_velo, calib), calib.P2, dist_map.shape)
    if len(uv) == 0:
        return float("nan")
    d = dist_map[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    return float(np.exp(-d / sigma_px).mean())


def estimate_yaw_drift(edge_pts_velo: np.ndarray, dist_map: np.ndarray, calib: KittiCalib,
                       search_deg: float = 4.0, step_deg: float = 0.25) -> tuple[float, float, np.ndarray]:
    """Quét yaw trong [-search, +search]. Trả về (yaw cho score max, score tại 0, mảng score).
    Drift ước lượng = -argmax (vì cần xoay ngược lại để bù)."""
    grid = np.arange(-search_deg, search_deg + 1e-9, step_deg)
    scores = np.array([edge_alignment_score(edge_pts_velo, dist_map, perturb_extrinsic(calib, yaw_deg=g))
                       for g in grid])
    best = float(grid[int(np.nanargmax(scores))])
    return best, float(scores[np.argmin(np.abs(grid))]), scores


def load_ring(data_root, frame_id: str) -> np.ndarray | None:
    """Ring index của từng điểm (chỉ nuScenes có; cột thứ 5 của .pcd.bin). KITTI trả None."""
    from starter.datasets import dataset_type
    from starter.nuscenes_io import lidar_path
    if dataset_type(data_root) != "nuscenes":
        return None
    return np.fromfile(lidar_path(data_root, frame_id), dtype=np.float32).reshape(-1, 5)[:, 4]
