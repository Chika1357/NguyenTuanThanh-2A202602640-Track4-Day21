"""Tạo ảnh demo + ảnh failure case cho REPORT (topic A).

    python -m src.make_figures

Sinh ra trong results/figures/:
  demo_3_distances.png          overlay điểm LiDAR lên object ở 3 khoảng cách (KITTI, calib gốc)
  demo_yaw0_vs_yaw2.png         cùng frame, calib gốc vs lệch yaw 2°
  edge_score_curve.png          edge-alignment score (cộng dồn 20 frame KITTI) theo yaw thử, ở 3 mức drift thật
  fail_01_nusc_scan_order.png   edge LiDAR của nuScenes khi đọc sai thứ tự điểm vs sau khi sắp theo ring
  fail_02_truncated_metric.png  object bị cắt ở mép ảnh: in-box ratio thấp dù calib đúng
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from starter.datasets import list_frames, load_frame
from starter.projection import (draw_box2d, overlay_points, perturb_extrinsic, project_velo_to_image,
                                velo_to_cam)
from src.calib_qa import (edge_alignment_score, image_edge_distance, in_bbox, lidar_depth_edges, load_ring,
                          object_inbox_ratio, points_in_box3d)

KITTI, NUSC = "data/kitti_mini", "data/nuscenes_mini_subset"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def put_title(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((34, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    return np.vstack([bar, img])


def overlay_frame(root: str, fid: str, **perturb) -> tuple[np.ndarray, dict]:
    fr = load_frame(root, fid)
    calib = perturb_extrinsic(fr["calib"], **perturb)
    uv, depth, _ = project_velo_to_image(fr["points"], calib, fr["image"].shape)
    vis = overlay_points(fr["image"], uv, depth)
    for obj in fr["labels"]:
        vis = draw_box2d(vis, obj.bbox, label=obj.type)
    return vis, fr


def demo_3_distances(out: Path) -> None:
    """Chọn tự động 1 xe gần (<12 m), 1 xe 20-30 m, 1 xe xa (>45 m) có >= 15 điểm, không bị che/cắt."""
    want = [(0, 12), (20, 30), (45, 80)]
    picks: list = [None] * 3
    for fid in list_frames(KITTI):
        fr = load_frame(KITTI, fid)
        cam = velo_to_cam(fr["points"][:, :3], fr["calib"])
        for obj in fr["labels"]:
            d = float(np.hypot(obj.location[0], obj.location[2]))
            for k, (lo, hi) in enumerate(want):
                if (picks[k] is None and obj.type == "Car" and lo <= d < hi and obj.truncated < 0.1
                        and obj.occluded == 0 and points_in_box3d(cam, obj).sum() >= 15):
                    picks[k] = (fid, obj, d)
    crops = []
    for fid, obj, d in picks:
        vis, _ = overlay_frame(KITTI, fid)
        x1, y1, x2, y2 = obj.bbox.astype(int)
        pad = max(20, (x2 - x1) // 2)
        crop = vis[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        crop = cv2.resize(crop, (int(crop.shape[1] * 260 / crop.shape[0]), 260), interpolation=cv2.INTER_NEAREST)
        crops.append(put_title(crop, f"{fid} Car {d:.0f} m"))
    cv2.imwrite(str(out / "demo_3_distances.png"), np.hstack([np.pad(c, ((0, 0), (0, 6), (0, 0)), constant_values=255)
                                                             for c in crops]))


def demo_yaw(out: Path, fid: str = "000011") -> None:
    a, fr = overlay_frame(KITTI, fid)
    b, _ = overlay_frame(KITTI, fid, yaw_deg=2.0)
    cv2.imwrite(str(out / "demo_yaw0_vs_yaw2.png"),
                np.vstack([put_title(a, f"KITTI {fid}: calib goc (yaw 0 deg)"),
                           put_title(b, f"KITTI {fid}: LiDAR lech yaw +2 deg -> diem truot ngang khoi nguoi/xe")]))


def edge_score_curve(out: Path) -> None:
    grid = np.arange(-4, 4 + 1e-9, 0.25)
    cache = []
    for fid in list_frames(KITTI):
        fr = load_frame(KITTI, fid)
        cache.append((fr["calib"], image_edge_distance(fr["image"]), lidar_depth_edges(fr["points"])))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for color, true_yaw in zip(SERIES, [0.0, 1.0, 2.0]):
        s = np.zeros(len(grid))
        for calib, dm, edges in cache:
            c = perturb_extrinsic(calib, yaw_deg=true_yaw)
            s += [edge_alignment_score(edges, dm, perturb_extrinsic(c, yaw_deg=g)) for g in grid]
        s /= len(cache)
        ax.plot(grid, s, color=color, lw=2, label=f"true drift {true_yaw:+.0f}°")
        ax.plot(grid[np.argmax(s)], s.max(), "o", ms=8, color=color, markeredgecolor="white", markeredgewidth=2)
    ax.axvline(0, color="#999", lw=1, ls="--")
    ax.set_xlabel("Yaw correction tried (°)  — peak at −drift = drift estimate")
    ax.set_ylabel("Mean edge-alignment score")
    ax.set_title("KITTI (20 frames): edge score peaks exactly at the correcting yaw", fontsize=11)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "edge_score_curve.png", dpi=130)
    plt.close(fig)


def fail_scan_order(out: Path, fid: str = "scene-0103_010") -> None:
    fr = load_frame(NUSC, fid)
    calib, img = fr["calib"], fr["image"]
    panels = []
    for ring, name in [(None, "SAI: thu tu file (theo cot) -> 'bien' = vet ngang mat duong"),
                       (load_ring(NUSC, fid), "DUNG: sap theo (ring, azimuth) -> bien o mep xe, cot")]:
        edges = lidar_depth_edges(fr["points"], ring=ring)
        uv, _, _ = project_velo_to_image(edges, calib, img.shape)
        vis = img.copy()
        for u, v in uv.astype(int):
            cv2.circle(vis, (int(u), int(v)), 3, (0, 0, 255), -1)
        panels.append(put_title(cv2.resize(vis, (800, 450)), f"{name} ({len(uv)} diem)"))
    cv2.imwrite(str(out / "fail_01_nusc_scan_order.png"), np.vstack(panels))


def fail_truncated(out: Path, fid: str = "000011") -> None:
    """Vẽ trên canvas nới rộng sang trái để thấy phần điểm của object bị chiếu RA NGOÀI ảnh."""
    fr = load_frame(KITTI, fid)
    calib, img = fr["calib"], fr["image"]
    obj = max(fr["labels"], key=lambda o: o.truncated)
    xyz = fr["points"][:, :3]
    cam = velo_to_cam(xyz, calib)
    pts = xyz[points_in_box3d(cam, obj) & (cam[:, 2] > 0.1)]
    ratio = object_inbox_ratio(pts, obj, calib, img.shape)
    pc = velo_to_cam(pts, calib)
    proj = np.hstack([pc, np.ones((len(pc), 1))]) @ calib.P2.T
    uv = proj[:, :2] / proj[:, 2:3]
    pad = int(min(max(0, -uv[:, 0].min()) + 20, 1500))
    canvas = np.full((img.shape[0], img.shape[1] + pad, 3), 90, np.uint8)
    canvas[:, pad:] = img
    inside = in_bbox(uv, obj.bbox)
    for (u, v), ok in zip(uv.astype(int), inside):
        if 0 <= v < img.shape[0] and -pad <= u < img.shape[1]:
            cv2.circle(canvas, (int(u) + pad, int(v)), 2, (0, 200, 0) if ok else (0, 0, 255), -1)
    x1, y1, x2, y2 = obj.bbox
    canvas = draw_box2d(canvas, (x1 + pad, y1, x2 + pad, y2), color=(0, 255, 255), label=f"{obj.type} trunc={obj.truncated:.2f}")
    cv2.line(canvas, (pad, 0), (pad, img.shape[0]), (255, 255, 255), 2)
    cv2.imwrite(str(out / "fail_02_truncated_metric.png"),
                put_title(canvas, f"KITTI {fid}: calib DUNG nhung chi {ratio:.0%} diem cua xe nam trong 2D box "
                                  f"(do = ngoai box/ngoai anh, vung xam = ngoai khung anh)"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Tạo ảnh demo và failure case cho REPORT topic A")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    demo_3_distances(out)
    demo_yaw(out)
    edge_score_curve(out)
    fail_scan_order(out)
    fail_truncated(out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
