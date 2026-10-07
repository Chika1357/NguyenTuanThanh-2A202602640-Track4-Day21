"""Thí nghiệm chính topic A: calibration drift -> projection mismatch.

Thí nghiệm 1 (có label): perturb extrinsic theo MỘT trục mỗi lần (yaw / pitch / roll / dịch ngang),
đo % điểm LiDAR của object còn rơi vào 2D box của nó, chia theo khoảng cách.
Thí nghiệm 2 (không label): ước lượng yaw drift bằng edge-alignment score, từng frame và cộng dồn
theo cửa sổ (mỗi scene / cả tập), so sánh có và không sắp xếp lại điểm theo ring.

Không có phép ngẫu nhiên nào -> chạy lại luôn ra cùng số.

    python -m src.sweep --data-root data/kitti_mini --tag kitti
    python -m src.sweep --data-root data/nuscenes_mini_subset --tag nusc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from starter.datasets import dataset_type, list_frames, load_frame
from starter.projection import perturb_extrinsic
from src.calib_qa import (estimate_yaw_drift, fov_ratio, image_edge_distance, lidar_depth_edges, load_ring,
                          object_inbox_ratio, object_point_sets)

ROT_LEVELS = [0.0, 0.5, 1.0, 2.0, 3.0]      # độ
TRANS_LEVELS = [0.0, 0.02, 0.05, 0.10]      # mét
DIST_BINS = [0, 15, 30, np.inf]
DIST_LABELS = ["<15 m", "15-30 m", ">30 m"]
MISMATCH_THR = 0.9                          # object "lệch" nếu < 90% điểm còn trong 2D box
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # thứ tự cố định: gần / trung bình / xa


def perturbations(ds: str) -> list[tuple[str, float, dict]]:
    """(tên trục, mức, kwargs cho perturb_extrinsic). Dịch ngang = trục y (trái) ở KITTI, trục x ở nuScenes."""
    out = []
    for axis in ("yaw", "pitch", "roll"):
        out += [(axis, lv, {f"{axis}_deg": lv}) for lv in ROT_LEVELS]
    lat = 0 if ds == "nuscenes" else 1
    for lv in TRANS_LEVELS:
        t = [0.0, 0.0, 0.0]
        t[lat] = lv
        out.append(("lateral_t", lv, {"t_xyz_m": tuple(t)}))
    return out


def run_inbox(root: str, frames: list[str], ds: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    obj_rows, fov_rows = [], []
    for fid in frames:
        fr = load_frame(root, fid)
        calib, shape = fr["calib"], fr["image"].shape
        objs = object_point_sets(fr["points"], calib, fr["labels"], shape)
        for axis, lv, kw in perturbations(ds):
            pc = perturb_extrinsic(calib, **kw)
            fov_rows.append({"frame": fid, "axis": axis, "level": lv, "fov_ratio": fov_ratio(fr["points"], pc, shape)})
            for i, (obj, pts) in enumerate(objs):
                obj_rows.append({"frame": fid, "obj_idx": i, "type": obj.type,
                                 "distance_m": float(np.hypot(obj.location[0], obj.location[2])),
                                 "n_points": len(pts), "axis": axis, "level": lv,
                                 "inbox_ratio": object_inbox_ratio(pts, obj, pc, shape)})
    return pd.DataFrame(obj_rows), pd.DataFrame(fov_rows)


def summarize(objs: pd.DataFrame, fov: pd.DataFrame) -> pd.DataFrame:
    objs = objs.assign(dist_bin=pd.cut(objs.distance_m, DIST_BINS, labels=DIST_LABELS, right=False))
    g = objs.groupby(["axis", "level", "dist_bin"], observed=True).inbox_ratio
    s = pd.DataFrame({"n_objects": g.size(), "mean_inbox_ratio": g.mean(),
                      "mismatch_rate": g.apply(lambda x: float((x < MISMATCH_THR).mean()))}).reset_index()
    allbin = objs.groupby(["axis", "level"]).inbox_ratio
    s_all = pd.DataFrame({"n_objects": allbin.size(), "mean_inbox_ratio": allbin.mean(),
                          "mismatch_rate": allbin.apply(lambda x: float((x < MISMATCH_THR).mean()))}).reset_index()
    s_all["dist_bin"] = "all"
    s = pd.concat([s, s_all], ignore_index=True)
    fov_mean = fov.groupby(["axis", "level"]).fov_ratio.mean().rename("mean_fov_ratio").reset_index()
    return s.merge(fov_mean, on=["axis", "level"]).round(4)


def run_edge(root: str, frames: list[str], levels=(0.0, 0.5, 1.0, 2.0, 3.0)) -> pd.DataFrame:
    """Ước lượng yaw drift từng frame + cộng dồn theo window (scene với nuScenes, cả tập với KITTI)."""
    grid = np.arange(-4, 4 + 1e-9, 0.25)
    rows = []
    for ring_sort in ([False, True] if dataset_type(root) == "nuscenes" else [False]):
        cache = {}
        for fid in frames:
            fr = load_frame(root, fid)
            ring = load_ring(root, fid) if ring_sort else None
            window = fid.rsplit("_", 1)[0] if dataset_type(root) == "nuscenes" else "kitti_all"
            cache[fid] = (window, fr["calib"], image_edge_distance(fr["image"]),
                          lidar_depth_edges(fr["points"], ring=ring))
        for lv in levels:
            sums: dict[str, np.ndarray] = {}
            for fid, (window, calib, dm, edges) in cache.items():
                best, _, scores = estimate_yaw_drift(edges, dm, perturb_extrinsic(calib, yaw_deg=lv))
                sums[window] = sums.get(window, 0) + np.nan_to_num(scores)
                rows.append({"ring_sort": ring_sort, "window": window, "frame": fid, "true_yaw_deg": lv,
                             "est_drift_deg": -best, "abs_err_deg": abs(-best - lv)})
            for window, s in sums.items():
                best = float(grid[int(np.argmax(s))])
                rows.append({"ring_sort": ring_sort, "window": window, "frame": "WINDOW", "true_yaw_deg": lv,
                             "est_drift_deg": -best, "abs_err_deg": abs(-best - lv)})
    return pd.DataFrame(rows)


def plot_inbox(summary: pd.DataFrame, tag: str, out: Path, ds: str = "kitti") -> None:
    # "Pitch của xe" = quay quanh trục ngang của LiDAR: trục y (trái) ở KITTI, trục x (phải) ở nuScenes.
    # perturb_extrinsic gọi quay quanh y là "pitch", nên với nuScenes phải lấy dòng "roll".
    pitch_axis = "roll" if ds == "nuscenes" else "pitch"
    axes_order = [("yaw", "Yaw drift (°)"), (pitch_axis, "Vehicle-pitch drift (°)"), ("lateral_t", "Lateral shift (m)")]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (axis, xlabel) in zip(axs, axes_order):
        for color, b in zip(SERIES, DIST_LABELS):
            d = summary[(summary.axis == axis) & (summary.dist_bin == b)].sort_values("level")
            ax.plot(d.level, d.mean_inbox_ratio * 100, color=color, lw=2, marker="o", ms=6, label=b)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("% object points inside own 2D box")
    axs[0].set_ylim(0, 102)
    axs[0].legend(title="Object distance", frameon=False, loc="lower left")
    fig.suptitle(f"{tag}: projection mismatch vs calibration drift (one axis perturbed at a time)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep calibration drift (yaw/pitch/roll/dịch ngang), đo in-box ratio "
                                             "và độ chính xác ước lượng drift bằng edge-alignment score")
    ap.add_argument("--data-root", default="data/kitti_mini", help="data/kitti_mini hoặc data/nuscenes_mini_subset")
    ap.add_argument("--tag", default=None, help="tiền tố tên file kết quả (mặc định: kitti / nusc)")
    ap.add_argument("--frames", nargs="*", default=None, help="danh sách frame id (mặc định: tất cả)")
    ap.add_argument("--out-dir", default="results", help="thư mục ghi CSV; ảnh ghi vào <out-dir>/figures")
    ap.add_argument("--skip-edge", action="store_true", help="bỏ thí nghiệm 2 (edge-alignment) cho nhanh")
    args = ap.parse_args()

    ds = dataset_type(args.data_root)
    tag = args.tag or ("nusc" if ds == "nuscenes" else "kitti")
    frames = args.frames or list_frames(args.data_root)
    out = Path(args.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    objs, fov = run_inbox(args.data_root, frames, ds)
    objs.round(4).to_csv(out / f"{tag}_inbox_sweep.csv", index=False)
    summary = summarize(objs, fov)
    summary.to_csv(out / f"{tag}_inbox_summary.csv", index=False)
    plot_inbox(summary, tag, out / "figures" / f"{tag}_inbox_vs_drift.png", ds)
    print(summary[summary.dist_bin == "all"].to_string(index=False))

    if not args.skip_edge:
        edge = run_edge(args.data_root, frames)
        edge.to_csv(out / f"{tag}_edge_drift.csv", index=False)
        per = edge[edge.frame != "WINDOW"].groupby(["ring_sort", "true_yaw_deg"]).abs_err_deg
        print("\nEdge drift, per-frame: % frame có |sai số| <= 0.25°")
        print((per.apply(lambda x: (x <= 0.25).mean()) * 100).round(0).to_string())
        print("\nEdge drift, window:")
        print(edge[edge.frame == "WINDOW"][["ring_sort", "window", "true_yaw_deg", "est_drift_deg"]].to_string(index=False))


if __name__ == "__main__":
    main()
