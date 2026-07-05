import argparse
import os

import numpy as np
import scipy.io as sio
from skimage import io

from stainpms.candidate import compute_b_candidates_oncrop


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find MoNuSeg crops that contain baseline false-negative GT nuclei."
    )
    parser.add_argument("--name", required=True, help="Image name without extension.")
    parser.add_argument("--baseline", required=True, help="Baseline prediction .npy.")
    parser.add_argument("--data_root", default="./data/monuseg")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--iou_thr", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--min_area", type=int, default=20)
    return parser.parse_args()


def crop_box(cx, cy, width, height, size):
    size = min(size, width, height)
    x0 = int(round(cx - size / 2))
    y0 = int(round(cy - size / 2))
    x0 = max(0, min(x0, width - size))
    y0 = max(0, min(y0, height - size))
    return x0, y0, size


def max_iou_with_pred(gt_mask, pred):
    pred_ids = np.unique(pred[gt_mask])
    pred_ids = pred_ids[pred_ids > 0]
    best = 0.0
    best_id = 0
    gt_area = float(gt_mask.sum())
    for pid in pred_ids:
        pred_mask = pred == pid
        inter = float((gt_mask & pred_mask).sum())
        union = gt_area + float(pred_mask.sum()) - inter
        iou = inter / max(union, 1.0)
        if iou > best:
            best = iou
            best_id = int(pid)
    return best, best_id


def main():
    args = parse_args()
    image_path = os.path.join(args.data_root, "test", "images", args.name + ".tif")
    gt_path = os.path.join(args.data_root, "test", "labels", args.name + ".mat")

    image = io.imread(image_path)[..., :3]
    gt = sio.loadmat(gt_path)["inst_map"].astype(np.int32)
    baseline = np.load(args.baseline).astype(np.int32)
    if gt.shape != baseline.shape:
        raise ValueError(f"Shape mismatch: gt={gt.shape}, baseline={baseline.shape}")

    coords, _, inst_ids, _ = compute_b_candidates_oncrop(
        image,
        gt,
        baseline_inst_map=baseline,
        baseline_dilate_radius=5,
        top_k=20,
        min_distance=12,
        open_disk=2,
        sigma=1.0,
        gt_match_radius=8,
        keep_negative=True,
        return_gt_inst_ids=True,
        return_evidence=True,
    )
    prompted_gt_ids = set(int(x) for x in inst_ids if int(x) > 0)

    rows = []
    height, width = gt.shape
    for gid in np.unique(gt):
        gid = int(gid)
        if gid == 0:
            continue
        gt_mask = gt == gid
        area = int(gt_mask.sum())
        if area < args.min_area:
            continue
        best_iou, best_pid = max_iou_with_pred(gt_mask, baseline)
        if best_iou >= args.iou_thr:
            continue
        ys, xs = np.where(gt_mask)
        cx = float(xs.mean())
        cy = float(ys.mean())
        x0, y0, size = crop_box(cx, cy, width, height, args.crop_size)
        crop_gt = gt[y0 : y0 + size, x0 : x0 + size]
        crop_base = baseline[y0 : y0 + size, x0 : x0 + size]
        fn_ids_in_crop = []
        for crop_gid in np.unique(crop_gt):
            crop_gid = int(crop_gid)
            if crop_gid == 0:
                continue
            crop_mask = gt == crop_gid
            crop_iou, _ = max_iou_with_pred(crop_mask, baseline)
            if crop_iou < args.iou_thr:
                fn_ids_in_crop.append(crop_gid)
        pos_prompt_in_crop = 0
        for (px, py), prompt_gid in zip(coords, inst_ids):
            if prompt_gid > 0 and x0 <= px < x0 + size and y0 <= py < y0 + size:
                pos_prompt_in_crop += 1
        residual_hint = int(((crop_gt > 0) & (crop_base == 0)).sum())
        rows.append(
            {
                "gid": gid,
                "x0": x0,
                "y0": y0,
                "size": size,
                "area": area,
                "best_iou": best_iou,
                "best_pred_id": best_pid,
                "fn_count_crop": len(fn_ids_in_crop),
                "pos_prompt_crop": pos_prompt_in_crop,
                "prompted": gid in prompted_gt_ids,
                "residual_hint": residual_hint,
            }
        )

    rows.sort(
        key=lambda r: (
            int(r["prompted"]),
            r["pos_prompt_crop"],
            r["fn_count_crop"],
            r["residual_hint"],
            r["area"],
        ),
        reverse=True,
    )

    if not rows:
        print("No baseline FN found under the current threshold.")
        return

    print("Top FN-centered crops:")
    for idx, r in enumerate(rows[: args.top], start=1):
        print(
            f"{idx:02d} crop=--crop {r['x0']} {r['y0']} {r['size']} "
            f"gid={r['gid']} best_iou={r['best_iou']:.3f} "
            f"fn_in_crop={r['fn_count_crop']} pos_prompts={r['pos_prompt_crop']} "
            f"prompted={r['prompted']}"
        )
    best = rows[0]
    print("\nRun:")
    print(
        "PYTHONPATH=. python tools/fig1b_stain_mining_vis.py "
        f"--name {args.name} --baseline {args.baseline} "
        f"--crop {best['x0']} {best['y0']} {best['size']}"
    )


if __name__ == "__main__":
    main()
