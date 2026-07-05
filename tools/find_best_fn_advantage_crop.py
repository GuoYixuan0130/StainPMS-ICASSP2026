import argparse
import glob
import os

import numpy as np
import scipy.io as sio


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find MoNuSeg test crops where baseline misses GT nuclei and "
            "StainPMS recovers them."
        )
    )
    parser.add_argument("--data_root", default="./data/monuseg")
    parser.add_argument("--baseline_dir", default="./preds_test_monuseg_baseline")
    parser.add_argument("--pms_dir", default="./preds_test_monuseg_pms")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--iou_thr", type=float, default=0.5)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--min_area", type=int, default=20)
    return parser.parse_args()


def best_iou(gt_mask, pred):
    pred_ids = np.unique(pred[gt_mask])
    pred_ids = pred_ids[pred_ids > 0]
    best = 0.0
    gt_area = int(gt_mask.sum())
    for pred_id in pred_ids:
        pred_mask = pred == pred_id
        inter = int((gt_mask & pred_mask).sum())
        union = gt_area + int(pred_mask.sum()) - inter
        best = max(best, inter / max(union, 1))
    return best


def crop_box(cx, cy, width, height, size):
    size = min(size, width, height)
    x0 = int(max(0, min(cx - size / 2, width - size)))
    y0 = int(max(0, min(cy - size / 2, height - size)))
    return x0, y0, size


def collect_instances(gt, baseline_pred, pms_pred, args):
    instances = []
    for gt_id in np.unique(gt):
        gt_id = int(gt_id)
        if gt_id == 0:
            continue
        gt_mask = gt == gt_id
        area = int(gt_mask.sum())
        if area < args.min_area:
            continue
        ys, xs = np.where(gt_mask)
        base_iou = best_iou(gt_mask, baseline_pred)
        pms_iou = best_iou(gt_mask, pms_pred)
        base_fn = base_iou < args.iou_thr
        recovered = base_fn and pms_iou >= args.iou_thr
        instances.append(
            {
                "gt_id": gt_id,
                "cx": float(xs.mean()),
                "cy": float(ys.mean()),
                "area": area,
                "base_iou": base_iou,
                "pms_iou": pms_iou,
                "base_fn": base_fn,
                "recovered": recovered,
            }
        )
    return instances


def main():
    args = parse_args()
    rows = []

    gt_paths = sorted(glob.glob(os.path.join(args.data_root, "test", "labels", "*.mat")))
    for gt_path in gt_paths:
        name = os.path.splitext(os.path.basename(gt_path))[0]
        baseline_path = os.path.join(args.baseline_dir, f"{name}.npy")
        pms_path = os.path.join(args.pms_dir, f"{name}.npy")
        if not (os.path.exists(baseline_path) and os.path.exists(pms_path)):
            continue

        gt = sio.loadmat(gt_path)["inst_map"].astype(np.int32)
        baseline_pred = np.load(baseline_path).astype(np.int32)
        pms_pred = np.load(pms_path).astype(np.int32)
        if gt.shape != baseline_pred.shape or gt.shape != pms_pred.shape:
            print(
                f"[skip] shape mismatch for {name}: "
                f"gt={gt.shape}, baseline={baseline_pred.shape}, pms={pms_pred.shape}"
            )
            continue

        height, width = gt.shape
        instances = collect_instances(gt, baseline_pred, pms_pred, args)
        base_fns = [item for item in instances if item["base_fn"]]
        for fn_item in base_fns:
            x0, y0, size = crop_box(
                fn_item["cx"],
                fn_item["cy"],
                width,
                height,
                args.crop_size,
            )
            inside = [
                item
                for item in instances
                if x0 <= item["cx"] < x0 + size and y0 <= item["cy"] < y0 + size
            ]
            baseline_fn_count = sum(item["base_fn"] for item in inside)
            recovered_count = sum(item["recovered"] for item in inside)
            still_fn_count = sum(
                item["base_fn"] and not item["recovered"] for item in inside
            )
            recovered_area = sum(item["area"] for item in inside if item["recovered"])
            baseline_fn_area = sum(item["area"] for item in inside if item["base_fn"])
            mean_gain = 0.0
            recovered_items = [item for item in inside if item["recovered"]]
            if recovered_items:
                mean_gain = float(
                    np.mean(
                        [
                            item["pms_iou"] - item["base_iou"]
                            for item in recovered_items
                        ]
                    )
                )

            rows.append(
                {
                    "score": (
                        recovered_count,
                        baseline_fn_count,
                        recovered_area,
                        baseline_fn_area,
                        mean_gain,
                        -still_fn_count,
                    ),
                    "name": name,
                    "x0": x0,
                    "y0": y0,
                    "size": size,
                    "baseline_fn_count": baseline_fn_count,
                    "recovered_count": recovered_count,
                    "still_fn_count": still_fn_count,
                    "baseline_fn_area": baseline_fn_area,
                    "recovered_area": recovered_area,
                    "mean_gain": mean_gain,
                }
            )

    rows.sort(key=lambda item: item["score"], reverse=True)

    if not rows:
        print("No candidate crop found. Check prediction dirs and IoU threshold.")
        return

    print("Top crops where StainPMS recovers baseline FNs:")
    for idx, row in enumerate(rows[: args.top], start=1):
        print(
            f"{idx:02d} name={row['name']} "
            f"--crop {row['x0']} {row['y0']} {row['size']} | "
            f"baseline_FN={row['baseline_fn_count']} | "
            f"recovered_by_StainPMS={row['recovered_count']} | "
            f"still_FN={row['still_fn_count']} | "
            f"FN_area={row['baseline_fn_area']} | "
            f"recovered_area={row['recovered_area']} | "
            f"mean_IoU_gain={row['mean_gain']:.3f}"
        )

    best = rows[0]
    baseline_path = os.path.join(args.baseline_dir, f"{best['name']}.npy")
    print("\nBest command for Fig. 1(b):")
    print(
        "PYTHONPATH=. python tools/fig1b_stain_mining_vis.py "
        f"--name {best['name']} "
        f"--baseline {baseline_path} "
        f"--crop {best['x0']} {best['y0']} {best['size']}"
    )


if __name__ == "__main__":
    main()
