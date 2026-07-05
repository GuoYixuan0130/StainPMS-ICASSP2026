import argparse
import os

import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation
from skimage import io, segmentation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate qualitative comparison crop for Fig. 2."
    )
    parser.add_argument("--name", required=True, help="Image name without extension.")
    parser.add_argument(
        "--crop",
        nargs=3,
        type=int,
        required=True,
        metavar=("X", "Y", "SIZE"),
        help="Top-left crop: x y size.",
    )
    parser.add_argument("--data_root", default="./data/monuseg")
    parser.add_argument("--baseline", required=True, help="CA-SAM2 baseline .npy.")
    parser.add_argument("--pms", required=True, help="StainPMS prediction .npy.")
    parser.add_argument("--out_dir", default="./fig2_outputs")
    parser.add_argument("--iou_thr", type=float, default=0.5)
    return parser.parse_args()


def load_gt(path):
    mat = sio.loadmat(path)
    if "inst_map" not in mat:
        raise KeyError(f"{path} does not contain key 'inst_map'")
    return mat["inst_map"].astype(np.int32)


def crop(arr, x0, y0, size):
    return arr[y0 : y0 + size, x0 : x0 + size]


def colorize_instance_map(inst, seed=123):
    out = np.zeros((*inst.shape, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    for inst_id in np.unique(inst):
        if inst_id == 0:
            continue
        color = rng.integers(60, 255, size=3, dtype=np.uint8)
        out[inst == inst_id] = color
    return out


def boundary_overlay(image, inst, color=(255, 220, 0), width=1):
    out = image.copy()
    boundary = segmentation.find_boundaries(inst, mode="outer")
    if width > 1:
        boundary = binary_dilation(boundary, iterations=width - 1)
    out[boundary] = np.asarray(color, dtype=np.uint8)
    return out


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


def compute_recovery_masks(gt, baseline, pms, iou_thr):
    recovered = np.zeros(gt.shape, dtype=bool)
    still_fn = np.zeros(gt.shape, dtype=bool)
    recovered_rows = []
    for gt_id in np.unique(gt):
        gt_id = int(gt_id)
        if gt_id == 0:
            continue
        gt_mask = gt == gt_id
        base_iou = best_iou(gt_mask, baseline)
        pms_iou = best_iou(gt_mask, pms)
        if base_iou < iou_thr <= pms_iou:
            recovered |= gt_mask
            recovered_rows.append((gt_id, int(gt_mask.sum()), base_iou, pms_iou))
        elif base_iou < iou_thr and pms_iou < iou_thr:
            still_fn |= gt_mask
    return recovered, still_fn, recovered_rows


def recovered_overlay(image, gt, recovered, still_fn):
    overlay = image.copy()
    if recovered.any():
        overlay[recovered] = (
            0.45 * overlay[recovered] + 0.55 * np.array([0, 230, 80])
        ).astype(np.uint8)
    if still_fn.any():
        overlay[still_fn] = (
            0.45 * overlay[still_fn] + 0.55 * np.array([255, 40, 40])
        ).astype(np.uint8)
    overlay = boundary_overlay(overlay, gt, color=(255, 255, 255), width=1)
    return overlay


def add_title(image, title, height=28):
    font = ImageFont.load_default()
    h, w = image.shape[:2]
    canvas = Image.new("RGB", (w, h + height), (255, 255, 255))
    canvas.paste(Image.fromarray(image), (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), title, fill=(0, 0, 0), font=font)
    return np.asarray(canvas)


def save_png(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)


def main():
    args = parse_args()
    name = args.name
    x0, y0, size = args.crop

    images_dir = os.path.join(args.data_root, "test", "images")
    image_path = None
    for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"):
        cand = os.path.join(images_dir, f"{name}{ext}")
        if os.path.exists(cand):
            image_path = cand
            break
    if image_path is None:
        raise FileNotFoundError(
            f"no image for '{name}' in {images_dir} "
            "(tried .tif/.tiff/.png/.jpg/.jpeg/.bmp)"
        )
    gt_path = os.path.join(args.data_root, "test", "labels", f"{name}.mat")
    image = io.imread(image_path)[..., :3].astype(np.uint8)
    gt = load_gt(gt_path)
    baseline = np.load(args.baseline).astype(np.int32)
    pms = np.load(args.pms).astype(np.int32)

    if image.shape[:2] != gt.shape or gt.shape != baseline.shape or gt.shape != pms.shape:
        raise ValueError(
            f"Shape mismatch: image={image.shape[:2]}, gt={gt.shape}, "
            f"baseline={baseline.shape}, pms={pms.shape}"
        )

    image_c = crop(image, x0, y0, size)
    gt_c = crop(gt, x0, y0, size)
    baseline_c = crop(baseline, x0, y0, size)
    pms_c = crop(pms, x0, y0, size)

    recovered, still_fn, recovered_rows = compute_recovery_masks(
        gt_c, baseline_c, pms_c, args.iou_thr
    )
    overlay = recovered_overlay(image_c, gt_c, recovered, still_fn)

    out_dir = os.path.join(args.out_dir, f"{name}_x{x0}_y{y0}")
    os.makedirs(out_dir, exist_ok=True)

    he = image_c
    gt_black = colorize_instance_map(gt_c, seed=1)
    baseline_black = colorize_instance_map(baseline_c, seed=2)
    pms_black = colorize_instance_map(pms_c, seed=3)
    baseline_boundary = boundary_overlay(image_c, baseline_c, color=(255, 220, 0), width=1)
    pms_boundary = boundary_overlay(image_c, pms_c, color=(0, 230, 80), width=1)

    save_png(os.path.join(out_dir, "01_he_crop.png"), he)
    save_png(os.path.join(out_dir, "02_gt_instance_black.png"), gt_black)
    save_png(os.path.join(out_dir, "03_baseline_instance_black.png"), baseline_black)
    save_png(os.path.join(out_dir, "04_stainpms_instance_black.png"), pms_black)
    save_png(os.path.join(out_dir, "03_baseline_boundary_overlay.png"), baseline_boundary)
    save_png(os.path.join(out_dir, "04_stainpms_boundary_overlay.png"), pms_boundary)
    save_png(os.path.join(out_dir, "05_recovered_overlay.png"), overlay)

    strip = np.concatenate(
        [
            add_title(he, "H&E crop"),
            add_title(gt_black, "GT"),
            add_title(baseline_black, "CA-SAM2"),
            add_title(pms_black, "StainPMS"),
            add_title(overlay, "Corrected nuclei"),
        ],
        axis=1,
    )
    save_png(os.path.join(out_dir, "fig2_strip.png"), strip)

    print(f"saved: {out_dir}")
    print(
        f"recovered instances: {len(recovered_rows)}, "
        f"recovered_pixels: {int(recovered.sum())}, "
        f"still_fn_pixels: {int(still_fn.sum())}"
    )
    for gt_id, area, base_iou, pms_iou in recovered_rows:
        print(
            f"  gt_id={gt_id} area={area} "
            f"baseline_iou={base_iou:.3f} stainpms_iou={pms_iou:.3f}"
        )


if __name__ == "__main__":
    main()
