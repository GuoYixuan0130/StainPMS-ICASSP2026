import argparse
import csv
import os

import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_fill_holes
from skimage import io, segmentation
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, disk

from stainpms.candidate import compute_b_candidates_oncrop, compute_h_evidence


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Fig. 1(b) visualizations for StainPMS candidate mining."
    )
    parser.add_argument("--name", required=True, help="Image name without extension.")
    parser.add_argument("--data_root", default="./data/monuseg")
    parser.add_argument("--image", default="", help="Optional explicit H&E image path.")
    parser.add_argument("--gt", default="", help="Optional explicit GT .mat path.")
    parser.add_argument(
        "--baseline",
        default="",
        help="Optional explicit cached baseline instance .npy path. If empty, common dirs are searched.",
    )
    parser.add_argument("--out_dir", default="./fig1_outputs/fig1b_vis")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument(
        "--crop",
        type=int,
        nargs=3,
        default=None,
        metavar=("X", "Y", "SIZE"),
        help="Optional top-left crop: x y size. Overrides auto crop.",
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_distance", type=int, default=12)
    parser.add_argument("--baseline_dilate_radius", type=int, default=5)
    parser.add_argument("--open_disk", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--gt_match_radius", type=int, default=8)
    return parser.parse_args()


def first_existing(paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError("None of these paths exists:\n" + "\n".join(paths))


def load_inst_map(mat_path):
    mat = sio.loadmat(mat_path)
    if "inst_map" not in mat:
        raise KeyError(f"{mat_path} does not contain key 'inst_map'")
    return mat["inst_map"].astype(np.int32)


def crop_box_from_center(cx, cy, width, height, size):
    size = min(size, width, height)
    x0 = int(round(cx - size / 2))
    y0 = int(round(cy - size / 2))
    x0 = max(0, min(x0, width - size))
    y0 = max(0, min(y0, height - size))
    return x0, y0, x0 + size, y0 + size


def auto_crop_from_candidates(image, gt, baseline, args):
    coords, _, inst_ids, _ = compute_b_candidates_oncrop(
        image,
        gt,
        baseline_inst_map=baseline,
        baseline_dilate_radius=args.baseline_dilate_radius,
        top_k=args.top_k,
        min_distance=args.min_distance,
        open_disk=args.open_disk,
        sigma=args.sigma,
        gt_match_radius=args.gt_match_radius,
        keep_negative=True,
        return_gt_inst_ids=True,
        return_evidence=True,
    )
    h, w = gt.shape
    if len(coords) == 0:
        return crop_box_from_center(w / 2, h / 2, w, h, args.crop_size)

    pos = np.where(inst_ids > 0)[0]
    pick = int(pos[0]) if len(pos) else 0
    cx, cy = coords[pick]
    return crop_box_from_center(cx, cy, w, h, args.crop_size)


def normalize01(arr):
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


def grayscale_rgb(arr):
    arr = normalize01(arr)
    gray = (arr * 255).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def overlay_mask(image, mask, color, alpha=0.45):
    out = image.copy().astype(np.float32)
    color = np.asarray(color, dtype=np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_boundary(image, label_map, color=(255, 220, 0), width=1):
    out = image.copy()
    bd = segmentation.find_boundaries(label_map, mode="outer")
    if width > 1:
        bd = binary_dilation(bd, iterations=width - 1)
    out[bd] = np.asarray(color, dtype=np.uint8)
    return out


def draw_prompts(image, coords_xy, inst_ids, radius=4):
    out = image.copy()
    draw = ImageDraw.Draw(Image.fromarray(out))
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    for (x, y), inst_id in zip(coords_xy, inst_ids):
        x = int(round(float(x)))
        y = int(round(float(y)))
        if inst_id > 0:
            fill = (0, 230, 80)
            outline = (0, 0, 0)
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=fill,
                outline=outline,
                width=2,
            )
        else:
            color = (255, 40, 40)
            draw.line([x - radius, y - radius, x + radius, y + radius], fill=color, width=3)
            draw.line([x - radius, y + radius, x + radius, y - radius], fill=color, width=3)
    return np.asarray(pil)


def binary_rgb(mask, color=(255, 255, 255)):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask] = np.asarray(color, dtype=np.uint8)
    return out


def draw_candidate_peaks(mask, coords_xy, radius=4):
    pil = Image.fromarray(binary_rgb(mask))
    draw = ImageDraw.Draw(pil)
    for x, y in coords_xy:
        x = int(round(float(x)))
        y = int(round(float(y)))
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(0, 220, 255),
            outline=(0, 0, 0),
            width=1,
        )
    return np.asarray(pil)


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

    image_path = args.image or os.path.join(args.data_root, "test", "images", f"{name}.tif")
    gt_path = args.gt or os.path.join(args.data_root, "test", "labels", f"{name}.mat")
    baseline_path = args.baseline or first_existing(
        [
            os.path.join("./fig1_outputs/monuseg_baseline_pred", f"{name}.npy"),
            os.path.join("./preds_test_monuseg_baseline", f"{name}.npy"),
            os.path.join("./baseline_masks_test_monuseg", f"{name}.npy"),
            os.path.join("./baseline_masks_test_monuseg_nms2", f"{name}.npy"),
        ]
    )

    image = io.imread(image_path)[..., :3].astype(np.uint8)
    gt = load_inst_map(gt_path)
    baseline = np.load(baseline_path).astype(np.int32)

    if image.shape[:2] != gt.shape or gt.shape != baseline.shape:
        raise ValueError(
            f"Shape mismatch: image={image.shape[:2]}, gt={gt.shape}, baseline={baseline.shape}"
        )

    if args.crop is not None:
        x0, y0, size = args.crop
        x1, y1 = x0 + size, y0 + size
    else:
        x0, y0, x1, y1 = auto_crop_from_candidates(image, gt, baseline, args)

    image_c = image[y0:y1, x0:x1]
    gt_c = gt[y0:y1, x0:x1]
    baseline_c = baseline[y0:y1, x0:x1]

    evidence = compute_h_evidence(image_c, sigma=args.sigma)
    thr = threshold_otsu(evidence)
    stain_fg = evidence >= thr
    stain_fg = binary_fill_holes(stain_fg)
    stain_fg = binary_opening(stain_fg, footprint=disk(args.open_disk))
    baseline_dil = binary_dilation(
        baseline_c > 0, structure=disk(args.baseline_dilate_radius)
    )
    residual = stain_fg & (~baseline_dil)

    coords, _, inst_ids, _ = compute_b_candidates_oncrop(
        image_c,
        gt_c,
        baseline_inst_map=baseline_c,
        baseline_dilate_radius=args.baseline_dilate_radius,
        top_k=args.top_k,
        min_distance=args.min_distance,
        open_disk=args.open_disk,
        sigma=args.sigma,
        gt_match_radius=args.gt_match_radius,
        keep_negative=True,
        return_gt_inst_ids=True,
        return_evidence=True,
    )

    out_dir = os.path.join(args.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    he = image_c
    h_gray = grayscale_rgb(evidence)
    stain_fg_panel = binary_rgb(stain_fg)
    baseline_panel = binary_rgb(baseline_dil)
    residual_panel = binary_rgb(residual)
    candidate_panel = draw_candidate_peaks(residual, coords, radius=4)
    prompt_panel = overlay_mask(he, residual, (0, 220, 80), alpha=0.18)
    prompt_panel = draw_boundary(prompt_panel, gt_c, color=(255, 255, 255), width=1)
    prompt_panel = draw_prompts(prompt_panel, coords, inst_ids, radius=4)

    save_png(os.path.join(out_dir, "01_he_crop.png"), he)
    save_png(os.path.join(out_dir, "02_hematoxylin_evidence_gray.png"), h_gray)
    save_png(os.path.join(out_dir, "03_stain_foreground_MH.png"), stain_fg_panel)
    save_png(os.path.join(out_dir, "04_baseline_coverage_B.png"), baseline_panel)
    save_png(os.path.join(out_dir, "05_residual_foreground_MH_minus_B.png"), residual_panel)
    save_png(os.path.join(out_dir, "06_candidate_peaks.png"), candidate_panel)
    save_png(os.path.join(out_dir, "07_pos_neg_stain_prompts.png"), prompt_panel)

    panels = [
        add_title(he, "H&E crop"),
        add_title(h_gray, "H evidence"),
        add_title(stain_fg_panel, "Stain foreground M_H"),
        add_title(baseline_panel, "Baseline coverage B"),
        add_title(residual_panel, "Residual M_H \\ B"),
        add_title(candidate_panel, "Candidate peaks"),
        add_title(prompt_panel, "Pos./Neg. stain prompts"),
    ]
    strip = np.concatenate(panels, axis=1)
    save_png(os.path.join(out_dir, "fig1b_strip.png"), strip)

    csv_path = os.path.join(out_dir, "prompt_points.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_crop", "y_crop", "x_full", "y_full", "label", "gt_inst_id"])
        for (x, y), inst_id in zip(coords, inst_ids):
            writer.writerow(
                [
                    float(x),
                    float(y),
                    float(x + x0),
                    float(y + y0),
                    "pos" if inst_id > 0 else "neg",
                    int(inst_id),
                ]
            )

    print(f"image: {image_path}")
    print(f"gt: {gt_path}")
    print(f"baseline M_b: {baseline_path}")
    print(f"crop: x={x0}, y={y0}, w={x1 - x0}, h={y1 - y0}")
    print(f"prompts: total={len(coords)}, pos={int((inst_ids > 0).sum())}, neg={int((inst_ids == 0).sum())}")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
