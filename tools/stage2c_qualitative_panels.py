"""Export Stage 2C qualitative crops around selected refinement actions."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation
from skimage import io, segmentation


IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def _read_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        lines = [line for line in f if line.strip()]
        return list(csv.DictReader(lines))


def _to_float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    if not np.isfinite(value):
        return default
    return float(value)


def _to_int(row: dict, key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def _find_image(image_root: Path, name: str) -> Path:
    for ext in IMAGE_EXTS:
        path = image_root / f"{name}{ext}"
        if path.exists() and path.is_file():
            return path
    matches = [path for path in image_root.glob(f"{name}.*") if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No image found for {name} under {image_root}")
    raise RuntimeError(f"Ambiguous image files for {name}: {matches}")


def _load_gt(label_root: Path, name: str) -> np.ndarray:
    path = label_root / f"{name}.mat"
    mat = sio.loadmat(path)
    if "inst_map" in mat:
        return mat["inst_map"].astype(np.int32)
    if "inst" in mat:
        return mat["inst"].astype(np.int32)
    raise KeyError(f"{path} does not contain inst_map or inst")


def _crop_bounds(x: int, y: int, h: int, w: int, size: int) -> tuple[int, int, int, int]:
    crop_w = min(size, w)
    crop_h = min(size, h)
    x0 = int(round(x - crop_w / 2))
    y0 = int(round(y - crop_h / 2))
    x0 = max(0, min(w - crop_w, x0))
    y0 = max(0, min(h - crop_h, y0))
    return x0, y0, x0 + crop_w, y0 + crop_h


def _crop(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return arr[y0:y1, x0:x1]


def _boundary_overlay(image: np.ndarray, inst: np.ndarray, color: tuple[int, int, int], width: int = 1) -> np.ndarray:
    out = image.copy()
    boundary = segmentation.find_boundaries(inst, mode="outer")
    if width > 1:
        boundary = binary_dilation(boundary, iterations=width - 1)
    out[boundary] = np.asarray(color, dtype=np.uint8)
    return out


def _mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.55,
) -> np.ndarray:
    out = image.copy()
    if mask.any():
        color_arr = np.asarray(color, dtype=np.float32)
        out[mask] = ((1.0 - alpha) * out[mask].astype(np.float32) + alpha * color_arr).astype(np.uint8)
    return out


def _draw_point(image: np.ndarray, x: int, y: int, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    out = Image.fromarray(image)
    draw = ImageDraw.Draw(out)
    r = 5
    draw.line((x - r, y, x + r, y), fill=color, width=2)
    draw.line((x, y - r, x, y + r), fill=color, width=2)
    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    return np.asarray(out)


def _title(image: np.ndarray, text: str, height: int = 30) -> np.ndarray:
    font = ImageFont.load_default()
    h, w = image.shape[:2]
    canvas = Image.new("RGB", (w, h + height), (255, 255, 255))
    canvas.paste(Image.fromarray(image), (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), text, fill=(0, 0, 0), font=font)
    return np.asarray(canvas)


def _save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _select_images(metrics: list[dict], names: list[str], top_k: int) -> list[str]:
    if names:
        return names
    ordered = sorted(metrics, key=lambda row: _to_float(row, "delta_pq"), reverse=True)
    return [str(row["image"]) for row in ordered[:top_k]]


def _metric_lookup(metrics: list[dict]) -> dict[str, dict]:
    return {str(row["image"]): row for row in metrics}


def _actions_by_image(actions: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in actions:
        grouped.setdefault(str(row["image"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: _to_int(row, "local_rank"))
    return grouped


def export_panels(args) -> None:
    split = args.split
    image_root = args.data_root / split / "images"
    label_root = args.data_root / split / "labels"
    metrics = _read_csv(args.image_metrics_csv)
    actions = _read_csv(args.selected_actions_csv)
    metrics_by_name = _metric_lookup(metrics)
    actions_by_name = _actions_by_image(actions)
    names = _select_images(metrics, args.names, args.top_k)

    summary_rows = []
    for name in names:
        image = io.imread(_find_image(image_root, name))[..., :3].astype(np.uint8)
        gt = _load_gt(label_root, name)
        base = np.load(args.artifacts_dir / f"{name}_pred.npy").astype(np.int32)
        refined = np.load(args.refined_dir / f"{name}_pred.npy").astype(np.int32)
        inserted = (refined > 0) & (base == 0)
        h, w = gt.shape
        metric = metrics_by_name.get(name, {})
        rows = actions_by_name.get(name, [])[: args.max_actions_per_image]
        if not rows:
            rows = [{"x": w // 2, "y": h // 2, "local_rank": 0, "action_rank": -1, "score": 0, "added_area": 0}]

        for action in rows:
            x = _to_int(action, "x", w // 2)
            y = _to_int(action, "y", h // 2)
            box = _crop_bounds(x, y, h, w, args.crop_size)
            x0, y0, _, _ = box
            local_x = int(x - x0)
            local_y = int(y - y0)

            image_c = _crop(image, box)
            gt_c = _crop(gt, box)
            base_c = _crop(base, box)
            refined_c = _crop(refined, box)
            inserted_c = _crop(inserted, box)

            raw = _draw_point(image_c, local_x, local_y)
            gt_panel = _boundary_overlay(image_c, gt_c, color=(255, 255, 255), width=1)
            base_panel = _boundary_overlay(image_c, base_c, color=(255, 220, 0), width=1)
            refined_panel = _boundary_overlay(image_c, refined_c, color=(0, 220, 80), width=1)
            delta_panel = _mask_overlay(image_c, inserted_c, color=(0, 210, 255), alpha=0.55)
            delta_panel = _boundary_overlay(delta_panel, refined_c, color=(0, 220, 80), width=1)
            delta_panel = _draw_point(delta_panel, local_x, local_y)

            strip = np.concatenate(
                [
                    _title(raw, "H&E + prompt"),
                    _title(gt_panel, "GT"),
                    _title(base_panel, "StainPMS"),
                    _title(refined_panel, "StainPQR"),
                    _title(delta_panel, "Inserted"),
                ],
                axis=1,
            )

            local_rank = _to_int(action, "local_rank")
            action_rank = _to_int(action, "action_rank")
            out_name = f"{name}_a{local_rank}_r{action_rank}_x{x}_y{y}.png"
            _save(args.out_dir / out_name, strip)
            summary_rows.append(
                {
                    "image": name,
                    "local_rank": local_rank,
                    "action_rank": action_rank,
                    "x": x,
                    "y": y,
                    "score": _to_float(action, "score"),
                    "added_area": _to_int(action, "added_area"),
                    "delta_pq": _to_float(metric, "delta_pq"),
                    "delta_aji": _to_float(metric, "delta_aji"),
                    "delta_dq": _to_float(metric, "delta_dq"),
                    "delta_sq": _to_float(metric, "delta_sq"),
                    "panel": out_name,
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "panel_index.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else ["image", "panel"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote panels: {args.out_dir}")
    print(f"Wrote index: {args.out_dir / 'panel_index.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=["test", "train_12"])
    parser.add_argument("--artifacts_dir", required=True, type=Path)
    parser.add_argument("--refined_dir", required=True, type=Path)
    parser.add_argument("--image_metrics_csv", required=True, type=Path)
    parser.add_argument("--selected_actions_csv", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--names", nargs="*", default=[])
    parser.add_argument("--top_k", default=4, type=int)
    parser.add_argument("--crop_size", default=192, type=int)
    parser.add_argument("--max_actions_per_image", default=1, type=int)
    return parser.parse_args()


def main() -> None:
    export_panels(parse_args())


if __name__ == "__main__":
    main()
