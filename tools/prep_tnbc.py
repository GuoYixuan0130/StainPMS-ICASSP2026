"""TNBC dataset prep for the CA-SAM2 pipeline.

Two subcommands:

  inspect : walk a downloaded TNBC tree and report, per directory, the
            shape / dtype / #unique-values of sampled files. This SETTLES the
            one make-or-break question before we write any converter: are the
            ground-truth masks BINARY foreground (0/255 -> touching nuclei get
            merged by connected components, which would corrupt instance GT AND
            under-count the missed-detection FN headroom PMS depends on) or are
            they already INSTANCE-labelled (0,1,2,...,N)?

            Run this FIRST and paste the output. It makes zero assumptions about
            the on-disk layout, so it works regardless of how BNS.zip extracted.

  convert : (written only after `inspect` confirms the layout) emit the
            MonuSeg-style layout the loader in run/dataset/monuseg.py expects:
              <dst>/train_XX/images/<name>.png
              <dst>/train_XX/labels/<name>.mat   # scipy mat, key 'inst_map'
              <dst>/test/images , <dst>/test/labels
            with a PATIENT-LEVEL train/test split (no slide leakage).

This script is read-only in `inspect` mode and only writes under <dst> in
`convert` mode. It never touches the source tree.
"""

import argparse
import os
import re
from collections import Counter

import numpy as np
import scipy.io as sio
from scipy import ndimage as ndi
from skimage import io
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries, relabel_sequential, watershed

IMG_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


def _load(path):
    """Return (array, error). Never raises."""
    try:
        return io.imread(path), None
    except Exception as e:  # noqa: BLE001 - report, don't crash the walk
        return None, repr(e)


def _describe(arr):
    """One-line description of an image/mask array."""
    shape = arr.shape
    dtype = arr.dtype
    if arr.ndim == 3 and shape[-1] in (3, 4):
        kind = f"RGB{'A' if shape[-1] == 4 else ''}"
        return f"{kind} {shape} {dtype}"
    # 2D (or single-channel) -> candidate mask
    flat = arr.reshape(-1) if arr.ndim == 2 else arr[..., 0].reshape(-1)
    uniq = np.unique(flat)
    n_uniq = uniq.size
    if n_uniq <= 2:
        verdict = "BINARY"
    elif n_uniq <= 8:
        verdict = f"{n_uniq}-LEVEL (few)"
    else:
        verdict = "INSTANCE-like"
    sample_vals = uniq[:6].tolist()
    return (f"2D {shape} {dtype} | #unique={n_uniq} "
            f"min={uniq.min()} max={uniq.max()} | first_vals={sample_vals} "
            f"-> {verdict}")


def cmd_inspect(args):
    root = os.path.abspath(args.src)
    if not os.path.isdir(root):
        raise SystemExit(f"[inspect] src is not a directory: {root}")

    print(f"[inspect] walking {root}\n")
    mask_unique_counts = []  # aggregate over all 2D files
    total_files = 0

    for dirpath, _dirnames, filenames in sorted(os.walk(root)):
        rel = os.path.relpath(dirpath, root)
        imgs = [f for f in sorted(filenames)
                if os.path.splitext(f)[1].lower() in IMG_EXTS]
        if not imgs:
            continue
        ext_counts = Counter(os.path.splitext(f)[1].lower() for f in imgs)
        total_files += len(imgs)
        print(f"DIR  {rel}/   ({len(imgs)} image-like files, exts={dict(ext_counts)})")

        for fname in imgs[: args.max_samples]:
            arr, err = _load(os.path.join(dirpath, fname))
            if err is not None:
                print(f"     {fname}: <unreadable> {err}")
                continue
            desc = _describe(arr)
            print(f"     {fname}: {desc}")
            # collect unique counts for 2D mask-candidates
            if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] not in (3, 4)):
                flat = arr.reshape(-1) if arr.ndim == 2 else arr[..., 0].reshape(-1)
                mask_unique_counts.append(int(np.unique(flat).size))
        if len(imgs) > args.max_samples:
            print(f"     ... ({len(imgs) - args.max_samples} more not sampled)")
        print()

    print("=" * 60)
    print(f"[inspect] total image-like files: {total_files}")
    if mask_unique_counts:
        arr = np.array(mask_unique_counts)
        n_binary = int((arr <= 2).sum())
        print(f"[inspect] 2D mask-candidate files sampled: {arr.size}")
        print(f"          #unique values  min={arr.min()} median={int(np.median(arr))} max={arr.max()}")
        print(f"          files with <=2 unique values (BINARY): {n_binary}/{arr.size}")
        if n_binary == arr.size:
            print("  VERDICT: masks are BINARY foreground -> instances must be derived"
                  " (connected components / watershed). Touching nuclei WILL merge.")
        elif n_binary == 0:
            print("  VERDICT: masks are INSTANCE-labelled -> use directly as inst_map.")
        else:
            print("  VERDICT: MIXED -> inspect individual dirs above; some binary, some instance.")
    else:
        print("[inspect] no 2D mask-candidate files found among sampled files.")
    print("=" * 60)


def _to_rgb(arr):
    """Coerce an imread result to HxWx3 uint8 RGB (drop alpha / expand gray)."""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]
    else:
        raise ValueError(f"unexpected image shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def binary_to_instances(binary, min_distance, sigma):
    """Split a binary foreground mask into instances via distance-transform watershed.

    Touching nuclei share one connected component; the EDT peaks act as one
    marker per nucleus so watershed cuts them apart. Falls back to connected
    components when no peak is found. Returns a contiguous int32 label map.
    """
    binary = np.asarray(binary).astype(bool)
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.int32)
    distance = ndi.distance_transform_edt(binary)
    if sigma and sigma > 0:
        distance = ndi.gaussian_filter(distance, sigma)
    coords = peak_local_max(distance, min_distance=min_distance, labels=binary)
    if coords.shape[0] == 0:
        labels, _ = ndi.label(binary)
    else:
        markers = np.zeros(distance.shape, dtype=np.int32)
        markers[tuple(coords.T)] = np.arange(1, coords.shape[0] + 1)
        labels = watershed(-distance, markers, mask=binary)
    labels, _, _ = relabel_sequential(labels.astype(np.int32))
    return labels.astype(np.int32)


def _viz_panel(img_rgb, labels):
    """original | instance-colored | boundary overlay, concatenated horizontally."""
    lab_rgb = (label2rgb(labels, bg_label=0) * 255).astype(np.uint8)
    overlay = img_rgb.copy()
    overlay[find_boundaries(labels, mode="outer")] = (255, 0, 0)
    return np.concatenate([img_rgb, lab_rgb, overlay], axis=1)


def cmd_convert(args):
    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    test_patients = {int(x) for x in args.test_patients.split(",") if x.strip()}
    viz_dir = args.viz_dir or os.path.join(dst, "_viz_watershed")

    slide_dirs = sorted(
        d for d in os.listdir(src)
        if d.lower().startswith("slide_") and os.path.isdir(os.path.join(src, d))
    )
    if not slide_dirs:
        raise SystemExit(f"[convert] no Slide_* dirs under {src}")

    counts = {"train_12": 0, "test": 0}
    inst_per_img = []
    n_viz = 0
    print(f"[convert] src={src}\n[convert] dst={dst}\n[convert] test patients={sorted(test_patients)} "
          f"| min_distance={args.min_distance} sigma={args.sigma}\n")

    for slide in slide_dirs:
        m = re.search(r"(\d+)", slide)
        pid = int(m.group(1))
        split = "test" if pid in test_patients else "train_12"
        gt_dir = os.path.join(src, f"GT_{pid:02d}")
        if not os.path.isdir(gt_dir):
            print(f"[convert] WARN missing GT dir for {slide} -> {gt_dir}, skipped")
            continue
        img_out = os.path.join(dst, split, "images")
        lbl_out = os.path.join(dst, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for fname in sorted(os.listdir(os.path.join(src, slide))):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in IMG_EXTS:
                continue
            gt_path = os.path.join(gt_dir, fname)
            if not os.path.exists(gt_path):
                print(f"[convert] WARN no GT for {slide}/{fname}, skipped")
                continue
            img = _to_rgb(io.imread(os.path.join(src, slide, fname)))
            gt = io.imread(gt_path)
            if gt.ndim == 3:
                gt = gt[..., 0]
            inst = binary_to_instances(gt > 0, args.min_distance, args.sigma)
            n_inst = int(inst.max())
            inst_per_img.append(n_inst)

            io.imsave(os.path.join(img_out, f"{stem}.png"), img, check_contrast=False)
            sio.savemat(os.path.join(lbl_out, f"{stem}.mat"), {"inst_map": inst})
            counts[split] += 1

            if n_viz < args.viz_n:
                os.makedirs(viz_dir, exist_ok=True)
                io.imsave(os.path.join(viz_dir, f"{stem}_{split}_{n_inst}inst.png"),
                          _viz_panel(img, inst), check_contrast=False)
                n_viz += 1

    arr = np.array(inst_per_img) if inst_per_img else np.array([0])
    print("=" * 60)
    print(f"[convert] train_12 images: {counts['train_12']} | test images: {counts['test']}")
    print(f"[convert] instances/image  min={arr.min()} median={int(np.median(arr))} "
          f"max={arr.max()} | total={int(arr.sum())}")
    print(f"[convert] (TNBC reference: ~4022 nuclei / 50 imgs = ~80/img; compare median)")
    print(f"[convert] wrote {n_viz} watershed-split previews to {viz_dir}")
    print(f"[convert] train with:  --dataset monuseg --data_path {os.path.relpath(dst)}")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="report TNBC tree structure + mask value stats")
    pi.add_argument("--src", required=True, help="root of the extracted TNBC dataset")
    pi.add_argument("--max-samples", type=int, default=4,
                    help="files to load per directory (default 4)")
    pi.set_defaults(func=cmd_inspect)

    pc = sub.add_parser("convert", help="binary GT -> watershed instances -> MonuSeg layout")
    pc.add_argument("--src", required=True, help="root with Slide_* and GT_* dirs")
    pc.add_argument("--dst", required=True, help="output root, e.g. data/tnbc")
    pc.add_argument("--test-patients", default="9,10,11",
                    help="comma-separated patient indices for the test split (default 9,10,11)")
    pc.add_argument("--min-distance", type=int, default=10,
                    help="peak_local_max min separation in px; lower=more splits (default 10)")
    pc.add_argument("--sigma", type=float, default=1.0,
                    help="gaussian smoothing on the distance map before peak detection (default 1.0)")
    pc.add_argument("--viz-dir", default="", help="where to dump split previews (default <dst>/_viz_watershed)")
    pc.add_argument("--viz-n", type=int, default=12, help="number of preview panels to write (default 12)")
    pc.set_defaults(func=cmd_convert)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
