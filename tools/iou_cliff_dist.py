"""IoU-cliff diagnostic: GT-side per-instance IoU distribution, baseline vs StainPMS.

Purpose
-------
On TNBC the aggregate pattern is AJI up while DQ/SQ/PQ are nearly flat. That
pattern is mechanism-ambiguous; it is equally consistent with

  A. missed nuclei being partially recovered but landing just BELOW IoU 0.5
     (adds overlap -> AJI, but no new match -> DQ/PQ flat)  [supports the
     "missed-nucleus recovery" headline], vs
  C. false-positive suppression shrinking the AJI union penalty             [a
     precision effect, contradicts the recall headline].

This script disambiguates them from already-dumped instance maps (no GPU, no
retraining). It reuses the exact GT-side best-IoU used elsewhere in the repo.

Inputs (dump first with: main.py --eval ... --dump_baseline_masks_dir <dir>)
  --data_root  : dataset root; GT read from <data_root>/test/labels/<name>.mat
                 key 'inst_map' (MonuSeg layout; TNBC is converted to it).
  --baseline_dir / --pms_dir : one <name>.npy int32 instance map per test image.

Outputs
  - printed summary (paste this back),
  - per-instance CSV (--out_csv),
  - overlaid IoU histogram PNG (--out_png) if matplotlib is available.
"""

import argparse
import glob
import os

import numpy as np
import scipy.io as sio


def best_iou(gt_mask, pred):
    """Max IoU between a single GT instance mask and any overlapping pred instance."""
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


def pred_best_iou_vs_gt(pred_mask, gt):
    """Max IoU between a single PRED instance and any overlapping GT instance.

    Used to flag false positives (pred that matches no GT at >= thr)."""
    gt_ids = np.unique(gt[pred_mask])
    gt_ids = gt_ids[gt_ids > 0]
    best = 0.0
    pred_area = int(pred_mask.sum())
    for gid in gt_ids:
        gmask = gt == gid
        inter = int((pred_mask & gmask).sum())
        union = pred_area + int(gmask.sum()) - inter
        best = max(best, inter / max(union, 1))
    return best


def count_false_positives(pred, gt, thr):
    """# predicted instances whose best IoU vs any GT is < thr (false positives)."""
    fp = 0
    n_pred = 0
    for pid in np.unique(pred):
        if pid == 0:
            continue
        n_pred += 1
        if pred_best_iou_vs_gt(pred == pid, gt) < thr:
            fp += 1
    return fp, n_pred


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default="./data/tnbc")
    p.add_argument("--baseline_dir", required=True,
                   help="dir of baseline <name>.npy (dump via eval --dump_baseline_masks_dir)")
    p.add_argument("--pms_dir", required=True,
                   help="dir of StainPMS <name>.npy")
    p.add_argument("--iou_thr", type=float, default=0.5,
                   help="PQ matching threshold (default 0.5)")
    p.add_argument("--subthr_low", type=float, default=0.3,
                   help="lower edge of the 'cliff band' [low, thr) (default 0.3)")
    p.add_argument("--lift_margin", type=float, default=0.05,
                   help="min IoU gain to count a missed GT as 'lifted' (default 0.05)")
    p.add_argument("--min_area", type=int, default=10,
                   help="ignore GT instances smaller than this (px); 0 = keep all")
    p.add_argument("--out_csv", default="iou_cliff_per_instance.csv")
    p.add_argument("--out_png", default="iou_cliff_hist.png")
    return p.parse_args()


def main():
    args = parse_args()

    gt_paths = sorted(glob.glob(os.path.join(args.data_root, "test", "labels", "*.mat")))
    if not gt_paths:
        raise SystemExit(f"no GT .mat under {args.data_root}/test/labels")

    base_ious, pms_ious, areas, names = [], [], [], []
    base_fp_total = pms_fp_total = base_pred_total = pms_pred_total = 0
    n_img = 0

    for gt_path in gt_paths:
        name = os.path.splitext(os.path.basename(gt_path))[0]
        bpath = os.path.join(args.baseline_dir, f"{name}.npy")
        ppath = os.path.join(args.pms_dir, f"{name}.npy")
        if not (os.path.exists(bpath) and os.path.exists(ppath)):
            print(f"[skip] missing pred for {name}")
            continue

        gt = sio.loadmat(gt_path)["inst_map"].astype(np.int32)
        base = np.load(bpath).astype(np.int32)
        pms = np.load(ppath).astype(np.int32)
        if gt.shape != base.shape or gt.shape != pms.shape:
            print(f"[skip] shape mismatch {name}: gt={gt.shape} base={base.shape} pms={pms.shape}")
            continue
        n_img += 1

        for gid in np.unique(gt):
            if gid == 0:
                continue
            gmask = gt == gid
            area = int(gmask.sum())
            if area < args.min_area:
                continue
            base_ious.append(best_iou(gmask, base))
            pms_ious.append(best_iou(gmask, pms))
            areas.append(area)
            names.append(name)

        bfp, bpred = count_false_positives(base, gt, args.iou_thr)
        pfp, ppred = count_false_positives(pms, gt, args.iou_thr)
        base_fp_total += bfp
        pms_fp_total += pfp
        base_pred_total += bpred
        pms_pred_total += ppred

    base_ious = np.asarray(base_ious)
    pms_ious = np.asarray(pms_ious)
    areas = np.asarray(areas)
    n_gt = base_ious.size
    if n_gt == 0:
        raise SystemExit("no GT instances collected; check dirs / min_area")

    thr = args.iou_thr
    low = args.subthr_low
    margin = args.lift_margin

    base_tp = int((base_ious >= thr).sum())
    pms_tp = int((pms_ious >= thr).sum())

    missed = base_ious < thr                       # baseline FN set
    n_missed = int(missed.sum())
    crossed = int((missed & (pms_ious >= thr)).sum())  # FN -> TP (would raise DQ)
    lifted = int((missed
                  & (pms_ious < thr)
                  & (pms_ious >= low)
                  & (pms_ious > base_ious + margin)).sum())  # cliff band (A evidence)
    regressed = int(((base_ious >= thr) & (pms_ious < thr)).sum())  # TP -> FN
    mean_gain_missed = float((pms_ious[missed] - base_ious[missed]).mean()) if n_missed else 0.0

    print("=" * 64)
    print(f"[iou-cliff] images={n_img}  GT instances={n_gt}  thr={thr}")
    print(f"  mean GT-side IoU :  baseline={base_ious.mean():.4f}  StainPMS={pms_ious.mean():.4f}"
          f"  (Δ={pms_ious.mean()-base_ious.mean():+.4f})")
    print(f"  matched @>={thr}  :  baseline={base_tp}  StainPMS={pms_tp}"
          f"  (ΔTP={pms_tp-base_tp:+d}  -> DQ-count change)")
    print("-" * 64)
    print(f"  baseline-missed GT (IoU<{thr}): {n_missed}")
    print(f"    -> crossed to >= {thr} (FN->TP)            : {crossed}")
    print(f"    -> lifted into [{low},{thr}) (cliff, +>{margin}): {lifted}   <-- mechanism A")
    print(f"    -> mean IoU gain on missed set            : {mean_gain_missed:+.4f}")
    print(f"  TP->FN regressions                           : {regressed}")
    print("-" * 64)
    print(f"  prediction-side false positives (best IoU<{thr}):")
    print(f"    baseline FP={base_fp_total}/{base_pred_total}   "
          f"StainPMS FP={pms_fp_total}/{pms_pred_total}   "
          f"(ΔFP={pms_fp_total-base_fp_total:+d})   <-- mechanism C")
    print("=" * 64)
    print("Reading: large 'lifted'/positive mean-gain with ΔTP~0 => A (sub-threshold "
          "recovery, supports recall headline). Large negative ΔFP with small 'lifted' "
          "=> C (FP cleanup, a precision effect).")

    # per-instance CSV
    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write("name,area,base_iou,pms_iou,delta\n")
        for nm, ar, bi, pi in zip(names, areas, base_ious, pms_ious):
            f.write(f"{nm},{ar},{bi:.4f},{pi:.4f},{pi-bi:.4f}\n")
    print(f"[iou-cliff] wrote per-instance CSV -> {args.out_csv}")

    # overlaid histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bins = np.linspace(0, 1, 21)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(base_ious, bins=bins, alpha=0.55, label=f"CA-SAM2 (mean {base_ious.mean():.3f})")
        ax.hist(pms_ious, bins=bins, alpha=0.55, label=f"StainPMS (mean {pms_ious.mean():.3f})")
        ax.axvline(thr, color="k", ls="--", lw=1, label=f"PQ match thr={thr}")
        ax.axvspan(low, thr, color="orange", alpha=0.12)
        ax.set_xlabel("GT-side best IoU")
        ax.set_ylabel("# GT instances")
        ax.set_title("Per-instance IoU distribution (TNBC test)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=200)
        print(f"[iou-cliff] wrote histogram -> {args.out_png}")
    except Exception as e:  # noqa: BLE001
        print(f"[iou-cliff] matplotlib unavailable ({e}); CSV written, plot skipped")


if __name__ == "__main__":
    main()
