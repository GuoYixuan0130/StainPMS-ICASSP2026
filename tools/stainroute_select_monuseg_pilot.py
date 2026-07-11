"""Precommit MoNuSeg futility-pilot images from ADD candidate counts only."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stainroute.utils import canonical_json_sha256, sha256_file


def select_pilot_images(rows: list[dict[str, str]], seed: int = 3407) -> dict:
    """Select two precommitted images from each ADD-count quartile.

    Only ``image`` and ``add_candidates`` are read.  A seeded shuffle within
    each sorted-count quartile makes the choice reproducible without seeing
    baseline PQ, GT errors, decoded masks, or action utility.
    """

    counts: dict[str, int] = {}
    for row in rows:
        image = str(row["image"])
        if image in counts:
            raise ValueError(f"Duplicate image in candidate audit: {image}")
        counts[image] = int(row["add_candidates"])
    ordered = sorted(counts.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) < 8:
        raise ValueError("Need at least eight images for two four-image pilot batches")
    groups = []
    first_batch = []
    second_batch = []
    for quartile in range(4):
        start = quartile * len(ordered) // 4
        end = (quartile + 1) * len(ordered) // 4
        group = list(ordered[start:end])
        if len(group) < 2:
            raise ValueError(f"Quartile {quartile} has fewer than two images")
        shuffled = list(group)
        random.Random(seed + quartile).shuffle(shuffled)
        first_batch.append(shuffled[0][0])
        second_batch.append(shuffled[1][0])
        groups.append(
            {
                "quartile": quartile + 1,
                "count_range": [group[0][1], group[-1][1]],
                "images": [{"image": image, "add_candidates": count} for image, count in group],
                "selected_batch_1": shuffled[0][0],
                "selected_batch_2": shuffled[1][0],
            }
        )
    body = {
        "schema_version": 1,
        "selection_method": "seeded_permutation_within_sorted_add_count_quartiles",
        "seed": int(seed),
        "selection_inputs": ["image", "add_candidates"],
        "pilot_batch_1": first_batch,
        "pilot_batch_2": second_batch,
        "quartiles": groups,
    }
    return {**body, "content_sha256": canonical_json_sha256(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-audit", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", default=3407, type=int)
    args = parser.parse_args()
    with args.candidate_audit.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image", "add_candidates"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Candidate audit must contain only-needed columns {sorted(required)}")
    payload = select_pilot_images(rows, seed=args.seed)
    payload["candidate_audit_path"] = str(args.candidate_audit)
    payload["candidate_audit_sha256"] = sha256_file(args.candidate_audit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out} (content_sha256={payload['content_sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
