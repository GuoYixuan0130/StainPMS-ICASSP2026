"""Create the frozen three-seed TNBC C0/C1 warm-start report.

The tool is read-only with respect to model artifacts. It aggregates only
already-written p7/p8 development diagnoses from the fixed epoch-5 states.
All three pre-specified seeds are mandatory inputs; no seed can be dropped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stainpms.phase2a_pqbest import choose_pq_best
from stainpms.phase2a_tnbc_screen import MECHANISM_METRICS, TASK_METRICS, metric_deltas


SEEDS = (3407, 2027, 1337)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validate_records(records: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(records, list) or len(records) != 5 or any(not isinstance(value, dict) for value in records):
        raise ValueError(f"{label} must contain exactly five diagnosis records")
    epochs = [int(value.get("epoch", -1)) for value in records]
    if epochs != [1, 2, 3, 4, 5]:
        raise ValueError(f"{label} epochs must be contiguous 1--5, got {epochs}")
    return records


def legacy_seed_records(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = read_json(path)
    return (
        _validate_records(payload.get("c0"), label="seed-3407 C0"),
        _validate_records(payload.get("c1"), label="seed-3407 C1"),
    )


def low_storage_records(path: Path, *, expected_seed: int, expected_arm: str, expected_protocol: str) -> list[dict[str, Any]]:
    payload = read_json(path)
    if payload.get("protocol") != expected_protocol:
        raise ValueError(f"{path} protocol mismatch: {payload.get('protocol')!r}")
    if int(payload.get("determinism", {}).get("seed", -1)) != expected_seed:
        raise ValueError(f"{path} seed mismatch")
    if payload.get("training_configuration", {}).get("arm") != expected_arm:
        raise ValueError(f"{path} arm mismatch")
    return _validate_records([item.get("diagnosis") for item in payload.get("epochs", [])], label=str(path))


def _group(record: dict[str, Any], level: str) -> dict[str, Any]:
    if level == "patient_macro":
        return record["patient_macro"]
    return record["patients"][level]


def _metric_values(per_seed: list[dict[str, Any]], *, arm: str, level: str, section: str, metric: str) -> list[float]:
    return [float(_group(item[arm], level)[section][metric]) for item in per_seed]


def stats(values: list[float]) -> dict[str, Any]:
    if len(values) != len(SEEDS) or not all(math.isfinite(value) for value in values):
        raise ValueError("three finite pre-specified seed values are required")
    return {
        "values_by_seed": {str(seed): value for seed, value in zip(SEEDS, values, strict=True)},
        "mean": statistics.mean(values),
        "std_sample": statistics.stdev(values),
        "n": len(values),
    }


def aggregate(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"arms": {}, "paired_c1_minus_c0": {}}
    for arm in ("c0", "c1_full"):
        result["arms"][arm] = {}
        for level in ("7", "8", "patient_macro"):
            result["arms"][arm][level] = {
                "task_metrics_image_macro": {
                    metric: stats(_metric_values(per_seed, arm=arm, level=level, section="task_metrics_image_macro", metric=metric))
                    for metric in TASK_METRICS
                },
                "mechanism": {
                    metric: stats(_metric_values(per_seed, arm=arm, level=level, section="mechanism", metric=metric))
                    for metric in MECHANISM_METRICS
                },
            }
    for level in ("7", "8", "patient_macro"):
        result["paired_c1_minus_c0"][level] = {
            "task_metrics_image_macro": {
                metric: stats(
                    [float(_group(item["c1_minus_c0"], level)["task_metrics_image_macro"][metric]) for item in per_seed]
                )
                for metric in TASK_METRICS
            },
            "mechanism": {
                metric: stats(
                    [float(_group(item["c1_minus_c0"], level)["mechanism"][metric]) for item in per_seed]
                )
                for metric in MECHANISM_METRICS
            },
        }
    return result


def advancement(aggregate_result: dict[str, Any]) -> dict[str, Any]:
    paired = aggregate_result["paired_c1_minus_c0"]["patient_macro"]["task_metrics_image_macro"]
    aji = paired["aji"]
    pq = paired["pq"]
    checks = {
        "mean_aji_delta_positive": aji["mean"] > 0.0,
        "mean_pq_delta_positive": pq["mean"] > 0.0,
        "aji_positive_seed_count_at_least_2": sum(value > 0.0 for value in aji["values_by_seed"].values()) >= 2,
        "pq_positive_seed_count_at_least_2": sum(value > 0.0 for value in pq["values_by_seed"].values()) >= 2,
    }
    return {
        "status": "pass_freeze_c1_and_request_owner_test_decision" if all(checks.values()) else "fail_stop_current_warmstart_route",
        "checks": checks,
        "aji_positive_seed_count": sum(value > 0.0 for value in aji["values_by_seed"].values()),
        "pq_positive_seed_count": sum(value > 0.0 for value in pq["values_by_seed"].values()),
    }


def csv_rows(per_seed: list[dict[str, Any]], aggregate_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in per_seed:
        for name, source in (("c0", record["c0"]), ("c1_full", record["c1_full"]), ("c1_full_minus_c0", record["c1_minus_c0"])):
            for level in ("7", "8", "patient_macro"):
                group = _group(source, level)
                row = {"seed": record["seed"], "comparison": name, "level": level, "kind": "fixed_epoch_5"}
                row.update(group["task_metrics_image_macro"])
                row.update(group["mechanism"])
                rows.append(row)
    for name, source in (("c0", "arms"), ("c1_full", "arms"), ("c1_full_minus_c0", "paired_c1_minus_c0")):
        values = aggregate_result[source][name] if source == "arms" else aggregate_result[source]
        for level in ("7", "8", "patient_macro"):
            group = values[level]
            for stat_name in ("mean", "std_sample"):
                row = {"seed": "all_3", "comparison": name, "level": level, "kind": stat_name}
                row.update({metric: group["task_metrics_image_macro"][metric][stat_name] for metric in TASK_METRICS})
                row.update({metric: group["mechanism"][metric][stat_name] for metric in MECHANISM_METRICS})
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["seed", "comparison", "level", "kind", *TASK_METRICS, *MECHANISM_METRICS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown(per_seed: list[dict[str, Any]], aggregate_result: dict[str, Any], gate: dict[str, Any]) -> str:
    delta = aggregate_result["paired_c1_minus_c0"]["patient_macro"]
    aji = delta["task_metrics_image_macro"]["aji"]
    pq = delta["task_metrics_image_macro"]["pq"]
    lines = [
        "# TNBC three-seed C0/C1 warm-start report",
        "",
        "Primary comparison is paired C1-full minus C0 at fixed epoch 5. All pre-specified seeds are retained; PQ-best is recorded separately and cannot replace this comparison.",
        "",
        f"- Advancement decision: `{gate['status']}`",
        f"- Patient-macro paired AJI delta: `{aji['mean']:+.6f} ± {aji['std_sample']:.6f}` (sample std; {gate['aji_positive_seed_count']}/3 positive).",
        f"- Patient-macro paired PQ delta: `{pq['mean']:+.6f} ± {pq['std_sample']:.6f}` (sample std; {gate['pq_positive_seed_count']}/3 positive).",
        "",
        "| seed | C0 AJI | C1 AJI | AJI delta | C0 PQ | C1 PQ | PQ delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in per_seed:
        c0 = item["c0"]["patient_macro"]["task_metrics_image_macro"]
        c1 = item["c1_full"]["patient_macro"]["task_metrics_image_macro"]
        d = item["c1_minus_c0"]["patient_macro"]["task_metrics_image_macro"]
        lines.append(f"| {item['seed']} | {c0['aji']:.6f} | {c1['aji']:.6f} | {d['aji']:+.6f} | {c0['pq']:.6f} | {c1['pq']:.6f} | {d['pq']:+.6f} |")
    lines.extend(
        [
            "",
            "Best-CCR, selected-CCR, and selection regret are retained in JSON/CSV as mechanism analysis only. They are not advancement gates and cannot establish stable candidate-generation improvement.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed3407-c0-c1-metrics", required=True)
    parser.add_argument("--seed2027-c0-summary", required=True)
    parser.add_argument("--seed2027-c1-summary", required=True)
    parser.add_argument("--seed1337-c0-summary", required=True)
    parser.add_argument("--seed1337-c1-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")

    seed3407_c0, seed3407_c1 = legacy_seed_records(Path(args.seed3407_c0_c1_metrics))
    seed2027_c0 = low_storage_records(Path(args.seed2027_c0_summary), expected_seed=2027, expected_arm="c0", expected_protocol="tnbc_c0_c1_second_seed_2027_v1")
    seed2027_c1 = low_storage_records(Path(args.seed2027_c1_summary), expected_seed=2027, expected_arm="c1", expected_protocol="tnbc_c0_c1_second_seed_2027_v1")
    seed1337_c0 = low_storage_records(Path(args.seed1337_c0_summary), expected_seed=1337, expected_arm="c0", expected_protocol="tnbc_c0_c1_third_seed_1337_v1")
    seed1337_c1 = low_storage_records(Path(args.seed1337_c1_summary), expected_seed=1337, expected_arm="c1", expected_protocol="tnbc_c0_c1_third_seed_1337_v1")

    per_seed: list[dict[str, Any]] = []
    for seed, c0, c1 in ((3407, seed3407_c0, seed3407_c1), (2027, seed2027_c0, seed2027_c1), (1337, seed1337_c0, seed1337_c1)):
        per_seed.append(
            {
                "seed": seed,
                "c0": c0[-1],
                "c1_full": c1[-1],
                "c1_minus_c0": metric_deltas(c0[-1], c1[-1]),
                "pq_best_selection": {"c0": choose_pq_best(c0), "c1_full": choose_pq_best(c1)},
            }
        )
    aggregate_result = aggregate(per_seed)
    gate = advancement(aggregate_result)
    payload = {
        "schema_version": 1,
        "protocol": "tnbc_c0_c1_three_seed_warmstart_v1",
        "seeds": list(SEEDS),
        "primary_comparison": "fixed_epoch_5_paired_C1_full_minus_C0",
        "per_seed": per_seed,
        "aggregate": aggregate_result,
        "advancement_rule": gate,
        "interpretation_boundary": {
            "all_seeds_retained": True,
            "p7_p8_role": "development-only; no sealed TNBC access",
            "candidate_metrics": "mechanism analysis only; not an advancement gate",
            "if_gate_fails": "stop current warm-start route without p1-p8/p9-p11 formal run",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "three_seed_summary.json", payload)
    write_csv(output_dir / "three_seed_summary.csv", csv_rows(per_seed, aggregate_result))
    (output_dir / "three_seed_summary.md").write_text(markdown(per_seed, aggregate_result, gate), encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "decision": gate["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
