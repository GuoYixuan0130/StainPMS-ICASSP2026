"""Create the fixed-selection ResiMix Stage-1 dual-dev report from run artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resimixpms.experiment import write_json  # noqa: E402


METRICS = ("dice1", "dice2", "aji", "aji_p", "dq", "sq", "pq")
SCHEDULES = {"tnbc": (0, 2, 4, 6, 8, 10), "monuseg_lite": (0, 5, 10)}
ITEM_COUNTS = {"tnbc": 7, "monuseg_lite": 12}
_TOLERANCE = 1e-12


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _evaluations(run_dir: Path, expected_epochs: tuple[int, ...]) -> list[dict[str, float | int]]:
    rows = []
    for epoch in expected_epochs:
        path = run_dir / f"evaluation_epoch_{epoch:02d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing registered evaluation node: {path}")
        row = _read_json(path)
        if int(row.get("completed_epochs", -1)) != epoch or not all(name in row for name in METRICS):
            raise ValueError(f"malformed evaluation row: {path}")
        rows.append({"completed_epochs": epoch, **{name: float(row[name]) for name in METRICS}})
    unexpected = sorted(run_dir.glob("evaluation_epoch_*.json"))
    if len(unexpected) != len(expected_epochs):
        raise ValueError(f"unexpected/missing evaluation nodes in {run_dir}")
    return rows


def _fixed_best(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    """Pre-registered selection: maximum PQ, then AJI, then latest epoch."""
    return max(rows, key=lambda row: (float(row["pq"]), float(row["aji"]), int(row["completed_epochs"])))


def _per_item(run_dir: Path, epoch: int, expected_count: int) -> list[dict[str, str]]:
    path = run_dir / f"per_image_epoch_{epoch:02d}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing per-item metrics: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = [row.get("image", "") for row in rows]
    if len(rows) != expected_count or len(set(names)) != expected_count or any(not name for name in names):
        raise ValueError(f"{path} must contain exactly {expected_count} unique items")
    for row in rows:
        for field in (*METRICS, "tp", "fp", "fn"):
            if field not in row:
                raise ValueError(f"per-item row lacks {field}: {path}")
    return rows


def _numeric(row: Mapping[str, Any], field: str) -> float:
    return float(row[field])


def _sum_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    return {field: sum(int(float(row[field])) for row in rows) for field in ("tp", "fp", "fn")}


def _aggregate(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate zero evaluation items")
    return {metric: sum(_numeric(row, metric) for row in rows) / len(rows) for metric in METRICS}


def _training_augmentation(run_dir: Path, dataset: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize every formal epoch, including the 0--1 no-ResiMix warm-up."""
    path = run_dir / "synthetic_acceptance.csv"
    source_rows: list[dict[str, str]] = []
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
    output = []
    for epoch in range(10):
        rows = [row for row in source_rows if int(row.get("epoch", -1) or -1) == epoch]
        proposals = [row for row in rows if row.get("status") in {"accepted", "rejected"}]
        accepted = [row for row in proposals if row.get("status") == "accepted"]
        row: dict[str, Any] = {
            "dataset": dataset,
            "epoch": epoch,
            "event_count": len(rows),
            "not_selected": sum(item.get("status") == "not_selected" for item in rows),
            "proposal_count": len(proposals),
            "accepted": len(accepted),
            "rejected": sum(item.get("status") == "rejected" for item in proposals),
            "proposal_acceptance_rate": len(accepted) / len(proposals) if proposals else 0.0,
            "synthetic_prompt_entry_rate": (
                sum(str(item.get("synthetic_prompt_added", "")).lower() == "true" for item in accepted) / len(accepted)
                if accepted else 0.0
            ),
        }
        for category in ("Missed", "IoU-Cliff", "Low-Quality Matched"):
            row[f"donor_{category}"] = sum(item.get("donor_category") == category for item in proposals)
        for mode in ("adjacent", "isolated"):
            row[f"host_{mode}"] = sum(item.get("host_mode") == mode for item in accepted)
        output.append(row)
    totals = {
        "accepted": sum(row["accepted"] for row in output),
        "proposal_count": sum(row["proposal_count"] for row in output),
        "donor_distribution": {category: sum(row[f"donor_{category}"] for row in output) for category in ("Missed", "IoU-Cliff", "Low-Quality Matched")},
        "host_distribution": {mode: sum(row[f"host_{mode}"] for row in output) for mode in ("adjacent", "isolated")},
    }
    totals["proposal_acceptance_rate"] = totals["accepted"] / totals["proposal_count"] if totals["proposal_count"] else 0.0
    return output, totals


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _per_item_compare(control: list[Mapping[str, str]], resimix: list[Mapping[str, str]]) -> list[dict[str, Any]]:
    by_name = {row["image"]: row for row in control}
    counterpart = {row["image"]: row for row in resimix}
    if set(by_name) != set(counterpart):
        raise ValueError("matched runs have different per-image/per-patch identifiers")
    rows = []
    for name in sorted(by_name):
        left, right = by_name[name], counterpart[name]
        row: dict[str, Any] = {"image": name}
        for metric in (*METRICS, "tp", "fp", "fn"):
            row[f"control_{metric}"] = left[metric]
            row[f"resimix_{metric}"] = right[metric]
            row[f"delta_{metric}"] = _numeric(right, metric) - _numeric(left, metric)
        rows.append(row)
    return rows


def _step0_equivalence(control: list[Mapping[str, str]], resimix: list[Mapping[str, str]]) -> dict[str, Any]:
    comparison = _per_item_compare(control, resimix)
    mismatches = []
    for row in comparison:
        for field in (*METRICS, "tp", "fp", "fn"):
            if abs(float(row[f"delta_{field}"])) > _TOLERANCE:
                mismatches.append({"image": row["image"], "field": field, "delta": row[f"delta_{field}"]})
    if mismatches:
        raise ValueError("matched Static-PMS and ResiMix step-0 metrics differ")
    return {"per_item_metric_exact": True, "checked_items": len(comparison)}


def _delta(left: Mapping[str, Any], right: Mapping[str, Any], metric: str) -> float:
    return float(right[metric]) - float(left[metric])


def _dataset_summary(root: Path, dataset: str, *, control_dir: Path | None = None) -> dict[str, Any]:
    control_dir = control_dir or root / dataset / "static_control"
    resimix_dir = root / dataset / "resimix"
    expected_epochs, expected_items = SCHEDULES[dataset], ITEM_COUNTS[dataset]
    control_nodes = _evaluations(control_dir, expected_epochs)
    resimix_nodes = _evaluations(resimix_dir, expected_epochs)
    control_best, resimix_best = _fixed_best(control_nodes), _fixed_best(resimix_nodes)
    control_step0 = next(row for row in control_nodes if row["completed_epochs"] == 0)
    resimix_step0 = next(row for row in resimix_nodes if row["completed_epochs"] == 0)
    control_items = _per_item(control_dir, int(control_best["completed_epochs"]), expected_items)
    resimix_items = _per_item(resimix_dir, int(resimix_best["completed_epochs"]), expected_items)
    control_step0_items = _per_item(control_dir, 0, expected_items)
    resimix_step0_items = _per_item(resimix_dir, 0, expected_items)
    control_metrics, resimix_metrics = _aggregate(control_items), _aggregate(resimix_items)
    control_step0_metrics, resimix_step0_metrics = _aggregate(control_step0_items), _aggregate(resimix_step0_items)
    # run_on_epoch now defines its overall metric as this all-item mean; keep a
    # tight reconciliation assertion so a future change cannot silently alter
    # the report's aggregation domain.
    for label, node, aggregate in (
        ("control", control_best, control_metrics), ("resimix", resimix_best, resimix_metrics),
        ("control step0", control_step0, control_step0_metrics), ("resimix step0", resimix_step0, resimix_step0_metrics),
    ):
        for metric in METRICS:
            if abs(float(node[metric]) - aggregate[metric]) > 1e-9:
                raise ValueError(f"{dataset} {label} aggregate mismatch for {metric}")
    return {
        "control_nodes": control_nodes,
        "resimix_nodes": resimix_nodes,
        "control_best": control_best,
        "resimix_best": resimix_best,
        "control_step0": control_step0,
        "resimix_step0": resimix_step0,
        "control_metrics": control_metrics,
        "resimix_metrics": resimix_metrics,
        "control_step0_metrics": control_step0_metrics,
        "resimix_step0_metrics": resimix_step0_metrics,
        "control_counts": _sum_counts(control_items),
        "resimix_counts": _sum_counts(resimix_items),
        "comparison": _per_item_compare(control_items, resimix_items),
        "step0_equivalence": _step0_equivalence(control_step0_items, resimix_step0_items),
        "training_augmentation": _training_augmentation(resimix_dir, dataset)[1],
    }


def _gate(tnbc: Mapping[str, Any], mono: Mapping[str, Any]) -> dict[str, Any]:
    t_control, t_resimix = tnbc["control_metrics"], tnbc["resimix_metrics"]
    m_control, m_resimix = mono["control_metrics"], mono["resimix_metrics"]
    t_aji, t_pq = _delta(t_control, t_resimix, "aji"), _delta(t_control, t_resimix, "pq")
    m_aji, m_pq = _delta(m_control, m_resimix, "aji"), _delta(m_control, m_resimix, "pq")
    mono_non_down = sum(row["delta_aji"] >= 0.0 and row["delta_pq"] >= 0.0 for row in mono["comparison"])
    t_strong = (t_aji >= 0.020 and t_pq >= 0.0) or (t_pq >= 0.010 and t_aji >= 0.0) or (t_aji >= 0.010 and t_pq >= 0.010)
    m_strong = ((m_aji >= 0.010 and m_pq >= 0.0) or (m_pq >= 0.010 and m_aji >= 0.0)) and mono_non_down >= 8
    t_promising = (t_aji >= 0.005 and t_pq >= 0.0) or (t_pq >= 0.005 and t_aji >= 0.0)
    m_promising = ((m_aji >= 0.005 and m_pq >= 0.0) or (m_pq >= 0.005 and m_aji >= 0.0)) and mono_non_down >= 7
    dq_or_fn = (
        _delta(t_control, t_resimix, "dq") > 0.0 or tnbc["resimix_counts"]["fn"] < tnbc["control_counts"]["fn"]
        or _delta(m_control, m_resimix, "dq") > 0.0 or mono["resimix_counts"]["fn"] < mono["control_counts"]["fn"]
    )
    aligned_small = t_aji > 0.0 and t_pq > 0.0 and m_aji > 0.0 and m_pq > 0.0 and dq_or_fn
    both_no_improvement = t_aji <= 0.0 and t_pq <= 0.0 and m_aji <= 0.0 and m_pq <= 0.0
    if t_strong or m_strong:
        verdict = "STRONG_GO"
    elif t_promising or m_promising or aligned_small:
        verdict = "PROMISING_FULL_MONUSEG_RECOMMENDED"
    elif both_no_improvement:
        verdict = "NO_GO_RECOMMENDED"
    else:
        verdict = "INCONCLUSIVE_OWNER_REVIEW"
    return {
        "verdict": verdict,
        "tnbc_delta_aji": t_aji, "tnbc_delta_pq": t_pq,
        "monuseg_lite_delta_aji": m_aji, "monuseg_lite_delta_pq": m_pq,
        "monuseg_lite_dual_non_decreasing_patches": mono_non_down,
        "strong_conditions": {"tnbc": t_strong, "monuseg_lite": m_strong},
        "promising_conditions": {"tnbc": t_promising, "monuseg_lite": m_promising, "aligned_small": aligned_small},
        "no_go_condition": both_no_improvement,
    }


def _tnbc_patient_map(root: Path) -> dict[str, int]:
    payload = _read_json(root / "tnbc_data_manifest.json")
    records = payload.get("development_records", [])
    mapping = {}
    for row in records:
        raw_name = str(row.get("image_name", row.get("image", row.get("name", ""))))
        if not raw_name:
            raise ValueError("TNBC copied manifest lacks an image name")
        mapping[Path(raw_name).stem] = int(row["patient_id"])
    return mapping


def _tnbc_per_patient(rows: list[Mapping[str, Any]], mapping: Mapping[str, int]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        try:
            patient = mapping[row["image"]]
        except KeyError as exc:
            raise ValueError(f"TNBC per-image result is absent from the frozen development manifest: {row['image']}") from exc
        grouped.setdefault(patient, []).append(row)
    result = []
    for patient in sorted(grouped):
        group = grouped[patient]
        row: dict[str, Any] = {"patient_id": patient, "image_count": len(group)}
        for metric in METRICS:
            row[f"control_{metric}"] = sum(float(item[f"control_{metric}"]) for item in group) / len(group)
            row[f"resimix_{metric}"] = sum(float(item[f"resimix_{metric}"]) for item in group) / len(group)
            row[f"delta_{metric}"] = row[f"resimix_{metric}"] - row[f"control_{metric}"]
        for field in ("tp", "fp", "fn"):
            row[f"control_{field}"] = sum(int(float(item[f"control_{field}"])) for item in group)
            row[f"resimix_{field}"] = sum(int(float(item[f"resimix_{field}"])) for item in group)
            row[f"delta_{field}"] = row[f"resimix_{field}"] - row[f"control_{field}"]
        result.append(row)
    return result


def _metric_rows(dataset: str, summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for method in ("control", "resimix"):
        selected = summary[f"{method}_best"]
        metrics = summary[f"{method}_metrics"]
        step0 = summary[f"{method}_step0_metrics"]
        control = summary["control_metrics"]
        row: dict[str, Any] = {
            "dataset": dataset, "method": method, "selection_policy": "maximum_pq_then_aji_then_latest",
            "selected_completed_epochs": int(selected["completed_epochs"]),
        }
        row.update(metrics)
        row.update({f"delta_vs_step0_{metric}": _delta(step0, metrics, metric) for metric in METRICS})
        row.update({f"delta_vs_control_{metric}": _delta(control, metrics, metric) for metric in METRICS})
        row.update(summary[f"{method}_counts"])
        rows.append(row)
    return rows


def _explicit_answers(tnbc: Mapping[str, Any], mono: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    def effects(summary: Mapping[str, Any]) -> dict[str, Any]:
        count_delta = {field: summary["resimix_counts"][field] - summary["control_counts"][field] for field in ("tp", "fp", "fn")}
        metric_delta = {metric: _delta(summary["control_metrics"], summary["resimix_metrics"], metric) for metric in ("dq", "sq", "aji", "pq")}
        return {"count_delta": count_delta, "metric_delta": metric_delta}
    t_effects, m_effects = effects(tnbc), effects(mono)
    full = gate["verdict"] in {"PROMISING_FULL_MONUSEG_RECOMMENDED", "STRONG_GO"}
    return {
        "1_tp_or_fn": {
            "tnbc": t_effects["count_delta"], "monuseg_lite": m_effects["count_delta"],
            "answer": "Positive ΔTP and/or negative ΔFN indicate that hard-nucleus transplantation improved detection.",
        },
        "2_dq_sq_or_aji": {
            "tnbc": t_effects["metric_delta"], "monuseg_lite": m_effects["metric_delta"],
            "answer": "Compare ΔDQ, ΔSQ, and ΔAJI above to attribute any gain to detection, segmentation quality, or area matching.",
        },
        "3_matched_static_control": {
            "tnbc_step0_equivalence": tnbc["step0_equivalence"],
            "monuseg_lite_step0_equivalence": mono["step0_equivalence"],
            "answer": "All scientific deltas are ResiMix-best minus separately selected matched Static-PMS Control-best.",
        },
        "4_cross_dataset_response": {
            "tnbc": {"delta_aji": gate["tnbc_delta_aji"], "delta_pq": gate["tnbc_delta_pq"]},
            "monuseg_lite": {"delta_aji": gate["monuseg_lite_delta_aji"], "delta_pq": gate["monuseg_lite_delta_pq"]},
            "answer": "Different signs or magnitudes across these two entries demonstrate a dataset-specific response.",
        },
        "5_full_monuseg": {
            "recommended": full,
            "verdict": gate["verdict"],
            "answer": "Run full MoNuSeg only when the fixed verdict is PROMISING_FULL_MONUSEG_RECOMMENDED or STRONG_GO; otherwise retain owner review/NO-GO.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument(
        "--monuseg-control-dir",
        type=Path,
        help="recovery-only override for an incomplete MoNuSeg-Lite Static-PMS arm",
    )
    options = parser.parse_args()
    root = options.artifact_dir
    tnbc = _dataset_summary(root, "tnbc")
    mono = _dataset_summary(root, "monuseg_lite", control_dir=options.monuseg_control_dir)
    gate = _gate(tnbc, mono)
    t_rows, m_rows = _metric_rows("tnbc", tnbc), _metric_rows("monuseg_lite", mono)
    _write_csv(root / "tnbc_metrics.csv", t_rows, list(t_rows[0]))
    _write_csv(root / "monuseg_lite_metrics.csv", m_rows, list(m_rows[0]))
    patient_map = _tnbc_patient_map(root)
    t_comparison = [dict(row, patient_id=patient_map[row["image"]]) for row in tnbc["comparison"]]
    _write_csv(root / "tnbc_per_image.csv", t_comparison, list(t_comparison[0]))
    _write_csv(root / "tnbc_per_patient.csv", _tnbc_per_patient(t_comparison, patient_map), list(_tnbc_per_patient(t_comparison, patient_map)[0]))
    _write_csv(root / "monuseg_lite_per_patch.csv", mono["comparison"], list(mono["comparison"][0]))
    augmentation_rows = []
    for dataset in ("tnbc", "monuseg_lite"):
        rows, _ = _training_augmentation(root / dataset / "resimix", dataset)
        augmentation_rows.extend(rows)
    _write_csv(root / "resimix_training_augmentation.csv", augmentation_rows, list(augmentation_rows[0]))
    report = {
        "selection_policy": "maximum PQ, then AJI, then latest registered epoch",
        "aggregation": "unweighted mean over every admitted Full-Dev image/patch; inclusive IoU >= 0.5",
        "gate": gate,
        "tnbc": tnbc,
        "monuseg_lite": mono,
        "answers": _explicit_answers(tnbc, mono, gate),
    }
    write_json(root / "report.json", report)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
