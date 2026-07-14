"""Patient-balanced LOPO epoch selection and one final quality-head fit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .model import ModelBundle, assert_frozen_without_grads
from .protocol import QUALITY_EPOCH_LIMIT, QUALITY_LR, QUALITY_WEIGHT_DECAY, SEED, finite_arrays, json_dump, quality_focal_loss, set_determinism


def _load_rows(cache_dir: Path, target_dir: Path, *, patients: set[int] | None = None) -> dict[int, dict[str, torch.Tensor]]:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    grouped: dict[int, dict[str, list[np.ndarray]]] = {}
    for record in manifest["records"]:
        patient = int(record["patient"])
        if patients is not None and patient not in patients:
            continue
        with np.load(cache_dir / record["file"], allow_pickle=False) as cache, np.load(target_dir / f"{record['image_id']}.npz", allow_pickle=False) as target:
            group = grouped.setdefault(patient, {"features": [], "target": [], "matched": []})
            group["features"].append(np.asarray(cache["quality_feature"], dtype=np.float16))
            group["target"].append(np.asarray(target["utility_target"], dtype=np.float32))
            group["matched"].append(np.asarray(target["matched"], dtype=np.bool_))
    if not grouped:
        raise ValueError("quality training received no authorized cache rows")
    return {
        patient: {
            "features": torch.from_numpy(np.concatenate(values["features"])),
            "target": torch.from_numpy(np.concatenate(values["target"])),
            "matched": torch.from_numpy(np.concatenate(values["matched"])),
        }
        for patient, values in grouped.items()
    }


def _new_head(device: torch.device) -> torch.nn.Module:
    from sam2_train.modeling.dpa_p2pnet import QualityHead

    head = QualityHead(256).to(device)
    return head


def _evaluate(head: torch.nn.Module, groups: dict[int, dict[str, torch.Tensor]], device: torch.device) -> float:
    head.eval()
    losses = []
    with torch.no_grad():
        for values in groups.values():
            logits = head(values["features"].to(device=device, dtype=torch.float32)).squeeze(-1)
            loss = quality_focal_loss(logits, values["target"].to(device), values["matched"].to(device))
            losses.append(float(loss.cpu()))
    return float(np.mean(losses))


def _balanced_epoch(head: torch.nn.Module, optimizer: torch.optim.Optimizer, groups: dict[int, dict[str, torch.Tensor]], device: torch.device, generator: torch.Generator, *, batch_per_patient: int = 512) -> float:
    """Equal per-patient batches; no large patient can dominate a gradient."""
    head.train()
    steps = max(int(np.ceil(len(values["target"]) / batch_per_patient)) for values in groups.values())
    losses = []
    for _ in range(steps):
        feature_batches, target_batches, matched_batches = [], [], []
        for patient in sorted(groups):
            values = groups[patient]
            index = torch.randint(len(values["target"]), (batch_per_patient,), generator=generator)
            feature_batches.append(values["features"][index])
            target_batches.append(values["target"][index])
            matched_batches.append(values["matched"][index])
        optimizer.zero_grad(set_to_none=True)
        logits = head(torch.cat(feature_batches).to(device=device, dtype=torch.float32)).squeeze(-1)
        target = torch.cat(target_batches).to(device)
        matched = torch.cat(matched_batches).to(device)
        loss = quality_focal_loss(logits, target, matched)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite PromptQ-v2 quality loss")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in head.parameters()):
            raise FloatingPointError("non-finite PromptQ-v2 quality gradient")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _fit(head: torch.nn.Module, train: dict[int, dict[str, torch.Tensor]], validation: dict[int, dict[str, torch.Tensor]] | None, device: torch.device, epochs: int) -> list[dict]:
    optimizer = torch.optim.AdamW(head.parameters(), lr=QUALITY_LR, weight_decay=QUALITY_WEIGHT_DECAY)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    curve = []
    for epoch in range(1, epochs + 1):
        train_loss = _balanced_epoch(head, optimizer, train, device, generator)
        row = {"epoch": epoch, "train_loss": train_loss, "train_quality_loss": _evaluate(head, train, device)}
        if validation is not None:
            row["heldout_quality_loss"] = _evaluate(head, validation, device)
        curve.append(row)
    return curve


def choose_lopo_epoch(cache_dir: Path, target_dir: Path, out_dir: Path, device: torch.device) -> dict:
    """One fixed LOPO sweep over the six authorized patients, no dev access."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite LOPO artifacts: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    all_groups = _load_rows(cache_dir, target_dir, patients=set(range(1, 7)))
    if set(all_groups) != set(range(1, 7)):
        raise RuntimeError("LOPO requires each authorized patient 1--6")
    rows = []
    for heldout in range(1, 7):
        set_determinism()
        head = _new_head(device)
        curve = _fit(head, {key: value for key, value in all_groups.items() if key != heldout}, {heldout: all_groups[heldout]}, device, QUALITY_EPOCH_LIMIT)
        rows.extend({"heldout_patient": heldout, **row} for row in curve)
    mean_by_epoch = []
    for epoch in range(1, QUALITY_EPOCH_LIMIT + 1):
        values = [row["heldout_quality_loss"] for row in rows if row["epoch"] == epoch]
        mean_by_epoch.append({"epoch": epoch, "mean_lopo_heldout_quality_loss": float(np.mean(values))})
    chosen = min(mean_by_epoch, key=lambda row: (row["mean_lopo_heldout_quality_loss"], row["epoch"]))
    with (out_dir / "lopo_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)
    json_dump(out_dir / "lopo_epoch_selection.json", {"selection_rule": "minimum mean held-out quality loss; earliest tie", "max_epochs": QUALITY_EPOCH_LIMIT, "chosen_epoch": chosen["epoch"], "curve": mean_by_epoch, "development_access": "none"})
    return {"chosen_epoch": int(chosen["epoch"]), "curve": mean_by_epoch}


def train_final_quality_head(bundle: ModelBundle, cache_dir: Path, target_dir: Path, epochs: int, out_dir: Path) -> dict:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite final quality training: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    groups = _load_rows(cache_dir, target_dir, patients=set(range(1, 7)))
    if set(groups) != set(range(1, 7)):
        raise RuntimeError("final train requires authorized patients 1--6")
    set_determinism()
    head = bundle.point_net.quality_head
    curve = _fit(head, groups, None, bundle.device, epochs)
    assert_frozen_without_grads(bundle)
    state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    torch.save(state, out_dir / "quality_head.pt")
    report = {"selected_epoch": epochs, "optimizer": "AdamW", "lr": QUALITY_LR, "weight_decay": QUALITY_WEIGHT_DECAY, "patient_balanced_sampling": True, "seed": SEED, "curve": curve, "final_train_quality_loss": _evaluate(head, groups, bundle.device)}
    json_dump(out_dir / "report.json", report)
    return report


@torch.no_grad()
def cached_quality_logits(bundle: ModelBundle, cache_dir: Path) -> dict[str, np.ndarray]:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    bundle.point_net.quality_head.eval()
    values = {}
    for record in manifest["records"]:
        with np.load(cache_dir / record["file"], allow_pickle=False) as cache:
            features = torch.as_tensor(np.asarray(cache["quality_feature"], dtype=np.float16), device=bundle.device, dtype=torch.float32)
            logits = bundle.point_net.quality_head(features).squeeze(-1).detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(logits).all():
            raise FloatingPointError("non-finite cached quality logit")
        values[record["image_id"]] = logits
    return values
