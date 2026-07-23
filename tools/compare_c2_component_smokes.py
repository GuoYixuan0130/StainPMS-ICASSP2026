#!/usr/bin/env python3
"""Gate the two fixed C2 component arms before formal five-epoch runs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from compare_c2_ar_smokes import regression_check, scale_check


def read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_arm(payload: dict[str, Any], arm: str, *, exclusivity: float, utility: float) -> dict[str, Any]:
    configuration = payload.get("training_configuration", {})
    losses = payload.get("losses", {})
    weighted = {
        "exclusivity": losses.get("loss_c2_ar_exclusivity"),
        "utility": losses.get("loss_c2_ar_utility"),
    }
    finite = all(isinstance(value, (float, int)) and math.isfinite(float(value)) for value in weighted.values())
    inactive = (
        abs(float(weighted["exclusivity"])) <= 1.0e-8 if exclusivity == 0.0 else True
    ) and (abs(float(weighted["utility"])) <= 1.0e-8 if utility == 0.0 else True)
    active = (
        float(weighted["exclusivity"]) > 0.0 if exclusivity > 0.0 else True
    ) and (float(weighted["utility"]) > 0.0 if utility > 0.0 else True)
    valid = payload.get("status") == "complete" and configuration.get("arm") == arm and finite and inactive and active
    return {"status": "pass" if valid else "fail", "arm": configuration.get("arm"), "weighted_losses": weighted, "finite": finite, "inactive_term_zero": inactive, "active_term_nonzero": active}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c1", required=True, type=Path)
    parser.add_argument("--c2-zero", required=True, type=Path)
    parser.add_argument("--c2-e", required=True, type=Path)
    parser.add_argument("--c2-u", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    c1, zero, excl, util = (read(path.resolve()) for path in (args.c1, args.c2_zero, args.c2_e, args.c2_u))
    regression = regression_check(c1, zero, absolute_tolerance=1.0e-6, relative_tolerance=1.0e-5)
    e_check = verify_arm(excl, "c2_e", exclusivity=0.25, utility=0.0)
    u_check = verify_arm(util, "c2_u", exclusivity=0.0, utility=0.25)
    scales = {"c2_e": scale_check(excl), "c2_u": scale_check(util)}
    payload = {
        "schema_version": 1,
        "protocol": "tnbc_c2_component_ablation_v1",
        "status": "pass" if regression["status"] == "pass" and e_check["status"] == "pass" and u_check["status"] == "pass" and all(row["status"] == "pass" for row in scales.values()) else "fail",
        "c1_regression_with_zero_c2": regression,
        "component_arm_checks": {"c2_e": e_check, "c2_u": u_check},
        "loss_scale_checks": scales,
    }
    write(args.output.resolve(), payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
