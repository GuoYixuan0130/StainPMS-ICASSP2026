"""Freeze the authorised STOP_SET_PMS owner decision beside Stage-1 artifacts.

This tool deliberately never rewrites ``report.json``.  It verifies that the
formal pre-registered verdict remains INCONCLUSIVE_OWNER_REVIEW, creates a
separate owner_decision.json once, and then rebuilds SHA256SUMS.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


FORMAL_VERDICT = "INCONCLUSIVE_OWNER_REVIEW"
OWNER_DECISION = "STOP_SET_PMS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_once(path: Path, payload: dict) -> None:
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        required = {
            "formal_artifact_verdict": FORMAL_VERDICT,
            "owner_decision": OWNER_DECISION,
        }
        if any(existing.get(key) != value for key, value in required.items()):
            raise RuntimeError(f"Existing {path.name} conflicts with the authorised decision")
        return

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_sha256s(artifact_root: Path) -> int:
    rows = []
    for path in sorted(candidate for candidate in artifact_root.rglob("*") if candidate.is_file()):
        if path.name == "SHA256SUMS":
            continue
        rows.append(f"{_sha256(path)}  {path.relative_to(artifact_root).as_posix()}")
    output = artifact_root / "SHA256SUMS"
    temporary = output.with_suffix(".tmp")
    temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    options = parser.parse_args()

    artifact_root = Path(options.artifact_root).resolve()
    report_path = artifact_root / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing formal report: {report_path}")

    report_sha_before = _sha256(report_path)
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    actual_verdict = report.get("decision", {}).get("category")
    if actual_verdict != FORMAL_VERDICT:
        raise RuntimeError(
            f"Formal report verdict is {actual_verdict!r}, expected {FORMAL_VERDICT!r}"
        )

    payload = {
        "formal_artifact_verdict": FORMAL_VERDICT,
        "owner_decision": OWNER_DECISION,
        "report_json_sha256": report_sha_before,
        "report_json_unchanged": True,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "rationale": [
            "TNBC SetPMS relative to matched Control: ΔAJI +0.00147 and ΔPQ +0.00123; the effect is too small.",
            "MoNuSeg-Lite relative to Control decreases both AJI and PQ, with only 4/12 patches non-decreasing on both metrics.",
            "Both datasets show only approximately +0.0012 SQ movement, without the expected DQ or set-level improvement.",
            "MoNuSeg-Lite AJI above step-0 is not attributable to SetPMS because matched Control performed better.",
        ],
        "prohibited_followups": [
            "Do not run full MoNuSeg.",
            "Do not access MoNuSeg official test or TNBC patients 9-11.",
            "Do not tune lambda_set, UOT parameters, soft threshold, or continuation epochs.",
            "Do not implement SetPMS-v2.",
        ],
        "artifact_policy": "Freeze current code, checkpoints, manifests, and artifacts; wait for new research authorisation.",
    }
    owner_path = artifact_root / "owner_decision.json"
    _write_json_once(owner_path, payload)

    report_sha_after = _sha256(report_path)
    if report_sha_after != report_sha_before:
        raise RuntimeError("report.json changed while recording the owner decision")
    file_count = _write_sha256s(artifact_root)
    print(
        json.dumps(
            {
                "artifact_root": str(artifact_root),
                "owner_decision": str(owner_path),
                "formal_artifact_verdict": FORMAL_VERDICT,
                "owner_decision_value": OWNER_DECISION,
                "report_json_sha256": report_sha_after,
                "sha256sum_entries": file_count,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
