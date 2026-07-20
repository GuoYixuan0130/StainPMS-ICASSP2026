"""Resolve extended7 case metadata from the official NCI GDC Cases API.

The output records project, disease, primary site and tissue-source-site (TSS)
metadata together with the exact request URL and a SHA256 of the raw response.
No image data are requested.  ``--offline-response`` supports auditable reruns
from a previously saved GDC JSON response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
GDC_TSS_TABLE = (
    "https://gdc.cancer.gov/resources-tcga-users/"
    "tcga-code-tables/tissue-source-site-codes"
)
FIELDS = [
    "submitter_id",
    "project.project_id",
    "project.name",
    "primary_site",
    "disease_type",
    "tissue_source_site.code",
    "tissue_source_site.name",
    "tissue_source_site.project",
    "tissue_source_site.bcr_id",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _request_url(case_ids: list[str]) -> str:
    filters = {
        "op": "in",
        "content": {"field": "submitter_id", "value": case_ids},
    }
    query = urllib.parse.urlencode(
        {
            "filters": json.dumps(filters, separators=(",", ":")),
            "fields": ",".join(FIELDS),
            "format": "JSON",
            "size": str(max(100, len(case_ids))),
        }
    )
    return f"{GDC_CASES_ENDPOINT}?{query}"


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "F3C-StainPMS-Phase0.5/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _hits(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("hits"), list):
        raise ValueError("GDC response has no data.hits list")
    return [value for value in data["hits"] if isinstance(value, dict)]


def _case_record(hit: dict[str, Any]) -> dict[str, Any]:
    case_id = str(hit.get("submitter_id") or "").upper()
    project = hit.get("project") if isinstance(hit.get("project"), dict) else {}
    tss = (
        hit.get("tissue_source_site")
        if isinstance(hit.get("tissue_source_site"), dict)
        else {}
    )
    barcode_parts = case_id.split("-")
    barcode_tss = barcode_parts[1] if len(barcode_parts) >= 3 else None
    api_tss = tss.get("code")
    return {
        "case": case_id,
        "tcga_project": project.get("project_id"),
        "project_name": project.get("name"),
        "primary_site": hit.get("primary_site"),
        "disease_type": hit.get("disease_type"),
        "tissue_source_site": {
            "barcode_code": barcode_tss,
            "api_code": api_tss,
            "code_matches_barcode": bool(api_tss) and str(api_tss).upper() == barcode_tss,
            "name": tss.get("name"),
            "project": tss.get("project"),
            "bcr_id": tss.get("bcr_id"),
        },
    }


def audit_metadata(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_json(Path(args.release_config))
    declared = {
        str(row["case"]).upper(): dict(row) for row in config["extended7"]
    }
    expected = list(declared)
    if len(expected) != 7 or len(set(expected)) != 7:
        raise ValueError("release config must define seven unique extended7 cases")
    url = _request_url(expected)
    if args.offline_response:
        raw = Path(args.offline_response).read_bytes()
        response_mode = "offline_response"
    else:
        raw = _fetch(url)
        response_mode = "live_gdc_api"
        if args.save_raw_response:
            raw_path = Path(args.save_raw_response)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(raw)
    payload = json.loads(raw.decode("utf-8"))
    records = [_case_record(hit) for hit in _hits(payload)]
    by_case = {record["case"]: record for record in records}
    missing = sorted(set(expected) - set(by_case))
    unexpected = sorted(set(by_case) - set(expected))
    ordered = [by_case[value] for value in expected if value in by_case]
    incomplete = [
        record["case"]
        for record in ordered
        if not record["tcga_project"]
        or not record["primary_site"]
        or not record["disease_type"]
        or not record["tissue_source_site"]["name"]
    ]
    mismatched_tss = [
        record["case"]
        for record in ordered
        if record["tissue_source_site"]["api_code"]
        and not record["tissue_source_site"]["code_matches_barcode"]
    ]
    snapshot_mismatches: list[dict[str, Any]] = []
    for record in ordered:
        expected_row = declared[record["case"]]
        comparisons = {
            "tcga_project": (expected_row.get("tcga_project"), record["tcga_project"]),
            "primary_site": (expected_row.get("primary_site"), record["primary_site"]),
            "disease": (expected_row.get("disease"), record["disease_type"]),
        }
        expected_tss = expected_row.get("tissue_source_site") or {}
        actual_tss = record["tissue_source_site"]
        for field in ("code", "name", "project", "bcr_id"):
            comparisons[f"tissue_source_site.{field}"] = (
                expected_tss.get(field),
                actual_tss.get("api_code" if field == "code" else field),
            )
        for field, (declared_value, api_value) in comparisons.items():
            if declared_value is not None and declared_value != api_value:
                snapshot_mismatches.append(
                    {
                        "case": record["case"],
                        "field": field,
                        "declared": declared_value,
                        "api": api_value,
                    }
                )
    status = (
        "complete"
        if not missing
        and not unexpected
        and not incomplete
        and not mismatched_tss
        and not snapshot_mismatches
        else "issues_found"
    )
    return {
        "schema_version": 1,
        "phase": "0.5",
        "dataset": "monuseg",
        "subset": "extended7",
        "status": status,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "response_mode": response_mode,
        "sources": {
            "gdc_cases_api_request": url,
            "gdc_tissue_source_site_table": GDC_TSS_TABLE,
        },
        "raw_response_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_cases": expected,
        "missing_cases": missing,
        "unexpected_cases": unexpected,
        "incomplete_cases": incomplete,
        "tss_code_mismatches": mismatched_tss,
        "declared_snapshot_mismatches": snapshot_mismatches,
        "records": ordered,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config",
        default="configs/manifests/monuseg_release_v1.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--offline-response", default="")
    parser.add_argument("--save-raw-response", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = audit_metadata(args)
        _write_json(Path(args.output), report)
    except Exception as exc:
        print(json.dumps({"status": "issues_found", "error": str(exc)}))
        return 2
    print(json.dumps({"status": report["status"], "cases": len(report["records"])}))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
