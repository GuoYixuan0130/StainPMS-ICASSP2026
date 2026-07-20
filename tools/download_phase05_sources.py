"""Download only owner-approved Phase 0.5 source assets with byte identities.

No archive is extracted or decoded.  Google Drive responses must provide a
real attachment filename and non-HTML content; otherwise the tool fails rather
than saving a confirmation/error page as data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import requests


ASSETS = {
    "monuseg_train": {
        "provider": "google_drive",
        "file_id": "1ZgqFJomqQGNnsx7w7QBzQQMVA16lbVCA",
    },
    "monuseg_test": {
        "provider": "google_drive",
        "file_id": "1NKkSQ5T0ZNQ8aUhh0a8Dt2YKYCQXIViw",
    },
    "monuseg_organ": {
        "provider": "google_drive",
        "file_id": "1xYyQ31CHFRnvTCTuuHdconlJCMk2SK7Z",
    },
    "monuseg_converter": {
        "provider": "google_drive",
        "file_id": "1YDtIiLZX0lQzZp_JbqneHXHvRo45ZWGX",
    },
    "tnbc_v1_1": {
        "provider": "zenodo",
        "url": (
            "https://zenodo.org/records/2579118/files/"
            "TNBC_NucleiSegmentation.zip?download=1"
        ),
        "expected_filename": "TNBC_NucleiSegmentation.zip",
        "expected_size_bytes": 25232361,
        "expected_md5": "1605712a752b201b57eacc8f866adb4f",
    },
}


def attachment_filename(header: str) -> str:
    extended = re.search(r"filename\*=UTF-8''([^;]+)", header, flags=re.I)
    if extended:
        value = unquote(extended.group(1).strip())
    else:
        basic = re.search(r'filename\s*=\s*(?:"([^"]+)"|([^;]+))', header, flags=re.I)
        if not basic:
            raise ValueError("response has no attachment filename")
        value = (basic.group(1) or basic.group(2)).strip()
    safe = Path(value).name
    if (
        not safe
        or safe in {".", ".."}
        or "/" in value
        or "\\" in value
    ):
        raise ValueError(f"unsafe attachment filename: {value!r}")
    return safe


def _asset_url(spec: dict[str, object]) -> str:
    if spec["provider"] == "google_drive":
        return (
            "https://drive.usercontent.google.com/download"
            f"?id={spec['file_id']}&export=download&confirm=t"
        )
    return str(spec["url"])


def download_asset(asset: str, output_dir: Path) -> dict[str, object]:
    spec = ASSETS[asset]
    url = _asset_url(spec)
    started = datetime.now(timezone.utc).isoformat()
    with requests.get(
        url,
        stream=True,
        timeout=(30, 300),
        allow_redirects=True,
        headers={"User-Agent": "F3C-StainPMS-Phase0.5/1.0"},
    ) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            raise RuntimeError(
                f"{asset} returned HTML instead of an attachment; use browser upload fallback"
            )
        disposition = response.headers.get("Content-Disposition", "")
        filename = attachment_filename(disposition)
        expected_filename = spec.get("expected_filename")
        if expected_filename and filename != expected_filename:
            raise RuntimeError(
                f"{asset} filename {filename!r} != expected {expected_filename!r}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / filename
        partial_path = output_dir / f".{filename}.{asset}.part"
        if final_path.exists() or partial_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing download or partial file: {final_path}"
            )
        md5 = hashlib.md5()  # noqa: S324 - publisher identity includes MD5
        sha256 = hashlib.sha256()
        size = 0
        with partial_path.open("xb") as handle:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if not block:
                    continue
                handle.write(block)
                md5.update(block)
                sha256.update(block)
                size += len(block)
        expected_size = spec.get("expected_size_bytes")
        expected_md5 = spec.get("expected_md5")
        if expected_size is not None and size != expected_size:
            raise RuntimeError(f"{asset} size {size} != expected {expected_size}")
        if expected_md5 is not None and md5.hexdigest() != expected_md5:
            raise RuntimeError(f"{asset} publisher MD5 mismatch")
        partial_path.replace(final_path)
        return {
            "asset": asset,
            "provider": spec["provider"],
            "file_id": spec.get("file_id"),
            "request_url": url,
            "final_url": response.url,
            "download_started_at_utc": started,
            "download_completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "filename": filename,
            "path": str(final_path.resolve()),
            "size_bytes": size,
            "md5": md5.hexdigest(),
            "sha256": sha256.hexdigest(),
            "content_type": response.headers.get("Content-Type"),
            "content_disposition": disposition,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset",
        action="append",
        required=True,
        choices=sorted(ASSETS),
        help="Repeat to download multiple approved assets.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    if args.report.exists():
        print(
            json.dumps(
                {"status": "issues_found", "error": f"refusing to overwrite report: {args.report}"}
            )
        )
        return 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        for asset in args.asset:
            rows.append(download_asset(asset, args.output_dir))
        report = {
            "schema_version": 1,
            "phase": "0.5",
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "assets": rows,
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "phase": "0.5",
            "status": "issues_found",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_assets": rows,
            "error": str(exc),
        }
        args.report.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {"status": "issues_found", "report": str(args.report), "error": str(exc)}
            )
        )
        return 2
    print(json.dumps({"status": "complete", "assets": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
