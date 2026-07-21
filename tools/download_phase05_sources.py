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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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


def hash_file(path: Path) -> tuple[int, str, str]:
    """Return byte count, MD5, and SHA256 without decoding an archive."""
    md5 = hashlib.md5()  # noqa: S324 - publisher identity includes MD5
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            md5.update(block)
            sha256.update(block)
            size += len(block)
    return size, md5.hexdigest(), sha256.hexdigest()


def validate_expected_identity(
    asset: str,
    spec: dict[str, object],
    filename: str,
    size: int,
    md5: str,
) -> None:
    expected_filename = spec.get("expected_filename")
    if expected_filename and filename != expected_filename:
        raise RuntimeError(
            f"{asset} filename {filename!r} != expected {expected_filename!r}"
        )
    expected_size = spec.get("expected_size_bytes")
    if expected_size is not None and size != expected_size:
        raise RuntimeError(f"{asset} size {size} != expected {expected_size}")
    expected_md5 = spec.get("expected_md5")
    if expected_md5 is not None and md5 != expected_md5:
        raise RuntimeError(f"{asset} publisher MD5 mismatch")


def register_existing_asset(asset: str, path: Path) -> dict[str, object]:
    """Register a browser-uploaded approved source without copying or decoding it."""
    spec = ASSETS[asset]
    if not path.is_file():
        raise FileNotFoundError(f"{asset} supplied path is not a regular file: {path}")
    filename = attachment_filename(f'attachment; filename="{path.name}"')
    size, md5, sha256 = hash_file(path)
    validate_expected_identity(asset, spec, filename, size, md5)
    return {
        "asset": asset,
        "provider": spec["provider"],
        "file_id": spec.get("file_id"),
        "request_url": _asset_url(spec),
        "acquisition": "manual_browser_upload",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "path": str(path.resolve()),
        "size_bytes": size,
        "md5": md5,
        "sha256": sha256,
    }


def build_session(retries: int, backoff_seconds: float) -> requests.Session:
    """Return a session that retries transient connection and HTTP failures."""
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff_seconds,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def download_asset(
    asset: str,
    output_dir: Path,
    *,
    session: requests.Session,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
) -> dict[str, object]:
    spec = ASSETS[asset]
    url = _asset_url(spec)
    started = datetime.now(timezone.utc).isoformat()
    with session.get(
        url,
        stream=True,
        timeout=(connect_timeout_seconds, read_timeout_seconds),
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
        validate_expected_identity(asset, spec, filename, size, md5.hexdigest())
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
        choices=sorted(ASSETS),
        help="Repeat to download multiple approved assets.",
    )
    parser.add_argument(
        "--register-existing",
        action="append",
        metavar="ASSET=PATH",
        help=(
            "Register one approved, browser-uploaded raw source file without copying "
            "or decoding it. Repeat as needed."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--connect-timeout-seconds",
        type=float,
        default=120.0,
        help="Per-attempt HTTPS connection timeout (default: 120).",
    )
    parser.add_argument(
        "--read-timeout-seconds",
        type=float,
        default=900.0,
        help="Per-attempt HTTPS read timeout after connection (default: 900).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries after the initial transient HTTPS failure (default: 3).",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=5.0,
        help="Exponential retry backoff factor in seconds (default: 5).",
    )
    return parser.parse_args()


def parse_registered_asset(value: str) -> tuple[str, Path]:
    asset, separator, raw_path = value.partition("=")
    if not separator or asset not in ASSETS or not raw_path:
        raise ValueError(
            "--register-existing must be ASSET=PATH, with ASSET one of: "
            + ", ".join(sorted(ASSETS))
        )
    return asset, Path(raw_path)


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    requested_assets = args.asset or []
    registrations = args.register_existing or []
    if not requested_assets and not registrations:
        print(json.dumps({"status": "issues_found", "error": "provide --asset or --register-existing"}))
        return 2
    if args.report.exists():
        print(
            json.dumps(
                {"status": "issues_found", "error": f"refusing to overwrite report: {args.report}"}
            )
        )
        return 2
    if args.connect_timeout_seconds <= 0 or args.read_timeout_seconds <= 0:
        print(json.dumps({"status": "issues_found", "error": "timeouts must be positive"}))
        return 2
    if args.retries < 0 or args.retry_backoff_seconds < 0:
        print(json.dumps({"status": "issues_found", "error": "retry values must be non-negative"}))
        return 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    session = build_session(args.retries, args.retry_backoff_seconds)
    request_policy = {
        "connect_timeout_seconds": args.connect_timeout_seconds,
        "read_timeout_seconds": args.read_timeout_seconds,
        "retries": args.retries,
        "retry_backoff_seconds": args.retry_backoff_seconds,
    }
    try:
        for asset in requested_assets:
            rows.append(
                download_asset(
                    asset,
                    args.output_dir,
                    session=session,
                    connect_timeout_seconds=args.connect_timeout_seconds,
                    read_timeout_seconds=args.read_timeout_seconds,
                )
            )
        for registration in registrations:
            asset, path = parse_registered_asset(registration)
            rows.append(register_existing_asset(asset, path))
        report = {
            "schema_version": 1,
            "phase": "0.5",
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "request_policy": request_policy,
            "assets": rows,
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "phase": "0.5",
            "status": "issues_found",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "request_policy": request_policy,
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
    finally:
        session.close()
    print(json.dumps({"status": "complete", "assets": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
