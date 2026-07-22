#!/usr/bin/env python3
"""Resolve one completed Phase 1 output directory from audited metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def resolve_output(
    root: Path,
    *,
    dataset: str,
    processed_records: int,
    required_files: tuple[str, ...],
) -> Path:
    matches: list[Path] = []
    for summary_path in sorted(root.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("dataset") != dataset or summary.get("status") != "complete":
            continue
        if int(summary.get("manifest", {}).get("processed_record_count", -1)) != processed_records:
            continue
        directory = summary_path.parent
        if all((directory / name).is_file() for name in required_files):
            matches.append(directory)

    if len(matches) != 1:
        rendered = [str(path) for path in matches]
        raise RuntimeError(
            f"expected exactly one complete {dataset} Phase 1 output with "
            f"{processed_records} processed records under {root}; found {rendered}"
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--processed-records", type=int, required=True)
    parser.add_argument(
        "--require-file",
        action="append",
        default=[],
        help="Filename required beside summary.json; repeat as needed.",
    )
    args = parser.parse_args()
    print(
        resolve_output(
            args.root,
            dataset=args.dataset,
            processed_records=args.processed_records,
            required_files=tuple(args.require_file),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
