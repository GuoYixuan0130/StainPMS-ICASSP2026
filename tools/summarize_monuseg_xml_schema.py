"""Render a concise human-readable summary from an existing XML schema audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # Supports both direct and module execution.
    from tools.audit_monuseg_xml_schema import render_summary
except ModuleNotFoundError:  # pragma: no cover - direct CLI path
    from audit_monuseg_xml_schema import render_summary  # type: ignore[no-redef]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("input must be a JSON object")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_summary(payload), encoding="utf-8")
    except Exception as exc:
        print(json.dumps({"status": "issues_found", "error": str(exc)}))
        return 2
    print(json.dumps({"status": "complete", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
