"""Read-only schema aid for selecting the already frozen MoNuSeg-Lite fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resimixpms.manifests import REQUIRED_FROZEN_FILES, validate_frozen_bundle  # noqa: E402


def _shape(value: Any, pointer: str = "") -> dict[str, Any]:
    if isinstance(value, list):
        return {"pointer": pointer or "/", "kind": "list", "count": len(value)}
    if isinstance(value, dict):
        children = []
        for key, child in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            children.append(_shape(child, f"{pointer}/{escaped}"))
        return {"pointer": pointer or "/", "kind": "object", "children": children}
    return {"pointer": pointer or "/", "kind": type(value).__name__}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    options = parser.parse_args()
    bundle = validate_frozen_bundle(options.bundle)
    payload = {"validated_bundle": bundle.as_dict(), "schemas": {}}
    for name in REQUIRED_FROZEN_FILES[:-1]:
        with (bundle.artifact_dir / name).open("r", encoding="utf-8") as handle:
            payload["schemas"][name] = _shape(json.load(handle))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
