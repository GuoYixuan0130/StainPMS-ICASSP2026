"""Finalize a Stage-1 artifact that completed training before an audit-only bug."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semipms.stage1 import finalize_existing_stage1


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize an existing SemiPMS Stage-1 artifact without training")
    parser.add_argument("--artifact", required=True)
    artifact = finalize_existing_stage1(Path(parser.parse_args().artifact))
    print(f"SemiPMS Stage 1 artifact finalized: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
