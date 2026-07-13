import sys
from pathlib import Path


# Running ``python tools/...`` makes ``tools/`` the first import location;
# explicitly add the repository root so this entry point is independent of
# the caller's current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semipms.stage1b import main


if __name__ == "__main__":
    raise SystemExit(main())
