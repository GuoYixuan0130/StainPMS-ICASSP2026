"""One-command frozen DeployPMS Phase 0 entry point."""

from pathlib import Path
import sys

# ``python tools/run_deploypms_phase0.py`` places ``tools/`` rather than the
# repository root on sys.path.  Keep the documented one-command invocation
# independent of the caller's PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploypms.phase0 import main


if __name__ == "__main__":
    raise SystemExit(main())
