"""Run the New-AC angular-boundary sensitivity experiment."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sensitivity_analysis._runner import run_cli


if __name__ == "__main__":
    raise SystemExit(run_cli(Path(__file__).resolve().parent))
