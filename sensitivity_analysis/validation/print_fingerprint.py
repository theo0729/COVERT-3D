"""Print the current production implementation fingerprint as JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sensitivity_analysis._fingerprint import implementation_metadata


if __name__ == "__main__":
    print(json.dumps(implementation_metadata(), separators=(",", ":")))
