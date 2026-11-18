"""Point required filesystem roots at temporary directories.

The assignment happens at import time, before any test module imports the
package, so helpers never fall through to an unset variable.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_ROOT = Path(tempfile.mkdtemp(prefix="scaling_surgical_ai_"))
_VARIABLES = (
    "DATA_ROOT",
    "OUTPUT_ROOT",
    "MODELS_ROOT",
    "HF_HOME",
    "CHOLECT50_ROOT",
    "PITVIS_ROOT",
    "SURGVU_ROOT",
)
for _variable in _VARIABLES:
    _path = _ROOT / _variable.lower()
    _path.mkdir(parents=True, exist_ok=True)
    os.environ[_variable] = str(_path)
os.environ["SDSC_EEA_CLASS_MAPPING"] = str(_ROOT / "class_mapping.json")
