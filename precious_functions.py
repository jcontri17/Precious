

import numpy as np
import re
from pathlib import Path





def read_precious_settings(path, required=None):
    """
    Read a text file of `name = value` lines into a dict.

    Rules:
      - order doesn't matter
      - whitespace doesn't matter
      - ignores blank lines and comments (# or //)
      - numbers -> float
      - true/false -> bool
      - everything else -> string
    """
    out = {}

    for raw in Path(path).read_text().splitlines():
        line = raw.strip()

        # skip blanks & comments
        if not line or line.startswith("#") or line.startswith("//"):
            continue

        m = _LINE_RE.match(raw)
        if not m:
            continue  # or raise ValueError for strict mode

        key, val = m.group(1), m.group(2).strip()

        # strip quotes
        if (val.startswith('"') and val.endswith('"')) or \
           (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]

        vlow = val.lower()

        # ---- casting rules ----
        if vlow in ("true", "false"):
            out[key] = (vlow == "true")
        else:
            try:
                out[key] = float(val)   # ALL numbers become floats
            except ValueError:
                out[key] = val          # keep strings as-is

    # optional required-variable check
    if required:
        missing = [k for k in required if k not in out]
        if missing:
            raise KeyError(f"Missing required keys: {missing}")

    return out


_LINE_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*$')

















