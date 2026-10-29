"""Regression check: a file produced here at the runs' own length must match the published one.
usage: compare_outputs.py bytes <new> <published>      (PNG / values json: byte-identical)
       compare_outputs.py json  <new> <published>      (score json: same keys, floats within 1e-9)
Prints one line 'identical|MATCH|DIFFERS  <new>' and exits 1 on a difference.
"""
import json
import math
import sys
from pathlib import Path

mode, new, pub = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])


def same(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, str) and isinstance(b, str):
        try:
            a, b = float(a), float(b)
        except ValueError:
            return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= 1e-9 * max(1.0, abs(b))
    return a == b


if mode == "bytes":
    good = new.read_bytes() == pub.read_bytes(); tag = "identical" if good else "DIFFERS"
else:
    good = same(json.loads(new.read_text()), json.loads(pub.read_text())); tag = "MATCH" if good else "DIFFERS"
print(f"  {tag:9s} {new}   vs {pub}")
sys.exit(0 if good else 1)
