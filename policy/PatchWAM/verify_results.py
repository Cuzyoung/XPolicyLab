#!/usr/bin/env python3
"""Verify the published 50-task C2R aggregate."""

import json
from pathlib import Path


path = Path(__file__).resolve().parent / "results" / "c2rdr4_140k_100ep_full.json"
data = json.loads(path.read_text())
clean = [value for key, value in data.items() if key.endswith("|clean")]
random = [value for key, value in data.items() if key.endswith("|random")]
assert len(clean) == len(random) == 50
clean_sr = sum(clean) / len(clean) * 100
random_sr = sum(random) / len(random) * 100
average = (clean_sr + random_sr) / 2
assert round(clean_sr, 2) == 91.56
assert round(random_sr, 2) == 66.72
assert round(average, 2) == 79.14
print(f"clean={clean_sr:.2f} random={random_sr:.2f} average={average:.2f}")
