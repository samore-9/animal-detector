"""Send one image to the running server and print the result.

  python scripts/test_image.py path/to/photo.jpg [min_confidence]
"""
import sys

import requests

if len(sys.argv) < 2:
    raise SystemExit("usage: python scripts/test_image.py <image> [min_confidence]")

conf = sys.argv[2] if len(sys.argv) > 2 else "0.35"
with open(sys.argv[1], "rb") as f:
    r = requests.post("http://127.0.0.1:8000/api/detect",
                      files={"file": f}, data={"conf": conf}, timeout=120)
r.raise_for_status()
data = r.json()

print(f"Model: {data['model']}  ({data['ms']} ms)  mode={data['mode']}")
if not data["animals"]:
    print("No animals found.")
for a in data["animals"]:
    print(f"  {a['label']:<14} x{a['count']}  best confidence {a['best_confidence']:.0%}")
