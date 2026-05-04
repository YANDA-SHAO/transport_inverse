import json
from pathlib import Path

DATA_DIR = Path("data/raw/airfrans/Dataset")

print("DATA_DIR:", DATA_DIR.resolve())
print("Exists:", DATA_DIR.exists())

manifest_path = DATA_DIR / "manifest.json"
print("Manifest exists:", manifest_path.exists())

if manifest_path.exists():
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    print("Manifest type:", type(manifest))
    if isinstance(manifest, dict):
        print("Manifest keys:", manifest.keys())
        print(json.dumps(manifest, indent=2)[:3000])

sample_dirs = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()])
print("Number of simulation folders:", len(sample_dirs))

first = sample_dirs[0]
print("First sample folder:", first.name)

print("Files in first sample:")
for p in first.iterdir():
    print(" ", p.name)