import airfrans as af
from pathlib import Path

root = Path("data/raw/airfrans")
root.mkdir(parents=True, exist_ok=True)

print("Downloading AirfRANS to:", root.resolve())

af.dataset.download(
    root=str(root),
    file_name="Dataset",
    unzip=True,
    OpenFOAM=False,
)

print("Done.")