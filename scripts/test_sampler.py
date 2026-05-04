from src.data.airfrans_dataset import AirfRANSDataset
from src.sensors.sampler import sample_sensors

ds = AirfRANSDataset(split="train")
sample = ds[0]

obs = sample_sensors(
    sample["x"],
    sample["u"],
    m=512,
    noise_std=0.0,
    offset_std=0.01,
)

print("y:", obs["y"].shape)
print("s:", obs["s"].shape)
print("idx:", obs["idx_true"].shape)