import torch

from src.data.airfrans_dataset import AirfRANSDataset
from src.sensors.sampler import sample_sensors
from src.inverse.nearest import nearest_node, nearest_prediction


ds = AirfRANSDataset(split="train")
sample = ds[0]

obs = sample_sensors(
    sample["x"],
    sample["u"],
    m=512,
    noise_std=0.0,
    offset_std=0.01,
)

idx_nearest = nearest_node(
    s=obs["s"],
    x=sample["x"],
    chunk_size=64,
)

y_hat, idx2 = nearest_prediction(
    s=obs["s"],
    x=sample["x"],
    u=sample["u"],
    chunk_size=64,
)

print("idx_nearest:", idx_nearest.shape)
print("y_hat:", y_hat.shape)

same = torch.equal(idx_nearest, idx2)
print("same idx:", same)

corr_acc = (idx_nearest == obs["idx_true"]).float().mean()
mse = ((obs["y"] - y_hat) ** 2).mean()

print("correspondence accuracy:", corr_acc.item())
print("nearest observation mse:", mse.item())