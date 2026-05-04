from pathlib import Path
from typing import Dict, Literal

import airfrans as af
import numpy as np
import torch
from torch.utils.data import Dataset


class AirfRANSDataset(Dataset):
    """
    AirfRANS dataset wrapper.

    Each sample returns:
        x:       [N, 2] node coordinates
        u:       [N, 4] target field: vx, vy, pressure/rho, nu_t
        surface: [N, 1] surface indicator
        name:    simulation name
    """

    def __init__(
        self,
        root: str = "data/raw/airfrans/Datesets",
        task: Literal["full", "scarce", "reynolds", "aoa"] = "full",
        split: Literal["train", "test"] = "train",
        dtype: torch.dtype = torch.float32,
    ):
        self.root = Path(root)
        self.task = task
        self.split = split
        self.dtype = dtype

        if not self.root.exists():
            raise FileNotFoundError(f"AirfRANS root not found: {self.root}")

        train = split == "train"

        self.data_list, self.names = af.dataset.load(
            root=str(self.root),
            task=task,
            train=train,
        )

        if len(self.data_list) != len(self.names):
            raise RuntimeError(
                f"Data/name mismatch: {len(self.data_list)} vs {len(self.names)}"
            )

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        arr = self.data_list[idx]

        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)

        if arr.ndim != 2 or arr.shape[1] < 12:
            raise ValueError(
                f"Expected array shape [N, >=12], got {arr.shape}"
            )

        # Official AirfRANS layout:
        # 0:2   position
        # 2:4   inlet velocity
        # 4:5   signed distance function
        # 5:7   normals
        # 7:9   velocity target
        # 9:10  pressure target
        # 10:11 turbulent viscosity target
        # 11:12 surface indicator
        x = torch.as_tensor(arr[:, 0:2], dtype=self.dtype)
        u = torch.as_tensor(arr[:, 7:11], dtype=self.dtype)
        surface = torch.as_tensor(arr[:, 11:12], dtype=self.dtype)

        return {
            "x": x,
            "u": u,
            "surface": surface,
            "name": self.names[idx],
        }


if __name__ == "__main__":
    ds_train = AirfRANSDataset(split="train")
    ds_test = AirfRANSDataset(split="test")

    print("Train size:", len(ds_train))
    print("Test size:", len(ds_test))

    sample = ds_train[0]
    print("Name:", sample["name"])
    print("x:", sample["x"].shape, sample["x"].dtype)
    print("u:", sample["u"].shape, sample["u"].dtype)
    print("surface:", sample["surface"].shape, sample["surface"].dtype)

    print("x min/max:", sample["x"].min(dim=0).values, sample["x"].max(dim=0).values)
    print("u mean/std:", sample["u"].mean(dim=0), sample["u"].std(dim=0))