import csv
from pathlib import Path

import torch

from src.data.airfrans_dataset import AirfRANSDataset
from src.sensors.sampler import sample_sensors
from src.inverse.nearest import nearest_node
from src.inverse.gaussian_soft import gaussian_soft_predict


def evaluate_one_case(
    x,
    u,
    sample_id: int,
    sample_name: str,
    seed: int,
    offset_std: float,
    m: int,
    noise_std: float,
    k: int,
    tau: float,
    chunk_size: int,
):
    torch.manual_seed(seed)

    obs = sample_sensors(
        x=x,
        u=u,
        m=m,
        noise_std=noise_std,
        offset_std=offset_std,
    )

    y = obs["y"]
    s = obs["s"]
    idx_true = obs["idx_true"]

    # -------------------------
    # Nearest
    # -------------------------
    idx_nearest = nearest_node(
        s=s,
        x=x,
        chunk_size=chunk_size,
    )

    y_hat_nearest = u[idx_nearest]

    nearest_mse = ((y - y_hat_nearest) ** 2).mean()
    nearest_acc = (idx_nearest == idx_true).float().mean()

    # -------------------------
    # Gaussian soft
    # -------------------------
    y_hat_gaussian, idx_knn, weights = gaussian_soft_predict(
        s=s,
        x=x,
        u=u,
        k=k,
        tau=tau,
        chunk_size=chunk_size,
    )

    gaussian_mse = ((y - y_hat_gaussian) ** 2).mean()
    true_in_knn_acc = (idx_knn == idx_true.unsqueeze(1)).any(dim=1).float().mean()
    weights_max_mean = weights.max(dim=1).values.mean()

    return {
        "sample_id": sample_id,
        "sample_name": sample_name,
        "seed": seed,
        "offset_std": offset_std,
        "m": m,
        "noise_std": noise_std,
        "k": k,
        "tau": tau,
        "nearest_mse": nearest_mse.item(),
        "gaussian_mse": gaussian_mse.item(),
        "nearest_acc": nearest_acc.item(),
        "true_in_knn_acc": true_in_knn_acc.item(),
        "weights_max_mean": weights_max_mean.item(),
        "improvement": nearest_mse.item() - gaussian_mse.item(),
        "improvement_ratio": gaussian_mse.item() / max(nearest_mse.item(), 1e-12),
    }


def main():
    # -------------------------
    # Settings
    # -------------------------
    root = "data/raw/airfrans/Dataset"
    split = "train"

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "nearest_gaussian_offset.csv"

    # 先不要太大，确认趋势
    num_samples = 5
    seeds = [0, 1, 2]

    offset_list = [0.0, 0.001, 0.002, 0.005, 0.01]

    m = 512
    noise_std = 0.0

    # Gaussian settings
    k = 512
    tau = 1e-3

    chunk_size = 64

    # -------------------------
    # Load dataset
    # -------------------------
    dataset = AirfRANSDataset(
        root=root,
        split=split,
    )

    print("dataset size:", len(dataset))
    print("output csv:", output_csv)
    print()

    rows = []

    for sample_id in range(num_samples):
        sample = dataset[sample_id]

        x = sample["x"]
        u = sample["u"]
        sample_name = sample["name"]

        print("=" * 80)
        print(f"sample_id={sample_id}, name={sample_name}")
        print("x:", x.shape, "u:", u.shape)
        print("=" * 80)

        for seed in seeds:
            for offset_std in offset_list:
                result = evaluate_one_case(
                    x=x,
                    u=u,
                    sample_id=sample_id,
                    sample_name=sample_name,
                    seed=seed,
                    offset_std=offset_std,
                    m=m,
                    noise_std=noise_std,
                    k=k,
                    tau=tau,
                    chunk_size=chunk_size,
                )

                rows.append(result)

                print(
                    f"seed={seed} | "
                    f"offset={offset_std:<7} | "
                    f"nearest_mse={result['nearest_mse']:.4f} | "
                    f"gaussian_mse={result['gaussian_mse']:.4f} | "
                    f"ratio={result['improvement_ratio']:.4f} | "
                    f"nearest_acc={result['nearest_acc']:.4f} | "
                    f"true_in_knn={result['true_in_knn_acc']:.4f} | "
                    f"w_max={result['weights_max_mean']:.4f}"
                )

        print()

    # -------------------------
    # Save CSV
    # -------------------------
    fieldnames = [
        "sample_id",
        "sample_name",
        "seed",
        "offset_std",
        "m",
        "noise_std",
        "k",
        "tau",
        "nearest_mse",
        "gaussian_mse",
        "nearest_acc",
        "true_in_knn_acc",
        "weights_max_mean",
        "improvement",
        "improvement_ratio",
    ]

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print("Saved:", output_csv)
    print("Total rows:", len(rows))
    print("=" * 80)

    # -------------------------
    # Simple summary
    # -------------------------
    print()
    print("Summary by offset:")
    print("-" * 80)

    for offset_std in offset_list:
        subset = [r for r in rows if r["offset_std"] == offset_std]

        nearest_mean = sum(r["nearest_mse"] for r in subset) / len(subset)
        gaussian_mean = sum(r["gaussian_mse"] for r in subset) / len(subset)
        ratio_mean = sum(r["improvement_ratio"] for r in subset) / len(subset)
        true_in_knn_mean = sum(r["true_in_knn_acc"] for r in subset) / len(subset)

        print(
            f"offset={offset_std:<7} | "
            f"nearest_mean={nearest_mean:.4f} | "
            f"gaussian_mean={gaussian_mean:.4f} | "
            f"ratio={ratio_mean:.4f} | "
            f"true_in_knn={true_in_knn_mean:.4f}"
        )


if __name__ == "__main__":
    main()