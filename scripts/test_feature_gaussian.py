import csv
from pathlib import Path

import torch

from src.data.airfrans_dataset import AirfRANSDataset
from src.sensors.sampler import sample_sensors
from src.inverse.nearest import nearest_node
from src.inverse.gaussian_soft import gaussian_soft_predict
from src.inverse.feature_gaussian_soft import feature_gaussian_soft_predict


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
    spatial_tau: float,
    feature_tau: float,
    lambda_feat: float,
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
    # Spatial Gaussian
    # -------------------------
    y_hat_gaussian, idx_knn, weights = gaussian_soft_predict(
        s=s,
        x=x,
        u=u,
        k=k,
        tau=spatial_tau,
        chunk_size=chunk_size,
    )

    gaussian_mse = ((y - y_hat_gaussian) ** 2).mean()
    true_in_knn_acc = (idx_knn == idx_true.unsqueeze(1)).any(dim=1).float().mean()
    gaussian_w_max = weights.max(dim=1).values.mean()

    # -------------------------
    # Feature-aware Gaussian
    # -------------------------
    y_hat_feature, idx_feat, w_feat = feature_gaussian_soft_predict(
        s=s,
        y=y,
        x=x,
        u=u,
        k=k,
        tau=feature_tau,
        lambda_feat=lambda_feat,
        chunk_size=chunk_size,
        normalize_cost=True,
    )

    feature_mse = ((y - y_hat_feature) ** 2).mean()
    true_in_feat_knn_acc = (
        idx_feat == idx_true.unsqueeze(1)
    ).any(dim=1).float().mean()
    feature_w_max = w_feat.max(dim=1).values.mean()

    return {
        "sample_id": sample_id,
        "sample_name": sample_name,
        "seed": seed,
        "offset_std": offset_std,
        "m": m,
        "noise_std": noise_std,
        "k": k,
        "spatial_tau": spatial_tau,
        "feature_tau": feature_tau,
        "lambda_feat": lambda_feat,
        "nearest_mse": nearest_mse.item(),
        "gaussian_mse": gaussian_mse.item(),
        "feature_mse": feature_mse.item(),
        "nearest_acc": nearest_acc.item(),
        "true_in_knn_acc": true_in_knn_acc.item(),
        "true_in_feat_knn_acc": true_in_feat_knn_acc.item(),
        "gaussian_w_max": gaussian_w_max.item(),
        "feature_w_max": feature_w_max.item(),
        "gaussian_ratio": gaussian_mse.item() / max(nearest_mse.item(), 1e-12),
        "feature_ratio": feature_mse.item() / max(nearest_mse.item(), 1e-12),
        "feature_vs_gaussian_ratio": feature_mse.item() / max(gaussian_mse.item(), 1e-12),
    }


def main():
    root = "data/raw/airfrans/Dataset"
    split = "train"

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "feature_gaussian_offset.csv"

    # Experiment settings
    num_samples = 5
    seeds = [0, 1, 2]
    offset_list = [0.0, 0.001, 0.002, 0.005, 0.01]

    m = 512
    noise_std = 0.0

    k = 512
    spatial_tau = 1e-3

    # Best setting from sanity test
    feature_tau = 1e-2
    lambda_feat = 10.0

    chunk_size = 64

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

        print("=" * 100)
        print(f"sample_id={sample_id}, name={sample_name}")
        print("x:", x.shape, "u:", u.shape)
        print("=" * 100)

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
                    spatial_tau=spatial_tau,
                    feature_tau=feature_tau,
                    lambda_feat=lambda_feat,
                    chunk_size=chunk_size,
                )

                rows.append(result)

                print(
                    f"seed={seed} | "
                    f"offset={offset_std:<7} | "
                    f"nearest={result['nearest_mse']:.4f} | "
                    f"gaussian={result['gaussian_mse']:.4f} | "
                    f"feature={result['feature_mse']:.4f} | "
                    f"g_ratio={result['gaussian_ratio']:.4f} | "
                    f"f_ratio={result['feature_ratio']:.4f} | "
                    f"f/g={result['feature_vs_gaussian_ratio']:.4f} | "
                    f"nearest_acc={result['nearest_acc']:.4f} | "
                    f"true_knn={result['true_in_knn_acc']:.4f} | "
                    f"g_wmax={result['gaussian_w_max']:.4f} | "
                    f"f_wmax={result['feature_w_max']:.4f}"
                )

        print()

    fieldnames = [
        "sample_id",
        "sample_name",
        "seed",
        "offset_std",
        "m",
        "noise_std",
        "k",
        "spatial_tau",
        "feature_tau",
        "lambda_feat",
        "nearest_mse",
        "gaussian_mse",
        "feature_mse",
        "nearest_acc",
        "true_in_knn_acc",
        "true_in_feat_knn_acc",
        "gaussian_w_max",
        "feature_w_max",
        "gaussian_ratio",
        "feature_ratio",
        "feature_vs_gaussian_ratio",
    ]

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 100)
    print("Saved:", output_csv)
    print("Total rows:", len(rows))
    print("=" * 100)

    print()
    print("Summary by offset:")
    print("-" * 100)

    for offset_std in offset_list:
        subset = [r for r in rows if r["offset_std"] == offset_std]

        nearest_mean = sum(r["nearest_mse"] for r in subset) / len(subset)
        gaussian_mean = sum(r["gaussian_mse"] for r in subset) / len(subset)
        feature_mean = sum(r["feature_mse"] for r in subset) / len(subset)

        gaussian_ratio_mean = sum(r["gaussian_ratio"] for r in subset) / len(subset)
        feature_ratio_mean = sum(r["feature_ratio"] for r in subset) / len(subset)
        feature_vs_gaussian_mean = (
            sum(r["feature_vs_gaussian_ratio"] for r in subset) / len(subset)
        )

        nearest_acc_mean = sum(r["nearest_acc"] for r in subset) / len(subset)
        true_knn_mean = sum(r["true_in_knn_acc"] for r in subset) / len(subset)
        gaussian_w_mean = sum(r["gaussian_w_max"] for r in subset) / len(subset)
        feature_w_mean = sum(r["feature_w_max"] for r in subset) / len(subset)

        print(
            f"offset={offset_std:<7} | "
            f"nearest={nearest_mean:.4f} | "
            f"gaussian={gaussian_mean:.4f} | "
            f"feature={feature_mean:.4f} | "
            f"g_ratio={gaussian_ratio_mean:.4f} | "
            f"f_ratio={feature_ratio_mean:.4f} | "
            f"f/g={feature_vs_gaussian_mean:.4f} | "
            f"nearest_acc={nearest_acc_mean:.4f} | "
            f"true_knn={true_knn_mean:.4f} | "
            f"g_wmax={gaussian_w_mean:.4f} | "
            f"f_wmax={feature_w_mean:.4f}"
        )


if __name__ == "__main__":
    main()