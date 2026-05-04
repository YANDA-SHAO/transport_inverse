import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.airfrans_dataset import AirfRANSDataset
from src.sensors.sampler import sample_sensors
from src.models.coord_latent_decoder import AutoDecoder
from src.inverse.nearest import nearest_node
from src.inverse.gaussian_soft import gaussian_soft_predict
from src.inverse.ot_inverse import (
    build_local_candidates,
    observation_aware_weights,
    predict_observation_from_weights,
    optimize_latent_inverse,
)


def relative_l2(pred, target):
    return torch.norm(pred - target) / torch.norm(target).clamp_min(1e-12)


def denormalize_u(u_norm, u_mean, u_std):
    return u_norm * u_std + u_mean


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    model = AutoDecoder(
        num_samples=len(ckpt["dataset_names"]),
        coord_dim=2,
        latent_dim=ckpt["latent_dim"],
        out_dim=4,
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        num_frequencies=ckpt["num_frequencies"],
        use_fourier=True,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    stats = {
        "x_mean": ckpt["x_mean"].to(device),
        "x_std": ckpt["x_std"].to(device),
        "u_mean": ckpt["u_mean"].to(device),
        "u_std": ckpt["u_std"].to(device),
    }

    return model, stats, ckpt


def optimize_fixed_nearest_inverse(
    decoder_fn,
    mesh_data,
    y,
    idx_nearest,
    u_true_norm,
    u_true,
    u_mean,
    u_std,
    latent_dim,
    device,
    latent_reg=1e-4,
    lr=1e-2,
    n_iters=500,
):
    z = torch.zeros(latent_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)

    history = []

    for _ in range(n_iters):
        optimizer.zero_grad()

        u_hat_norm = decoder_fn(z, mesh_data)
        y_hat = u_hat_norm[idx_nearest]

        data_loss = F.mse_loss(y_hat, y)
        reg_loss = 0.5 * (z ** 2).mean()
        loss = data_loss + latent_reg * reg_loss

        loss.backward()
        optimizer.step()

        history.append(loss.item())

    with torch.no_grad():
        u_hat_norm = decoder_fn(z, mesh_data)
        y_hat = u_hat_norm[idx_nearest]

        obs_mse = F.mse_loss(y_hat, y)
        full_mse_norm = F.mse_loss(u_hat_norm, u_true_norm)
        full_rel_l2_norm = relative_l2(u_hat_norm, u_true_norm)

        u_hat = denormalize_u(u_hat_norm, u_mean, u_std)
        full_mse = F.mse_loss(u_hat, u_true)
        full_rel_l2 = relative_l2(u_hat, u_true)

    return {
        "obs_mse": obs_mse.item(),
        "full_mse_norm": full_mse_norm.item(),
        "full_rel_l2_norm": full_rel_l2_norm.item(),
        "full_mse": full_mse.item(),
        "full_rel_l2": full_rel_l2.item(),
        "final_loss": history[-1],
    }


def optimize_fixed_spatial_gaussian_inverse(
    decoder_fn,
    mesh_data,
    y,
    idx_knn,
    weights,
    u_true_norm,
    u_true,
    u_mean,
    u_std,
    latent_dim,
    device,
    latent_reg=1e-4,
    lr=1e-2,
    n_iters=500,
):
    z = torch.zeros(latent_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)

    history = []

    for _ in range(n_iters):
        optimizer.zero_grad()

        u_hat_norm = decoder_fn(z, mesh_data)
        y_hat = torch.sum(weights.unsqueeze(-1) * u_hat_norm[idx_knn], dim=1)

        data_loss = F.mse_loss(y_hat, y)
        reg_loss = 0.5 * (z ** 2).mean()
        loss = data_loss + latent_reg * reg_loss

        loss.backward()
        optimizer.step()

        history.append(loss.item())

    with torch.no_grad():
        u_hat_norm = decoder_fn(z, mesh_data)
        y_hat = torch.sum(weights.unsqueeze(-1) * u_hat_norm[idx_knn], dim=1)

        obs_mse = F.mse_loss(y_hat, y)
        full_mse_norm = F.mse_loss(u_hat_norm, u_true_norm)
        full_rel_l2_norm = relative_l2(u_hat_norm, u_true_norm)

        u_hat = denormalize_u(u_hat_norm, u_mean, u_std)
        full_mse = F.mse_loss(u_hat, u_true)
        full_rel_l2 = relative_l2(u_hat, u_true)

    return {
        "obs_mse": obs_mse.item(),
        "full_mse_norm": full_mse_norm.item(),
        "full_rel_l2_norm": full_rel_l2_norm.item(),
        "full_mse": full_mse.item(),
        "full_rel_l2": full_rel_l2.item(),
        "final_loss": history[-1],
    }


def evaluate_one_case(
    model,
    stats,
    ckpt,
    sample,
    sample_id,
    seed,
    offset_std,
    noise_std,
    m,
    k,
    spatial_tau,
    feature_tau,
    lambda_feat,
    latent_reg,
    lr,
    n_iters,
    chunk_size,
    device,
):
    torch.manual_seed(seed)

    x = sample["x"].to(device)
    u = sample["u"].to(device)

    x_mean = stats["x_mean"]
    x_std = stats["x_std"]
    u_mean = stats["u_mean"]
    u_std = stats["u_std"]

    x_norm = (x - x_mean) / x_std
    u_norm = (u - u_mean) / u_std

    obs = sample_sensors(
        x=x_norm,
        u=u_norm,
        m=m,
        noise_std=noise_std,
        offset_std=offset_std,
    )

    y = obs["y"].to(device)
    s = obs["s"].to(device)
    idx_true = obs["idx_true"].to(device)

    latent_dim = ckpt["latent_dim"]

    def decoder_fn(z, mesh_data):
        return model.decode(
            x=mesh_data["x_norm"],
            z=z,
        )

    mesh_data = {
        "x_norm": x_norm,
    }

    # --------------------------------------------------
    # Build fixed correspondence baselines
    # --------------------------------------------------
    idx_nearest = nearest_node(
        s=s,
        x=x_norm,
        chunk_size=chunk_size,
    )

    nearest_corr_acc = (idx_nearest == idx_true).float().mean()

    _, idx_knn, spatial_weights = gaussian_soft_predict(
        s=s,
        x=x_norm,
        u=u_norm,
        k=k,
        tau=spatial_tau,
        chunk_size=chunk_size,
    )

    true_in_knn = (idx_knn == idx_true.unsqueeze(1)).any(dim=1).float().mean()
    spatial_wmax = spatial_weights.max(dim=1).values.mean()

    # --------------------------------------------------
    # Nearest-fixed inverse
    # --------------------------------------------------
    nearest_result = optimize_fixed_nearest_inverse(
        decoder_fn=decoder_fn,
        mesh_data=mesh_data,
        y=y,
        idx_nearest=idx_nearest,
        u_true_norm=u_norm,
        u_true=u,
        u_mean=u_mean,
        u_std=u_std,
        latent_dim=latent_dim,
        device=device,
        latent_reg=latent_reg,
        lr=lr,
        n_iters=n_iters,
    )

    # --------------------------------------------------
    # Spatial-Gaussian-fixed inverse
    # --------------------------------------------------
    gaussian_result = optimize_fixed_spatial_gaussian_inverse(
        decoder_fn=decoder_fn,
        mesh_data=mesh_data,
        y=y,
        idx_knn=idx_knn,
        weights=spatial_weights,
        u_true_norm=u_norm,
        u_true=u,
        u_mean=u_mean,
        u_std=u_std,
        latent_dim=latent_dim,
        device=device,
        latent_reg=latent_reg,
        lr=lr,
        n_iters=n_iters,
    )

    # --------------------------------------------------
    # Feature-aware adaptive inverse
    # --------------------------------------------------
    feature_result = optimize_latent_inverse(
        decoder=decoder_fn,
        y=y,
        s=s,
        x=x_norm,
        z_init=torch.zeros(latent_dim, device=device),
        mesh_data=mesh_data,
        k=k,
        tau=feature_tau,
        lambda_feat=lambda_feat,
        latent_reg=latent_reg,
        lr=lr,
        n_iters=n_iters,
        chunk_size=chunk_size,
        normalize_cost=True,
        verbose=False,
    )

    u_hat_feature_norm = feature_result["u_hat"]
    y_hat_feature = feature_result["y_hat"]
    feature_weights = feature_result["weights"]

    with torch.no_grad():
        feature_obs_mse = F.mse_loss(y_hat_feature, y)
        feature_full_mse_norm = F.mse_loss(u_hat_feature_norm, u_norm)
        feature_full_rel_l2_norm = relative_l2(u_hat_feature_norm, u_norm)

        u_hat_feature = denormalize_u(u_hat_feature_norm, u_mean, u_std)
        feature_full_mse = F.mse_loss(u_hat_feature, u)
        feature_full_rel_l2 = relative_l2(u_hat_feature, u)

        feature_wmax = feature_weights.max(dim=1).values.mean()

    return {
        "sample_id": sample_id,
        "sample_name": sample["name"],
        "seed": seed,
        "offset_std": offset_std,
        "noise_std": noise_std,
        "m": m,
        "k": k,
        "spatial_tau": spatial_tau,
        "feature_tau": feature_tau,
        "lambda_feat": lambda_feat,
        "nearest_corr_acc": nearest_corr_acc.item(),
        "true_in_knn": true_in_knn.item(),
        "spatial_wmax": spatial_wmax.item(),
        "feature_wmax": feature_wmax.item(),

        "nearest_obs_mse": nearest_result["obs_mse"],
        "nearest_full_mse_norm": nearest_result["full_mse_norm"],
        "nearest_full_rel_l2_norm": nearest_result["full_rel_l2_norm"],
        "nearest_full_mse": nearest_result["full_mse"],
        "nearest_full_rel_l2": nearest_result["full_rel_l2"],

        "gaussian_obs_mse": gaussian_result["obs_mse"],
        "gaussian_full_mse_norm": gaussian_result["full_mse_norm"],
        "gaussian_full_rel_l2_norm": gaussian_result["full_rel_l2_norm"],
        "gaussian_full_mse": gaussian_result["full_mse"],
        "gaussian_full_rel_l2": gaussian_result["full_rel_l2"],

        "feature_obs_mse": feature_obs_mse.item(),
        "feature_full_mse_norm": feature_full_mse_norm.item(),
        "feature_full_rel_l2_norm": feature_full_rel_l2_norm.item(),
        "feature_full_mse": feature_full_mse.item(),
        "feature_full_rel_l2": feature_full_rel_l2.item(),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt_path = Path("checkpoints/coord_latent_decoder_airfrans.pt")
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "latent_inverse_offset.csv"

    model, stats, ckpt = load_model(ckpt_path, device)

    dataset = AirfRANSDataset(
        root="data/raw/airfrans/Dataset",
        split="test",
    )

    print("test size:", len(dataset))
    print("output csv:", output_csv)

    # Keep this small first. This is expensive.
    num_samples = 3
    seeds = [0]
    offset_list = [0.0, 0.001, 0.002, 0.005, 0.01]

    m = 512
    noise_std = 0.0
    k = 512

    spatial_tau = 1e-3
    feature_tau = 1e-2
    lambda_feat = 10.0

    latent_reg = 1e-4
    lr = 1e-2
    n_iters = 500
    chunk_size = 64

    rows = []

    for sample_id in range(num_samples):
        sample = dataset[sample_id]

        print("=" * 100)
        print(f"sample_id={sample_id}, name={sample['name']}")
        print("=" * 100)

        for seed in seeds:
            for offset_std in offset_list:
                result = evaluate_one_case(
                    model=model,
                    stats=stats,
                    ckpt=ckpt,
                    sample=sample,
                    sample_id=sample_id,
                    seed=seed,
                    offset_std=offset_std,
                    noise_std=noise_std,
                    m=m,
                    k=k,
                    spatial_tau=spatial_tau,
                    feature_tau=feature_tau,
                    lambda_feat=lambda_feat,
                    latent_reg=latent_reg,
                    lr=lr,
                    n_iters=n_iters,
                    chunk_size=chunk_size,
                    device=device,
                )

                rows.append(result)

                print(
                    f"seed={seed} | "
                    f"offset={offset_std:<7} | "
                    f"nearest_rel={result['nearest_full_rel_l2']:.4f} | "
                    f"gaussian_rel={result['gaussian_full_rel_l2']:.4f} | "
                    f"feature_rel={result['feature_full_rel_l2']:.4f} | "
                    f"nearest_obs={result['nearest_obs_mse']:.6f} | "
                    f"gaussian_obs={result['gaussian_obs_mse']:.6f} | "
                    f"feature_obs={result['feature_obs_mse']:.6f} | "
                    f"nearest_acc={result['nearest_corr_acc']:.4f} | "
                    f"true_knn={result['true_in_knn']:.4f} | "
                    f"f_wmax={result['feature_wmax']:.4f}"
                )

    fieldnames = list(rows[0].keys())

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

        nearest_rel = sum(r["nearest_full_rel_l2"] for r in subset) / len(subset)
        gaussian_rel = sum(r["gaussian_full_rel_l2"] for r in subset) / len(subset)
        feature_rel = sum(r["feature_full_rel_l2"] for r in subset) / len(subset)

        nearest_obs = sum(r["nearest_obs_mse"] for r in subset) / len(subset)
        gaussian_obs = sum(r["gaussian_obs_mse"] for r in subset) / len(subset)
        feature_obs = sum(r["feature_obs_mse"] for r in subset) / len(subset)

        print(
            f"offset={offset_std:<7} | "
            f"nearest_rel={nearest_rel:.4f} | "
            f"gaussian_rel={gaussian_rel:.4f} | "
            f"feature_rel={feature_rel:.4f} | "
            f"nearest_obs={nearest_obs:.6f} | "
            f"gaussian_obs={gaussian_obs:.6f} | "
            f"feature_obs={feature_obs:.6f}"
        )


if __name__ == "__main__":
    main()