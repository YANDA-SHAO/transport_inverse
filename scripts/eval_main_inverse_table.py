import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.airfrans_dataset import AirfRANSDataset
from src.sensors.sampler import sample_sensors
from src.models.coord_latent_decoder import AutoDecoder
from src.inverse.nearest import nearest_node
from src.inverse.gaussian_soft import gaussian_soft_predict
from src.inverse.ot_inverse import optimize_latent_inverse


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


def compute_full_metrics(u_hat_norm, u_norm, u, u_mean, u_std):
    u_hat = denormalize_u(u_hat_norm, u_mean, u_std)

    return {
        "full_mse_norm": F.mse_loss(u_hat_norm, u_norm).item(),
        "full_rel_l2_norm": relative_l2(u_hat_norm, u_norm).item(),
        "full_mse": F.mse_loss(u_hat, u).item(),
        "full_rel_l2": relative_l2(u_hat, u).item(),
    }


def fit_oracle_latent(
    decoder_fn,
    mesh_data,
    u_norm,
    u,
    u_mean,
    u_std,
    latent_dim,
    device,
    lr=1e-2,
    n_iters=1000,
    latent_reg=1e-4,
):
    z = torch.zeros(latent_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)

    for _ in range(n_iters):
        optimizer.zero_grad()

        u_hat_norm = decoder_fn(z, mesh_data)

        recon_loss = F.mse_loss(u_hat_norm, u_norm)
        reg_loss = 0.5 * (z ** 2).mean()
        loss = recon_loss + latent_reg * reg_loss

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        u_hat_norm = decoder_fn(z, mesh_data)
        metrics = compute_full_metrics(
            u_hat_norm=u_hat_norm,
            u_norm=u_norm,
            u=u,
            u_mean=u_mean,
            u_std=u_std,
        )

    return metrics


def fit_nearest_inverse(
    decoder_fn,
    mesh_data,
    y,
    idx_nearest,
    u_norm,
    u,
    u_mean,
    u_std,
    latent_dim,
    device,
    lr=1e-2,
    n_iters=500,
    latent_reg=1e-4,
):
    z = torch.zeros(latent_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)

    for _ in range(n_iters):
        optimizer.zero_grad()

        u_hat_norm = decoder_fn(z, mesh_data)
        y_hat = u_hat_norm[idx_nearest]

        data_loss = F.mse_loss(y_hat, y)
        reg_loss = 0.5 * (z ** 2).mean()
        loss = data_loss + latent_reg * reg_loss

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        u_hat_norm = decoder_fn(z, mesh_data)
        y_hat = u_hat_norm[idx_nearest]

        metrics = compute_full_metrics(
            u_hat_norm=u_hat_norm,
            u_norm=u_norm,
            u=u,
            u_mean=u_mean,
            u_std=u_std,
        )
        metrics["obs_mse"] = F.mse_loss(y_hat, y).item()

    return metrics


def fit_spatial_gaussian_inverse(
    decoder_fn,
    mesh_data,
    y,
    idx_knn,
    weights,
    u_norm,
    u,
    u_mean,
    u_std,
    latent_dim,
    device,
    lr=1e-2,
    n_iters=500,
    latent_reg=1e-4,
):
    z = torch.zeros(latent_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)

    for _ in range(n_iters):
        optimizer.zero_grad()

        u_hat_norm = decoder_fn(z, mesh_data)
        y_hat = torch.sum(weights.unsqueeze(-1) * u_hat_norm[idx_knn], dim=1)

        data_loss = F.mse_loss(y_hat, y)
        reg_loss = 0.5 * (z ** 2).mean()
        loss = data_loss + latent_reg * reg_loss

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        u_hat_norm = decoder_fn(z, mesh_data)
        y_hat = torch.sum(weights.unsqueeze(-1) * u_hat_norm[idx_knn], dim=1)

        metrics = compute_full_metrics(
            u_hat_norm=u_hat_norm,
            u_norm=u_norm,
            u=u,
            u_mean=u_mean,
            u_std=u_std,
        )
        metrics["obs_mse"] = F.mse_loss(y_hat, y).item()

    return metrics


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
    inverse_iters,
    oracle_iters,
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

    latent_dim = ckpt["latent_dim"]

    def decoder_fn(z, mesh_data):
        return model.decode(
            x=mesh_data["x_norm"],
            z=z,
        )

    mesh_data = {"x_norm": x_norm}

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

    # --------------------------------------------------
    # Oracle latent fit: full-field upper bound of current prior
    # --------------------------------------------------
    oracle_metrics = fit_oracle_latent(
        decoder_fn=decoder_fn,
        mesh_data=mesh_data,
        u_norm=u_norm,
        u=u,
        u_mean=u_mean,
        u_std=u_std,
        latent_dim=latent_dim,
        device=device,
        lr=lr,
        n_iters=oracle_iters,
        latent_reg=latent_reg,
    )

    # --------------------------------------------------
    # Build correspondence for fixed baselines
    # --------------------------------------------------
    idx_nearest = nearest_node(
        s=s,
        x=x_norm,
        chunk_size=chunk_size,
    )
    nearest_corr_acc = (idx_nearest == idx_true).float().mean().item()

    _, idx_knn, spatial_weights = gaussian_soft_predict(
        s=s,
        x=x_norm,
        u=u_norm,
        k=k,
        tau=spatial_tau,
        chunk_size=chunk_size,
    )
    true_in_knn = (idx_knn == idx_true.unsqueeze(1)).any(dim=1).float().mean().item()
    spatial_wmax = spatial_weights.max(dim=1).values.mean().item()

    # --------------------------------------------------
    # Nearest-fixed inverse
    # --------------------------------------------------
    nearest_metrics = fit_nearest_inverse(
        decoder_fn=decoder_fn,
        mesh_data=mesh_data,
        y=y,
        idx_nearest=idx_nearest,
        u_norm=u_norm,
        u=u,
        u_mean=u_mean,
        u_std=u_std,
        latent_dim=latent_dim,
        device=device,
        lr=lr,
        n_iters=inverse_iters,
        latent_reg=latent_reg,
    )

    # --------------------------------------------------
    # Spatial-Gaussian-fixed inverse
    # --------------------------------------------------
    gaussian_metrics = fit_spatial_gaussian_inverse(
        decoder_fn=decoder_fn,
        mesh_data=mesh_data,
        y=y,
        idx_knn=idx_knn,
        weights=spatial_weights,
        u_norm=u_norm,
        u=u,
        u_mean=u_mean,
        u_std=u_std,
        latent_dim=latent_dim,
        device=device,
        lr=lr,
        n_iters=inverse_iters,
        latent_reg=latent_reg,
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
        n_iters=inverse_iters,
        chunk_size=chunk_size,
        normalize_cost=True,
        verbose=False,
        detach_matching=False,
    )

    with torch.no_grad():
        u_hat_feature_norm = feature_result["u_hat"]
        y_hat_feature = feature_result["y_hat"]
        weights_feature = feature_result["weights"]

        feature_wmax = weights_feature.max(dim=1).values.mean().item()

        feature_metrics = compute_full_metrics(
            u_hat_norm=u_hat_feature_norm,
            u_norm=u_norm,
            u=u,
            u_mean=u_mean,
            u_std=u_std,
        )
        feature_metrics["obs_mse"] = F.mse_loss(y_hat_feature, y).item()

    # --------------------------------------------------
    # Feature-aware detached inverse
    # --------------------------------------------------
    feature_detach_result = optimize_latent_inverse(
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
        n_iters=inverse_iters,
        chunk_size=chunk_size,
        normalize_cost=True,
        verbose=False,
        detach_matching=True,
    )

    with torch.no_grad():
        u_hat_feature_detach_norm = feature_detach_result["u_hat"]
        y_hat_feature_detach = feature_detach_result["y_hat"]
        weights_feature_detach = feature_detach_result["weights"]

        feature_detach_wmax = weights_feature_detach.max(dim=1).values.mean().item()

        feature_detach_metrics = compute_full_metrics(
            u_hat_norm=u_hat_feature_detach_norm,
            u_norm=u_norm,
            u=u,
            u_mean=u_mean,
            u_std=u_std,
        )
        feature_detach_metrics["obs_mse"] = F.mse_loss(
            y_hat_feature_detach, y
        ).item()

    # --------------------------------------------------
    # Return one row per method
    # --------------------------------------------------
    base = {
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
        "nearest_corr_acc": nearest_corr_acc,
        "true_in_knn": true_in_knn,
        "spatial_wmax": spatial_wmax,
        "feature_wmax": feature_wmax,
        "feature_detach_wmax": feature_detach_wmax,
    }

    rows = []

    rows.append({
        **base,
        "method": "Oracle latent fit",
        "obs_mse": "",
        **oracle_metrics,
    })

    rows.append({
        **base,
        "method": "Nearest-fixed inverse",
        **nearest_metrics,
    })

    rows.append({
        **base,
        "method": "Spatial-Gaussian-fixed inverse",
        **gaussian_metrics,
    })

    rows.append({
        **base,
        "method": "Feature-aware detached inverse",
        **feature_detach_metrics,
    })

    rows.append({
        **base,
        "method": "Feature-aware adaptive inverse",
        **feature_metrics,
    })

    return rows


def summarize(rows, methods, offset_list):
    print()
    print("=" * 120)
    print("Main table: mean full_rel_l2 original")
    print("=" * 120)

    header = "offset".ljust(12)
    for method in methods:
        header += method[:28].ljust(32)
    print(header)

    for offset_std in offset_list:
        line = str(offset_std).ljust(12)

        for method in methods:
            subset = [
                r for r in rows
                if r["offset_std"] == offset_std and r["method"] == method
            ]
            mean_rel = sum(r["full_rel_l2"] for r in subset) / len(subset)
            line += f"{mean_rel:.4f}".ljust(32)

        print(line)

    print()
    print("=" * 120)
    print("Main table: mean observation MSE")
    print("=" * 120)

    header = "offset".ljust(12)
    for method in methods:
        header += method[:28].ljust(32)
    print(header)

    for offset_std in offset_list:
        line = str(offset_std).ljust(12)

        for method in methods:
            subset = [
                r for r in rows
                if r["offset_std"] == offset_std and r["method"] == method
            ]

            if method == "Oracle latent fit":
                line += "N/A".ljust(32)
            else:
                mean_obs = sum(float(r["obs_mse"]) for r in subset) / len(subset)
                line += f"{mean_obs:.6f}".ljust(32)

        print(line)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt_path = Path("checkpoints/coord_latent_decoder_airfrans.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "main_inverse_table.csv"

    model, stats, ckpt = load_model(ckpt_path, device)

    dataset = AirfRANSDataset(
        root="data/raw/airfrans/Dataset",
        split="test",
    )

    print("test size:", len(dataset))
    print("output csv:", output_csv)

    # Keep small first. This script is expensive.
    num_samples = 3
    seeds = [0]
    offset_list = [0, 0.005, 0.01, 0.02]

    m = 512
    noise_std = 0.0
    k = 512

    spatial_tau = 1e-3
    feature_tau = 1e-2
    lambda_feat = 10.0

    latent_reg = 1e-4
    lr = 1e-2
    inverse_iters = 500
    oracle_iters = 1000
    chunk_size = 64

    methods = [
        "Oracle latent fit",
        "Nearest-fixed inverse",
        "Spatial-Gaussian-fixed inverse",
        "Feature-aware detached inverse",
        "Feature-aware adaptive inverse",
    ]

    rows = []

    for sample_id in range(num_samples):
        sample = dataset[sample_id]

        print("=" * 120)
        print(f"sample_id={sample_id}, name={sample['name']}")
        print("=" * 120)

        for seed in seeds:
            for offset_std in offset_list:
                case_rows = evaluate_one_case(
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
                    inverse_iters=inverse_iters,
                    oracle_iters=oracle_iters,
                    chunk_size=chunk_size,
                    device=device,
                )

                rows.extend(case_rows)

                print(f"seed={seed} | offset={offset_std}")
                for r in case_rows:
                    obs_text = "N/A" if r["method"] == "Oracle latent fit" else f"{float(r['obs_mse']):.6f}"
                    print(
                        f"  {r['method']:<34} | "
                        f"rel_l2={r['full_rel_l2']:.4f} | "
                        f"rel_l2_norm={r['full_rel_l2_norm']:.4f} | "
                        f"obs_mse={obs_text}"
                    )

    fieldnames = [
        "sample_id",
        "sample_name",
        "seed",
        "offset_std",
        "noise_std",
        "m",
        "k",
        "spatial_tau",
        "feature_tau",
        "lambda_feat",
        "nearest_corr_acc",
        "true_in_knn",
        "spatial_wmax",
        "feature_wmax",
        "feature_detach_wmax",
        "method",
        "obs_mse",
        "full_mse_norm",
        "full_rel_l2_norm",
        "full_mse",
        "full_rel_l2",
    ]

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 120)
    print("Saved:", output_csv)
    print("Total rows:", len(rows))
    print("=" * 120)

    summarize(rows, methods, offset_list)


if __name__ == "__main__":
    main()