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


def full_metrics(u_hat_norm, u_norm, u, u_mean, u_std):
    u_hat = denormalize_u(u_hat_norm, u_mean, u_std)
    return {
        "rel_l2_norm": relative_l2(u_hat_norm, u_norm).item(),
        "rel_l2": relative_l2(u_hat, u).item(),
        "mse_norm": F.mse_loss(u_hat_norm, u_norm).item(),
        "mse": F.mse_loss(u_hat, u).item(),
    }


def fit_fixed_inverse(
    decoder_fn,
    mesh_data,
    y,
    idx_obs,
    u_norm,
    u,
    u_mean,
    u_std,
    latent_dim,
    device,
    lr=1e-2,
    n_iters=300,
    latent_reg=1e-4,
):
    z = torch.zeros(latent_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)

    for _ in range(n_iters):
        optimizer.zero_grad()

        u_hat_norm = decoder_fn(z, mesh_data)
        y_hat = u_hat_norm[idx_obs]

        data_loss = F.mse_loss(y_hat, y)
        reg_loss = 0.5 * (z ** 2).mean()
        loss = data_loss + latent_reg * reg_loss

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        u_hat_norm = decoder_fn(z, mesh_data)
        y_hat = u_hat_norm[idx_obs]

        metrics = full_metrics(
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
    n_iters=300,
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

        metrics = full_metrics(
            u_hat_norm=u_hat_norm,
            u_norm=u_norm,
            u=u,
            u_mean=u_mean,
            u_std=u_std,
        )
        metrics["obs_mse"] = F.mse_loss(y_hat, y).item()

    return metrics


def main():
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt_path = Path("checkpoints/coord_latent_decoder_airfrans.pt")
    model, stats, ckpt = load_model(ckpt_path, device)

    dataset = AirfRANSDataset(
        root="data/raw/airfrans/Dataset",
        split="test",
    )

    sample_id = 0
    seed = 0
    offset_std = 0.002
    noise_std = 0.0

    m = 512
    k = 512
    spatial_tau = 1e-3
    feature_tau = 1e-2

    lr = 1e-2
    n_iters = 300
    latent_reg = 1e-4
    chunk_size = 64

    lambda_list = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]

    sample = dataset[sample_id]

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
        return model.decode(x=mesh_data["x_norm"], z=z)

    mesh_data = {"x_norm": x_norm}

    torch.manual_seed(seed)
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

    print("=" * 100)
    print(f"sample_id={sample_id}, name={sample['name']}")
    print(f"offset_std={offset_std}, noise_std={noise_std}, seed={seed}")
    print("=" * 100)

    idx_nearest = nearest_node(
        s=s,
        x=x_norm,
        chunk_size=chunk_size,
    )

    _, idx_knn, spatial_weights = gaussian_soft_predict(
        s=s,
        x=x_norm,
        u=u_norm,
        k=k,
        tau=spatial_tau,
        chunk_size=chunk_size,
    )

    nearest_acc = (idx_nearest == idx_true).float().mean().item()
    true_in_knn = (idx_knn == idx_true.unsqueeze(1)).any(dim=1).float().mean().item()

    print(f"nearest_corr_acc={nearest_acc:.4f}")
    print(f"true_in_knn={true_in_knn:.4f}")
    print()

    print("Running fixed baselines...")

    oracle_corr = fit_fixed_inverse(
        decoder_fn=decoder_fn,
        mesh_data=mesh_data,
        y=y,
        idx_obs=idx_true,
        u_norm=u_norm,
        u=u,
        u_mean=u_mean,
        u_std=u_std,
        latent_dim=latent_dim,
        device=device,
        lr=lr,
        n_iters=n_iters,
        latent_reg=latent_reg,
    )

    nearest = fit_fixed_inverse(
        decoder_fn=decoder_fn,
        mesh_data=mesh_data,
        y=y,
        idx_obs=idx_nearest,
        u_norm=u_norm,
        u=u,
        u_mean=u_mean,
        u_std=u_std,
        latent_dim=latent_dim,
        device=device,
        lr=lr,
        n_iters=n_iters,
        latent_reg=latent_reg,
    )

    gaussian = fit_spatial_gaussian_inverse(
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
        n_iters=n_iters,
        latent_reg=latent_reg,
    )

    print()
    print(f"{'method':<36} {'lambda':>8} {'rel_l2':>10} {'obs_mse':>12}")
    print("-" * 72)
    print(f"{'GABI-oracle-fixed':<36} {'-':>8} {oracle_corr['rel_l2']:>10.4f} {oracle_corr['obs_mse']:>12.6f}")
    print(f"{'GABI-nearest-fixed':<36} {'-':>8} {nearest['rel_l2']:>10.4f} {nearest['obs_mse']:>12.6f}")
    print(f"{'GABI-spatial-gaussian-fixed':<36} {'-':>8} {gaussian['rel_l2']:>10.4f} {gaussian['obs_mse']:>12.6f}")

    for lam in lambda_list:
        result_detach = optimize_latent_inverse(
            decoder=decoder_fn,
            y=y,
            s=s,
            x=x_norm,
            z_init=torch.zeros(latent_dim, device=device),
            mesh_data=mesh_data,
            k=k,
            tau=feature_tau,
            lambda_feat=lam,
            latent_reg=latent_reg,
            lr=lr,
            n_iters=n_iters,
            chunk_size=chunk_size,
            normalize_cost=True,
            verbose=False,
            detach_matching=True,
        )

        with torch.no_grad():
            metrics = full_metrics(
                u_hat_norm=result_detach["u_hat"],
                u_norm=u_norm,
                u=u,
                u_mean=u_mean,
                u_std=u_std,
            )
            obs_mse = F.mse_loss(result_detach["y_hat"], y).item()

        print(f"{'Feature-detached':<36} {lam:>8.2f} {metrics['rel_l2']:>10.4f} {obs_mse:>12.6f}")

    print("-" * 72)
    print("Done.")


if __name__ == "__main__":
    main()