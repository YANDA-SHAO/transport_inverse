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
    n_iters=500,
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

        metrics = full_metrics(
            u_hat_norm=u_hat_norm,
            u_norm=u_norm,
            u=u,
            u_mean=u_mean,
            u_std=u_std,
        )

    return metrics


def optimize_latent_inverse_warmup(
    decoder,
    y,
    s,
    x,
    z_init,
    mesh_data,
    u_norm,
    u,
    u_mean,
    u_std,
    latent_dim,
    device,
    k=512,
    tau=1e-2,
    lambda_feat=0.3,
    warmup_iters=100,
    total_iters=500,
    latent_reg=1e-4,
    lr=1e-2,
    chunk_size=64,
):
    # phase 1: spatial only
    result = optimize_latent_inverse(
        decoder=decoder,
        y=y,
        s=s,
        x=x,
        z_init=z_init,
        mesh_data=mesh_data,
        k=k,
        tau=tau,
        lambda_feat=0.0,
        latent_reg=latent_reg,
        lr=lr,
        n_iters=warmup_iters,
        chunk_size=chunk_size,
        normalize_cost=True,
        verbose=False,
        detach_matching=True,
    )

    # phase 2: feature on
    result = optimize_latent_inverse(
        decoder=decoder,
        y=y,
        s=s,
        x=x,
        z_init=result["z_hat"],
        mesh_data=mesh_data,
        k=k,
        tau=tau,
        lambda_feat=lambda_feat,
        latent_reg=latent_reg,
        lr=lr,
        n_iters=total_iters - warmup_iters,
        chunk_size=chunk_size,
        normalize_cost=True,
        verbose=False,
        detach_matching=True,
    )

    with torch.no_grad():
        metrics = full_metrics(
            u_hat_norm=result["u_hat"],
            u_norm=u_norm,
            u=u,
            u_mean=u_mean,
            u_std=u_std,
        )
        metrics["obs_mse"] = F.mse_loss(result["y_hat"], y).item()

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

    seed = 0
    offset_std = 0.002
    noise_std = 0.0
    k = 512
    lr = 1e-2
    latent_reg = 1e-4
    chunk_size = 64
    feature_tau = 1e-2

    print("=" * 100)
    print(f"sample_id={sample_id}, name={sample['name']}")
    print(f"offset_std={offset_std}, noise_std={noise_std}, seed={seed}")
    print("=" * 100)

    # --------------------------------------------------
    # 1. Oracle latent fit
    # --------------------------------------------------
    print("\n[1] Oracle latent fit")
    oracle = fit_oracle_latent(
        decoder_fn=decoder_fn,
        mesh_data=mesh_data,
        u_norm=u_norm,
        u=u,
        u_mean=u_mean,
        u_std=u_std,
        latent_dim=latent_dim,
        device=device,
        lr=lr,
        n_iters=1000,
        latent_reg=latent_reg,
    )
    print(f"Oracle latent fit | rel_l2={oracle['rel_l2']:.4f}")

    # --------------------------------------------------
    # 2. inverse_iters sweep for GABI-oracle
    # --------------------------------------------------
    print("\n[2] inverse_iters sweep: GABI-oracle correspondence")
    print(f"{'iters':>8} {'rel_l2':>10} {'obs_mse':>12}")

    m = 512
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

    for n_iters in [100, 300, 500, 1000, 2000]:
        metrics = fit_fixed_inverse(
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
        print(f"{n_iters:>8} {metrics['rel_l2']:>10.4f} {metrics['obs_mse']:>12.6f}")

    # --------------------------------------------------
    # 3. m sweep
    # --------------------------------------------------
    print("\n[3] sensor number sweep")
    print(f"{'m':>8} {'nearest_acc':>12} {'GABI-oracle':>14} {'GABI-nearest':>14} {'Feature-0.1':>14} {'Feature-0.3':>14}")

    for m in [128, 256, 512, 1024, 2048]:
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

        idx_nearest = nearest_node(
            s=s,
            x=x_norm,
            chunk_size=chunk_size,
        )
        nearest_acc = (idx_nearest == idx_true).float().mean().item()

        gabi_oracle = fit_fixed_inverse(
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
            n_iters=500,
            latent_reg=latent_reg,
        )

        gabi_nearest = fit_fixed_inverse(
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
            n_iters=500,
            latent_reg=latent_reg,
        )

        feature_01 = optimize_latent_inverse(
            decoder=decoder_fn,
            y=y,
            s=s,
            x=x_norm,
            z_init=torch.zeros(latent_dim, device=device),
            mesh_data=mesh_data,
            k=k,
            tau=feature_tau,
            lambda_feat=0.1,
            latent_reg=latent_reg,
            lr=lr,
            n_iters=500,
            chunk_size=chunk_size,
            normalize_cost=True,
            verbose=False,
            detach_matching=True,
        )
        feature_03 = optimize_latent_inverse(
            decoder=decoder_fn,
            y=y,
            s=s,
            x=x_norm,
            z_init=torch.zeros(latent_dim, device=device),
            mesh_data=mesh_data,
            k=k,
            tau=feature_tau,
            lambda_feat=0.3,
            latent_reg=latent_reg,
            lr=lr,
            n_iters=500,
            chunk_size=chunk_size,
            normalize_cost=True,
            verbose=False,
            detach_matching=True,
        )

        feature_01_metrics = full_metrics(
            feature_01["u_hat"], u_norm, u, u_mean, u_std
        )
        feature_03_metrics = full_metrics(
            feature_03["u_hat"], u_norm, u, u_mean, u_std
        )

        print(
            f"{m:>8} "
            f"{nearest_acc:>12.4f} "
            f"{gabi_oracle['rel_l2']:>14.4f} "
            f"{gabi_nearest['rel_l2']:>14.4f} "
            f"{feature_01_metrics['rel_l2']:>14.4f} "
            f"{feature_03_metrics['rel_l2']:>14.4f}"
        )

    # --------------------------------------------------
    # 4. warm-up sweep
    # --------------------------------------------------
    print("\n[4] warm-up feature sweep")
    print(f"{'warmup':>8} {'lambda':>8} {'rel_l2':>10} {'obs_mse':>12}")

    m = 512
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

    for warmup_iters in [0, 50, 100, 200, 300]:
        for lam in [0.1, 0.3, 0.5, 1.0]:
            if warmup_iters == 0:
                result = optimize_latent_inverse(
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
                    n_iters=500,
                    chunk_size=chunk_size,
                    normalize_cost=True,
                    verbose=False,
                    detach_matching=True,
                )

                with torch.no_grad():
                    metrics = full_metrics(result["u_hat"], u_norm, u, u_mean, u_std)
                    metrics["obs_mse"] = F.mse_loss(result["y_hat"], y).item()
            else:
                metrics = optimize_latent_inverse_warmup(
                    decoder=decoder_fn,
                    y=y,
                    s=s,
                    x=x_norm,
                    z_init=torch.zeros(latent_dim, device=device),
                    mesh_data=mesh_data,
                    u_norm=u_norm,
                    u=u,
                    u_mean=u_mean,
                    u_std=u_std,
                    latent_dim=latent_dim,
                    device=device,
                    k=k,
                    tau=feature_tau,
                    lambda_feat=lam,
                    warmup_iters=warmup_iters,
                    total_iters=500,
                    latent_reg=latent_reg,
                    lr=lr,
                    chunk_size=chunk_size,
                )

            print(
                f"{warmup_iters:>8} {lam:>8.2f} "
                f"{metrics['rel_l2']:>10.4f} {metrics['obs_mse']:>12.6f}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()