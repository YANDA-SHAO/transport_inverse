from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.airfrans_dataset import AirfRANSDataset
from src.sensors.sampler import sample_sensors
from src.models.coord_latent_decoder import AutoDecoder
from src.inverse.nearest import nearest_node
from src.inverse.gaussian_soft import gaussian_soft_predict
from src.inverse.latent_transport import fit_latent_conditioned_transport_inverse
from src.plots.plot_reconstruction import plot_reconstruction


def relative_l2(pred, target):
    return torch.norm(pred - target) / torch.norm(target).clamp_min(1e-12)


def denormalize_u(u_norm, u_mean, u_std):
    return u_norm * u_std + u_mean


def geometry_scale(x):
    bbox_min = x.min(dim=0).values
    bbox_max = x.max(dim=0).values
    return torch.norm(bbox_max - bbox_min).item()


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


@torch.no_grad()
def decode_full_in_chunks(decoder_fn, z, mesh_data, chunk_size=65536):
    x_norm = mesh_data["x_norm"]
    outs = []

    for start in range(0, x_norm.shape[0], chunk_size):
        end = min(start + chunk_size, x_norm.shape[0])
        sub_mesh = {"x_norm": x_norm[start:end]}
        outs.append(decoder_fn(z, sub_mesh))

    return torch.cat(outs, dim=0)


def compute_full_metrics(decoder_fn, z, mesh_data, u_norm, u, u_mean, u_std):
    with torch.no_grad():
        u_hat_norm = decode_full_in_chunks(decoder_fn, z, mesh_data)
        u_hat = denormalize_u(u_hat_norm, u_mean, u_std)
        mae = torch.mean(torch.abs(u_hat - u))

        return {
            "rel_l2_norm": relative_l2(u_hat_norm, u_norm).item(),
            "rel_l2": relative_l2(u_hat, u).item(),
            "mse_norm": F.mse_loss(u_hat_norm, u_norm).item(),
            "mse": F.mse_loss(u_hat, u).item(),
            "mae": mae.item(),
        }


def fit_fixed_inverse_fast(
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
    n_iters=200,
    latent_reg=1e-4,
):
    x_norm = mesh_data["x_norm"]

    unique_idx, inverse_idx = torch.unique(idx_obs, sorted=True, return_inverse=True)
    mesh_obs = {"x_norm": x_norm[unique_idx]}

    z = torch.zeros(latent_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)

    for _ in range(n_iters):
        optimizer.zero_grad(set_to_none=True)

        u_obs_norm = decoder_fn(z, mesh_obs)
        y_hat = u_obs_norm[inverse_idx]

        data_loss = F.mse_loss(y_hat, y)
        reg_loss = 0.5 * (z ** 2).mean()
        loss = data_loss + latent_reg * reg_loss

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        u_obs_norm = decoder_fn(z, mesh_obs)
        y_hat = u_obs_norm[inverse_idx]

    metrics = compute_full_metrics(
        decoder_fn=decoder_fn,
        z=z.detach(),
        mesh_data=mesh_data,
        u_norm=u_norm,
        u=u,
        u_mean=u_mean,
        u_std=u_std,
    )
    metrics["obs_mse"] = F.mse_loss(y_hat, y).item()

    return metrics


def fit_weighted_inverse_fast(
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
    n_iters=200,
    latent_reg=1e-4,
):
    x_norm = mesh_data["x_norm"]

    m, k = idx_knn.shape
    flat_idx = idx_knn.reshape(-1)

    unique_idx, inverse_flat = torch.unique(
        flat_idx,
        sorted=True,
        return_inverse=True,
    )
    inverse_knn = inverse_flat.reshape(m, k)

    mesh_obs = {"x_norm": x_norm[unique_idx]}

    z = torch.zeros(latent_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)

    weights = weights.detach()

    for _ in range(n_iters):
        optimizer.zero_grad(set_to_none=True)

        u_unique_norm = decoder_fn(z, mesh_obs)
        u_knn_norm = u_unique_norm[inverse_knn]

        y_hat = torch.sum(weights.unsqueeze(-1) * u_knn_norm, dim=1)

        data_loss = F.mse_loss(y_hat, y)
        reg_loss = 0.5 * (z ** 2).mean()
        loss = data_loss + latent_reg * reg_loss

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        u_unique_norm = decoder_fn(z, mesh_obs)
        u_knn_norm = u_unique_norm[inverse_knn]
        y_hat = torch.sum(weights.unsqueeze(-1) * u_knn_norm, dim=1)

    metrics = compute_full_metrics(
        decoder_fn=decoder_fn,
        z=z.detach(),
        mesh_data=mesh_data,
        u_norm=u_norm,
        u=u,
        u_mean=u_mean,
        u_std=u_std,
    )
    metrics["obs_mse"] = F.mse_loss(y_hat, y).item()

    return metrics, z.detach()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt_path = "checkpoints/coord_latent_decoder_airfrans_strong/best.pt"
    model, stats, ckpt = load_model(ckpt_path, device)

    dataset = AirfRANSDataset(
        root="data/raw/airfrans/Dataset",
        split="test",
    )

    num_samples = 3
    seed = 0

    noise_std = 0.0
    m = 512
    k = 512

    spatial_tau = 1e-3
    chunk_size = 64

    lr = 1e-2
    n_iters = 200
    latent_reg = 1e-4

    latent_outer_iters = 3
    latent_inner_iters = 100
    latent_value_tau = 1e-2
    latent_alpha_value = 0.05

    offset_percent_list = [0.5, 1.0, 2.0]

    x_mean = stats["x_mean"]
    x_std = stats["x_std"]
    u_mean = stats["u_mean"]
    u_std = stats["u_std"]
    latent_dim = ckpt["latent_dim"]

    all_rows = []

    print("=" * 120)
    print("Running FAST mismatch stress test")
    print(f"num_samples={num_samples}, m={m}, k={k}, noise_std={noise_std}")
    print(f"gaussian_iters={n_iters}")
    print(
        f"latent_outer_iters={latent_outer_iters}, "
        f"latent_inner_iters={latent_inner_iters}, "
        f"alpha={latent_alpha_value}, value_tau={latent_value_tau}"
    )
    print(f"offset_percent_list={offset_percent_list}")
    print("=" * 120)

    for sample_id in range(num_samples):
        sample = dataset[sample_id]

        x = sample["x"].to(device)
        u = sample["u"].to(device)

        x_norm = (x - x_mean) / x_std
        u_norm = (u - u_mean) / u_std

        scale = geometry_scale(x_norm)
        mesh_data = {"x_norm": x_norm}

        def decoder_fn(z, mesh_data):
            return model.decode(x=mesh_data["x_norm"], z=z)

        print("=" * 120)
        print(f"sample_id={sample_id}, name={sample['name']}")
        print(f"geometry_scale_norm={scale:.6f}")
        print("=" * 120)

        header = (
            f"{'offset_%':>9} "
            f"{'nearest':>8} "
            f"{'in_knn':>8} "
            f"{'oracle':>10} "
            f"{'gauss':>10} "
            f"{'latent':>10} "
            f"{'gap_g':>10} "
            f"{'gap_l':>10} "
            f"{'obs_g':>10} "
            f"{'obs_l':>10} "
            f"{'mae_g':>10} "
            f"{'mae_l':>10} "
        )
        print(header)
        print("-" * 120)

        for offset_percent in offset_percent_list:
            offset_std = (offset_percent / 100.0) * scale

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

            _, idx_knn, spatial_weights = gaussian_soft_predict(
                s=s,
                x=x_norm,
                u=u_norm,
                k=k,
                tau=spatial_tau,
                chunk_size=chunk_size,
            )

            nearest_acc = (idx_nearest == idx_true).float().mean().item()
            true_in_knn = (
                (idx_knn == idx_true.unsqueeze(1))
                .any(dim=1)
                .float()
                .mean()
                .item()
            )

            gabi_oracle = fit_fixed_inverse_fast(
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

            gabi_gaussian, z_gauss = fit_weighted_inverse_fast(
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

            gabi_latent, z_latent = fit_latent_conditioned_transport_inverse(
                decoder_fn=decoder_fn,
                mesh_data=mesh_data,
                y=y,
                s=s,
                x_norm=x_norm,
                idx_init=idx_knn,
                weights_init=spatial_weights,
                u_norm=u_norm,
                u=u,
                u_mean=u_mean,
                u_std=u_std,
                latent_dim=latent_dim,
                device=device,
                lr=lr,
                outer_iters=latent_outer_iters,
                inner_iters=latent_inner_iters,
                latent_reg=latent_reg,
                k=k,
                spatial_tau=spatial_tau,
                value_tau=latent_value_tau,
                alpha_value=latent_alpha_value,
                chunk_size=chunk_size,
                return_debug=False,
            )

            # ====== decode full field ======
            u_hat_gaussian = decode_full_in_chunks(decoder_fn, z_gauss, mesh_data)
            u_hat_latent = decode_full_in_chunks(decoder_fn, z_latent, mesh_data)

            u_hat_gaussian = denormalize_u(u_hat_gaussian, u_mean, u_std)
            u_hat_latent = denormalize_u(u_hat_latent, u_mean, u_std)

            # ====== visualize（只画一个 case）======
            if sample_id == 0 and abs(offset_percent - 1.0) < 1e-6:
                plot_reconstruction(
                    x=x,
                    u_gt=u,
                    u_gauss=u_hat_gaussian,
                    u_latent=u_hat_latent,
                    sensors=s,
                    title=f"sample {sample_id} offset {offset_percent}%",
                    save_path=f"plots/sample_{sample_id}_offset_{int(offset_percent)}.png",
                )

            gap_gaussian = gabi_gaussian["rel_l2"] - gabi_oracle["rel_l2"]
            gap_latent = gabi_latent["rel_l2"] - gabi_oracle["rel_l2"]

            row = {
                "sample_id": sample_id,
                "offset_percent": offset_percent,
                "offset_std": offset_std,
                "geometry_scale": scale,
                "nearest_acc": nearest_acc,
                "true_in_knn": true_in_knn,
                "gabi_oracle": gabi_oracle["rel_l2"],
                "gabi_gaussian": gabi_gaussian["rel_l2"],
                "gabi_latent": gabi_latent["rel_l2"],
                "gap_gaussian": gap_gaussian,
                "gap_latent": gap_latent,
                "obs_gaussian": gabi_gaussian["obs_mse"],
                "obs_latent": gabi_latent["obs_mse"],
            }
            all_rows.append(row)

            print(
                f"{offset_percent:9.2f} "
                f"{nearest_acc:8.4f} "
                f"{true_in_knn:8.4f} "
                f"{gabi_oracle['rel_l2']:10.4f} "
                f"{gabi_gaussian['rel_l2']:10.4f} "
                f"{gabi_latent['rel_l2']:10.4f} "
                f"{gap_gaussian:10.4f} "
                f"{gap_latent:10.4f} "
                f"{gabi_gaussian['obs_mse']:10.4f} "
                f"{gabi_latent['obs_mse']:10.4f} "
                f"{gabi_gaussian['mae']:10.4f} "
                f"{gabi_latent['mae']:10.4f}"
            )

        print("-" * 120)

    print()
    print("=" * 120)
    print("SUMMARY: mean over samples")
    print("=" * 120)

    header = (
        f"{'offset_%':>9} "
        f"{'nearest':>8} "
        f"{'in_knn':>8} "
        f"{'oracle':>10} "
        f"{'gauss':>10} "
        f"{'latent':>10} "
        f"{'gap_g':>10} "
        f"{'gap_l':>10} "
        f"{'obs_g':>10} "
        f"{'obs_l':>10}"
    )
    print(header)
    print("-" * 120)

    for offset_percent in offset_percent_list:
        subset = [r for r in all_rows if r["offset_percent"] == offset_percent]

        def mean(key):
            return sum(r[key] for r in subset) / len(subset)

        import math

        def std(key):
            m = mean(key)
            return math.sqrt(sum((r[key] - m) ** 2 for r in subset) / len(subset))

        print(
            f"{offset_percent:9.2f} "
            f"{mean('nearest_acc'):8.4f} "
            f"{mean('true_in_knn'):8.4f} "
            f"{mean('gabi_oracle'):10.4f} "
            f"{mean('gabi_gaussian'):10.4f} "
            f"{mean('gabi_latent'):10.4f} "
            f"{mean('gap_gaussian'):10.4f} "
            f"{mean('gap_latent'):10.4f} "
            f"{mean('obs_gaussian'):10.4f} "
            f"{mean('obs_latent'):10.4f}"
            f"{mean('gabi_gaussian'):10.4f}±{std('gabi_gaussian'):.4f} "
            f"{mean('gabi_latent'):10.4f}±{std('gabi_latent'):.4f} "
        )

    print("-" * 120)
    print("Done.")


if __name__ == "__main__":
    main()