from pathlib import Path

import torch
import matplotlib.pyplot as plt

from src.data.airfrans_dataset import AirfRANSDataset
from src.sensors.sampler import sample_sensors
from src.models.coord_latent_decoder import AutoDecoder
from src.inverse.nearest import nearest_node
from src.inverse.gaussian_soft import gaussian_soft_predict


# ============================================================
# Basic utilities
# ============================================================

def load_model_and_stats(ckpt_path, device):
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


def normalize_x(x, x_mean, x_std):
    return (x - x_mean) / x_std


def normalize_u(u, u_mean, u_std):
    return (u - u_mean) / u_std


def denormalize_x(x_norm, x_mean, x_std):
    return x_norm * x_std + x_mean


def denormalize_u(u_norm, u_mean, u_std):
    return u_norm * u_std + u_mean


def geometry_scale(x):
    bbox_min = x.min(dim=0).values
    bbox_max = x.max(dim=0).values
    return torch.norm(bbox_max - bbox_min).item()


def sample_same_indices(n, max_points, device):
    if n <= max_points:
        return torch.arange(n, device=device)
    return torch.randperm(n, device=device)[:max_points]


def scatter_field(
    ax,
    x,
    values,
    title,
    sensors=None,
    cmap="viridis",
    vmin=None,
    vmax=None,
    s=1.0,
):
    sc = ax.scatter(
        x[:, 0].detach().cpu(),
        x[:, 1].detach().cpu(),
        c=values.detach().cpu(),
        s=s,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if sensors is not None:
        ax.scatter(
            sensors[:, 0].detach().cpu(),
            sensors[:, 1].detach().cpu(),
            c="red",
            s=8,
            marker="x",
            linewidths=0.5,
        )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return sc

def print_tensor_stats(name, t):
    t = t.detach()
    print(f"[{name}]")
    print("  shape:", tuple(t.shape))
    print("  mean :", t.mean(dim=0) if t.ndim > 1 else t.mean())
    print("  std  :", t.std(dim=0) if t.ndim > 1 else t.std())
    print("  min  :", t.min(dim=0).values if t.ndim > 1 else t.min())
    print("  max  :", t.max(dim=0).values if t.ndim > 1 else t.max())
    print("  abs mean:", t.abs().mean().item())
    print("  nan count:", torch.isnan(t).sum().item())
    print("  inf count:", torch.isinf(t).sum().item())


def print_field_channel_stats(u, prefix="u"):
    names = ["vx", "vy", "p/rho", "nu_t"]
    for c in range(u.shape[1]):
        name = names[c] if c < len(names) else f"ch{c}"
        vals = u[:, c]
        print(f"[{prefix} {name}] mean={vals.mean().item():.6f}, "
              f"std={vals.std().item():.6f}, "
              f"min={vals.min().item():.6f}, "
              f"max={vals.max().item():.6f}, "
              f"mae_to_zero={vals.abs().mean().item():.6f}")


def print_reconstruction_metrics(name, pred, target):
    err = pred - target
    rel_l2 = torch.linalg.norm(err.reshape(-1)) / torch.linalg.norm(target.reshape(-1)).clamp_min(1e-12)
    mse = torch.mean(err ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(err))
    max_abs = torch.max(torch.abs(err))

    print(f"[{name} reconstruction metrics]")
    print(f"  rel_l2 : {rel_l2.item():.6f}")
    print(f"  mse    : {mse.item():.6f}")
    print(f"  rmse   : {rmse.item():.6f}")
    print(f"  mae    : {mae.item():.6f}")
    print(f"  max_abs: {max_abs.item():.6f}")

    names = ["vx", "vy", "p/rho", "nu_t"]
    for c in range(target.shape[1]):
        cname = names[c] if c < len(names) else f"ch{c}"
        e = pred[:, c] - target[:, c]
        rel = torch.linalg.norm(e) / torch.linalg.norm(target[:, c]).clamp_min(1e-12)
        print(
            f"  {cname:>5}: "
            f"rel_l2={rel.item():.6f}, "
            f"mae={torch.mean(torch.abs(e)).item():.6f}, "
            f"rmse={torch.sqrt(torch.mean(e ** 2)).item():.6f}, "
            f"max_abs={torch.max(torch.abs(e)).item():.6f}"
        )



# ============================================================
# 1. Raw dataset visualization
# ============================================================

def visualize_raw_sample(
    x,
    u,
    out_path,
    max_points=40000,
):
    """
    Visualize raw original-scale dataset fields.
    Channel convention:
        0: vx
        1: vy
        2: pressure / rho
        3: turbulent viscosity
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    idx = sample_same_indices(x.shape[0], max_points, x.device)
    x_s = x[idx]
    u_s = u[idx]

    channel_names = ["vx", "vy", "p/rho", "nu_t"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for c in range(min(4, u.shape[1])):
        vals = u_s[:, c]
        scatter_field(
            axes[c],
            x_s,
            vals,
            title=f"Raw {channel_names[c]}",
            cmap="viridis",
        )

    fig.suptitle("Raw dataset sample")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 2. Normalization sanity check
# ============================================================

def visualize_normalization_check(
    x,
    u,
    x_norm,
    u_norm,
    x_rec,
    u_rec,
    out_path,
    max_points=40000,
):
    """
    Check whether normalization and denormalization are consistent.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_err = torch.norm(x - x_rec, dim=1)
    u_err = torch.norm(u - u_rec, dim=1)

    print("[normalization check]")
    print("x max abs error:", torch.max(torch.abs(x - x_rec)).item())
    print("u max abs error:", torch.max(torch.abs(u - u_rec)).item())
    print("x_norm mean:", x_norm.mean(dim=0))
    print("x_norm std :", x_norm.std(dim=0))
    print("u_norm mean:", u_norm.mean(dim=0))
    print("u_norm std :", u_norm.std(dim=0))

    idx = sample_same_indices(x.shape[0], max_points, x.device)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    scatter_field(
        axes[0],
        x[idx],
        x_err[idx],
        title="x denorm error",
        cmap="hot",
    )

    scatter_field(
        axes[1],
        x[idx],
        u_err[idx],
        title="u denorm error",
        cmap="hot",
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 3. Sensor sampling visualization
# ============================================================

def visualize_sensor_sampling(
    x,
    x_norm,
    u,
    u_norm,
    x_mean,
    x_std,
    out_path,
    m=512,
    noise_std=0.0,
    offset_percent=0.0,
    max_points=40000,
    seed=0,
):
    """
    Visualize:
        - true sensor node locations
        - perturbed sensor locations
        - noisy / clean observations
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scale_norm = geometry_scale(x_norm)
    offset_std = offset_percent / 100.0 * scale_norm

    torch.manual_seed(seed)

    obs = sample_sensors(
        x=x_norm,
        u=u_norm,
        m=m,
        noise_std=noise_std,
        offset_std=offset_std,
    )

    y = obs["y"]
    s_norm = obs["s"]
    idx_true = obs["idx_true"]

    s_raw = denormalize_x(s_norm, x_mean, x_std)
    x_true_raw = x[idx_true]

    idx = sample_same_indices(x.shape[0], max_points, x.device)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # pressure as background
    p = u[idx, 2]

    scatter_field(
        axes[0],
        x[idx],
        p,
        title="GT pressure + true sensor nodes",
        sensors=x_true_raw,
        cmap="viridis",
    )

    scatter_field(
        axes[1],
        x[idx],
        p,
        title=f"GT pressure + perturbed sensors\nnoise={noise_std}, offset={offset_percent}%",
        sensors=s_raw,
        cmap="viridis",
    )

    disp = torch.norm(s_raw - x_true_raw, dim=1)
    axes[2].hist(disp.detach().cpu().numpy(), bins=40)
    axes[2].set_title("sensor displacement histogram")
    axes[2].set_xlabel("distance in raw coordinates")
    axes[2].set_ylabel("count")

    print("[sensor sampling]")
    print("offset_percent:", offset_percent)
    print("offset_std_norm:", offset_std)
    print("noise_std:", noise_std)
    print("mean displacement raw:", disp.mean().item())
    print("max displacement raw :", disp.max().item())
    print("y mean:", y.mean(dim=0))
    print("y std :", y.std(dim=0))

    y_clean = u_norm[idx_true]
    noise = y - y_clean

    print("[sensor observation metrics]")
    print("offset_percent:", offset_percent)
    print("offset_std_norm:", offset_std)
    print("noise_std:", noise_std)
    print("sensor displacement raw mean:", disp.mean().item())
    print("sensor displacement raw std :", disp.std().item())
    print("sensor displacement raw min :", disp.min().item())
    print("sensor displacement raw max :", disp.max().item())

    print_tensor_stats("sensor s_norm", s_norm)
    print_tensor_stats("sensor s_raw", s_raw)
    print_tensor_stats("sensor y", y)
    print_tensor_stats("sensor clean y_true", y_clean)
    print_tensor_stats("sensor noise actual", noise)

    print_reconstruction_metrics("sensor y vs clean y_true", y, y_clean)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return obs


# ============================================================
# 4. Matching debug visualization
# ============================================================

def visualize_matching_debug(
    x,
    x_norm,
    u,
    obs,
    x_mean,
    x_std,
    out_path,
    k=512,
    tau=1e-3,
    chunk_size=64,
    max_points=40000,
):
    """
    Visualize nearest / Gaussian matching relation.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    s_norm = obs["s"]
    idx_true = obs["idx_true"]

    idx_nearest = nearest_node(
        s=s_norm,
        x=x_norm,
        chunk_size=chunk_size,
    )
    idx_true = idx_true.to(idx_nearest.device)
    _, idx_knn, weights = gaussian_soft_predict(
        s=s_norm,
        x=x_norm,
        u=torch.zeros_like(u),
        k=k,
        tau=tau,
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

    max_weight = weights.max(dim=1).values

    rank_true = []
    for i in range(idx_knn.shape[0]):
        hits = torch.where(idx_knn[i] == idx_true[i])[0]
        if len(hits) == 0:
            rank_true.append(k + 1)
        else:
            rank_true.append(int(hits[0].item()) + 1)

    rank_true = torch.tensor(rank_true, device=x.device, dtype=torch.float32)

    s_raw = denormalize_x(s_norm, x_mean, x_std)
    x_true_raw = x[idx_true]
    x_nearest_raw = x[idx_nearest]

    dist_true = torch.norm(s_raw - x_true_raw, dim=1)
    dist_nearest = torch.norm(s_raw - x_nearest_raw, dim=1)

    print("[matching distance metrics]")
    print("true distance mean:", dist_true.mean().item())
    print("true distance std :", dist_true.std().item())
    print("true distance min :", dist_true.min().item())
    print("true distance max :", dist_true.max().item())
    print("nearest distance mean:", dist_nearest.mean().item())
    print("nearest distance std :", dist_nearest.std().item())
    print("nearest distance min :", dist_nearest.min().item())
    print("nearest distance max :", dist_nearest.max().item())

    print("[knn rank metrics]")
    print("true rank mean:", rank_true.mean().item())
    print("true rank median:", rank_true.median().item())
    print("true rank min:", rank_true.min().item())
    print("true rank max:", rank_true.max().item())
    print("true missing count:", (rank_true > k).sum().item())

    print("[gaussian weight metrics]")
    print("weight max mean:", max_weight.mean().item())
    print("weight max std :", max_weight.std().item())
    print("weight max min :", max_weight.min().item())
    print("weight max max :", max_weight.max().item())
    print("weight entropy mean:", (-(weights * torch.log(weights + 1e-12)).sum(dim=1)).mean().item())

    print("[matching debug]")
    print("nearest_acc:", nearest_acc)
    print("true_in_knn :", true_in_knn)
    print("weight max mean:", max_weight.mean().item())
    print("weight max min :", max_weight.min().item())
    print("weight max max :", max_weight.max().item())

    s_raw = denormalize_x(s_norm, x_mean, x_std)
    x_true_raw = x[idx_true]
    x_nearest_raw = x[idx_nearest]

    idx = sample_same_indices(x.shape[0], max_points, x.device)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    scatter_field(
        axes[0],
        x[idx],
        u[idx, 2],
        title="GT pressure + sensor locations",
        sensors=s_raw,
    )

    scatter_field(
        axes[1],
        x[idx],
        u[idx, 2],
        title="GT pressure + true nodes",
        sensors=x_true_raw,
    )

    scatter_field(
        axes[2],
        x[idx],
        u[idx, 2],
        title="GT pressure + nearest nodes",
        sensors=x_nearest_raw,
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    out_dir = Path("debug_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = "checkpoints/coord_latent_decoder_airfrans_strong/best.pt"

    model, stats, ckpt = load_model_and_stats(ckpt_path, device)

    x_mean = stats["x_mean"]
    x_std = stats["x_std"]
    u_mean = stats["u_mean"]
    u_std = stats["u_std"]

    dataset_root = "data/raw/airfrans/Dataset"

    train_dataset = AirfRANSDataset(
        root=dataset_root,
        split="train",
    )

    test_dataset = AirfRANSDataset(
        root=dataset_root,
        split="test",
    )

    # --------------------------------------------------------
    # Debug one train sample and one test sample
    # --------------------------------------------------------
    cases = [
        ("train", train_dataset, 0),
        ("test", test_dataset, 0),
    ]

    for split_name, dataset, sample_id in cases:
        print("=" * 100)
        print(f"DEBUG {split_name} sample {sample_id}")
        print("=" * 100)

        sample = dataset[sample_id]

        x = sample["x"].to(device)
        u = sample["u"].to(device)

        print("sample name:", sample["name"])
        print("x shape:", x.shape)
        print("u shape:", u.shape)
        print("x min:", x.min(dim=0).values)
        print("x max:", x.max(dim=0).values)
        print("u mean:", u.mean(dim=0))
        print("u std :", u.std(dim=0))
        print_tensor_stats("x raw", x)
        print_tensor_stats("u raw", u)
        print_field_channel_stats(u, prefix="raw u")

        x_norm = normalize_x(x, x_mean, x_std)
        u_norm = normalize_u(u, u_mean, u_std)

        x_rec = denormalize_x(x_norm, x_mean, x_std)
        u_rec = denormalize_u(u_norm, u_mean, u_std)
        print_tensor_stats("x_norm", x_norm)
        print_tensor_stats("u_norm", u_norm)
        print_field_channel_stats(u_norm, prefix="norm u")

        print_reconstruction_metrics("x denorm", x_rec, x)
        print_reconstruction_metrics("u denorm", u_rec, u)

        visualize_raw_sample(
            x=x,
            u=u,
            out_path=out_dir / f"{split_name}_sample_{sample_id}_raw_fields.png",
        )

        visualize_normalization_check(
            x=x,
            u=u,
            x_norm=x_norm,
            u_norm=u_norm,
            x_rec=x_rec,
            u_rec=u_rec,
            out_path=out_dir / f"{split_name}_sample_{sample_id}_normalization_check.png",
        )

        # ----------------------------------------------------
        # Sensor sampling: clean
        # ----------------------------------------------------
        obs_clean = visualize_sensor_sampling(
            x=x,
            x_norm=x_norm,
            u=u,
            u_norm=u_norm,
            x_mean=x_mean,
            x_std=x_std,
            out_path=out_dir / f"{split_name}_sample_{sample_id}_sensors_clean.png",
            m=512,
            noise_std=0.0,
            offset_percent=0.0,
            seed=0,
        )

        visualize_matching_debug(
            x=x,
            x_norm=x_norm,
            u=u,
            obs=obs_clean,
            x_mean=x_mean,
            x_std=x_std,
            out_path=out_dir / f"{split_name}_sample_{sample_id}_matching_clean.png",
        )

        # ----------------------------------------------------
        # Sensor sampling: offset only
        # ----------------------------------------------------
        obs_offset = visualize_sensor_sampling(
            x=x,
            x_norm=x_norm,
            u=u,
            u_norm=u_norm,
            x_mean=x_mean,
            x_std=x_std,
            out_path=out_dir / f"{split_name}_sample_{sample_id}_sensors_offset_1pct.png",
            m=512,
            noise_std=0.0,
            offset_percent=0.1,
            seed=0,
        )

        visualize_matching_debug(
            x=x,
            x_norm=x_norm,
            u=u,
            obs=obs_offset,
            x_mean=x_mean,
            x_std=x_std,
            out_path=out_dir / f"{split_name}_sample_{sample_id}_matching_offset_1pct.png",
        )

        # ----------------------------------------------------
        # Sensor sampling: noise + offset
        # ----------------------------------------------------
        obs_noise_offset = visualize_sensor_sampling(
            x=x,
            x_norm=x_norm,
            u=u,
            u_norm=u_norm,
            x_mean=x_mean,
            x_std=x_std,
            out_path=out_dir / f"{split_name}_sample_{sample_id}_sensors_noise_offset.png",
            m=512,
            noise_std=0.01,
            offset_percent=0.1,
            seed=0,
        )

        visualize_matching_debug(
            x=x,
            x_norm=x_norm,
            u=u,
            obs=obs_noise_offset,
            x_mean=x_mean,
            x_std=x_std,
            out_path=out_dir / f"{split_name}_sample_{sample_id}_matching_noise_offset.png",
        )

    print("=" * 100)
    print("Debug visualizations saved to:", out_dir.resolve())
    print("=" * 100)


if __name__ == "__main__":
    main()