from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.airfrans_dataset import AirfRANSDataset
from src.sensors.sampler import sample_sensors
from src.models.coord_latent_decoder import AutoDecoder
from src.inverse.ot_inverse import optimize_latent_inverse
from src.inverse.nearest import nearest_node
from src.inverse.gaussian_soft import gaussian_soft_predict
from src.inverse.feature_gaussian_soft import feature_gaussian_soft_predict


def denormalize_u(u_norm, u_mean, u_std):
    return u_norm * u_std + u_mean


def relative_l2(pred, target):
    return torch.norm(pred - target) / torch.norm(target).clamp_min(1e-12)


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


def main():
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt_path = Path("checkpoints/coord_latent_decoder_airfrans.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, stats, ckpt = load_model(ckpt_path, device)

    x_mean = stats["x_mean"]
    x_std = stats["x_std"]
    u_mean = stats["u_mean"]
    u_std = stats["u_std"]

    # --------------------------------------------------
    # Load test sample
    # --------------------------------------------------
    dataset = AirfRANSDataset(
        root="data/raw/airfrans/Dataset",
        split="test",
    )

    sample_id = 0
    sample = dataset[sample_id]

    x = sample["x"].to(device)
    u = sample["u"].to(device)

    x_norm = (x - x_mean) / x_std
    u_norm = (u - u_mean) / u_std

    print("sample:", sample["name"])
    print("x:", x.shape)
    print("u:", u.shape)

    # --------------------------------------------------
    # Sensor setting
    # --------------------------------------------------
    m = 512
    noise_std = 0.0
    offset_std = 0.002

    k = 512
    spatial_tau = 1e-3
    feature_tau = 1e-2
    lambda_feat = 10.0
    chunk_size = 64

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

    print("y:", y.shape)
    print("s:", s.shape)

    # --------------------------------------------------
    # Observation-space baselines
    # These do not reconstruct full field.
    # They only show correspondence / observation fitting quality.
    # --------------------------------------------------
    idx_nearest = nearest_node(
        s=s,
        x=x_norm,
        chunk_size=chunk_size,
    )
    y_hat_nearest = u_norm[idx_nearest]

    nearest_obs_mse = F.mse_loss(y_hat_nearest, y)
    nearest_acc = (idx_nearest == idx_true).float().mean()

    y_hat_gaussian, idx_knn, w_gaussian = gaussian_soft_predict(
        s=s,
        x=x_norm,
        u=u_norm,
        k=k,
        tau=spatial_tau,
        chunk_size=chunk_size,
    )

    gaussian_obs_mse = F.mse_loss(y_hat_gaussian, y)
    true_in_knn = (idx_knn == idx_true.unsqueeze(1)).any(dim=1).float().mean()

    y_hat_feature, idx_feat, w_feat = feature_gaussian_soft_predict(
        s=s,
        y=y,
        x=x_norm,
        u=u_norm,
        k=k,
        tau=feature_tau,
        lambda_feat=lambda_feat,
        chunk_size=chunk_size,
        normalize_cost=True,
    )

    feature_obs_mse = F.mse_loss(y_hat_feature, y)

    print()
    print("=" * 80)
    print("Observation-space correspondence baselines")
    print("=" * 80)
    print("nearest obs mse:", nearest_obs_mse.item())
    print("nearest corr acc:", nearest_acc.item())
    print("spatial gaussian obs mse:", gaussian_obs_mse.item())
    print("feature gaussian obs mse:", feature_obs_mse.item())
    print("true in kNN:", true_in_knn.item())
    print("gaussian wmax:", w_gaussian.max(dim=1).values.mean().item())
    print("feature wmax:", w_feat.max(dim=1).values.mean().item())

    # --------------------------------------------------
    # Latent inverse reconstruction
    # --------------------------------------------------
    latent_dim = ckpt["latent_dim"]
    z_init = torch.zeros(latent_dim, device=device)

    def decoder_fn(z, mesh_data):
        return model.decode(
            x=mesh_data["x_norm"],
            z=z,
        )

    mesh_data = {
        "x_norm": x_norm,
    }

    print()
    print("=" * 80)
    print("Latent inverse optimization")
    print("=" * 80)

    result = optimize_latent_inverse(
        decoder=decoder_fn,
        y=y,
        s=s,
        x=x_norm,
        z_init=z_init,
        mesh_data=mesh_data,
        k=k,
        tau=feature_tau,
        lambda_feat=lambda_feat,
        latent_reg=1e-4,
        lr=1e-2,
        n_iters=500,
        chunk_size=chunk_size,
        normalize_cost=True,
        verbose=True,
    )

    u_hat_norm = result["u_hat"]
    y_hat_inv = result["y_hat"]
    weights_inv = result["weights"]

    # --------------------------------------------------
    # Metrics in normalized space
    # --------------------------------------------------
    full_mse_norm = F.mse_loss(u_hat_norm, u_norm)
    full_rel_l2_norm = relative_l2(u_hat_norm, u_norm)
    obs_mse_inv = F.mse_loss(y_hat_inv, y)

    # --------------------------------------------------
    # Metrics in physical / original field scale
    # --------------------------------------------------
    u_hat = denormalize_u(u_hat_norm, u_mean, u_std)

    full_mse = F.mse_loss(u_hat, u)
    full_rel_l2 = relative_l2(u_hat, u)

    print()
    print("=" * 80)
    print("Final inverse results")
    print("=" * 80)
    print("observation mse inverse:", obs_mse_inv.item())
    print("full field mse normalized:", full_mse_norm.item())
    print("full field rel l2 normalized:", full_rel_l2_norm.item())
    print("full field mse original:", full_mse.item())
    print("full field rel l2 original:", full_rel_l2.item())
    print("inverse weights wmax:", weights_inv.max(dim=1).values.mean().item())

    print()
    print("=" * 80)
    print("Comparison")
    print("=" * 80)
    print("nearest obs mse:", nearest_obs_mse.item())
    print("spatial gaussian obs mse:", gaussian_obs_mse.item())
    print("feature gaussian obs mse:", feature_obs_mse.item())
    print("inverse obs mse:", obs_mse_inv.item())


if __name__ == "__main__":
    main()