from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.airfrans_dataset import AirfRANSDataset
from src.models.coord_latent_decoder import AutoDecoder


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


def fit_latent_to_full_field(
    model,
    x_norm,
    u_norm,
    latent_dim,
    lr=1e-2,
    n_iters=1000,
    latent_reg=1e-4,
    verbose=True,
):
    device = x_norm.device

    z = torch.zeros(latent_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)

    history = []

    for it in range(n_iters):
        optimizer.zero_grad()

        u_hat_norm = model.decode(x_norm, z)

        recon_loss = F.mse_loss(u_hat_norm, u_norm)
        reg_loss = 0.5 * (z ** 2).mean()
        loss = recon_loss + latent_reg * reg_loss

        loss.backward()
        optimizer.step()

        history.append(
            {
                "loss": loss.item(),
                "recon": recon_loss.item(),
                "reg": reg_loss.item(),
            }
        )

        if verbose and (it % 100 == 0 or it == n_iters - 1):
            rel = relative_l2(u_hat_norm, u_norm)
            print(
                f"iter={it:04d} | "
                f"loss={loss.item():.6f} | "
                f"recon={recon_loss.item():.6f} | "
                f"reg={reg_loss.item():.6f} | "
                f"rel_l2_norm={rel.item():.4f}"
            )

    with torch.no_grad():
        u_hat_norm = model.decode(x_norm, z)

    return z.detach(), u_hat_norm.detach(), history


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

    latent_dim = ckpt["latent_dim"]

    print()
    print("=" * 80)
    print("Oracle latent fit to full field")
    print("=" * 80)

    z_hat, u_hat_norm, history = fit_latent_to_full_field(
        model=model,
        x_norm=x_norm,
        u_norm=u_norm,
        latent_dim=latent_dim,
        lr=1e-2,
        n_iters=1000,
        latent_reg=1e-4,
        verbose=True,
    )

    with torch.no_grad():
        mse_norm = F.mse_loss(u_hat_norm, u_norm)
        rel_l2_norm = relative_l2(u_hat_norm, u_norm)

        u_hat = denormalize_u(u_hat_norm, u_mean, u_std)
        mse_original = F.mse_loss(u_hat, u)
        rel_l2_original = relative_l2(u_hat, u)

        z_norm = torch.norm(z_hat).item()

    print()
    print("=" * 80)
    print("Final oracle fit result")
    print("=" * 80)
    print("mse normalized:", mse_norm.item())
    print("rel l2 normalized:", rel_l2_norm.item())
    print("mse original:", mse_original.item())
    print("rel l2 original:", rel_l2_original.item())
    print("z norm:", z_norm)

    print()
    print("=" * 80)
    print("Interpretation")
    print("=" * 80)
    print("If oracle rel_l2 is high, the decoder prior is the bottleneck.")
    print("If oracle rel_l2 is low but sparse inverse rel_l2 is high, the inverse objective is the bottleneck.")


if __name__ == "__main__":
    main()