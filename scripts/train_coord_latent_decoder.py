import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.airfrans_dataset import AirfRANSDataset
from src.models.coord_latent_decoder import AutoDecoder


def collate_keep_list(batch):
    """
    AirfRANS samples have different numbers of nodes.
    So we keep them as a list instead of stacking.
    """
    return batch


def sample_nodes(x, u, num_points: int):
    """
    Randomly subsample mesh nodes for efficient training.

    Args:
        x: [N, 2]
        u: [N, 4]
        num_points: number of nodes to sample

    Returns:
        x_sub: [num_points, 2]
        u_sub: [num_points, 4]
    """
    n = x.shape[0]

    if num_points >= n:
        return x, u

    idx = torch.randperm(n, device=x.device)[:num_points]

    return x[idx], u[idx]


def compute_train_stats(dataset, num_samples_for_stats: int = 20):
    """
    Compute rough normalization statistics from a subset of training samples.

    Returns:
        x_mean, x_std, u_mean, u_std
    """
    xs = []
    us = []

    n = min(num_samples_for_stats, len(dataset))

    for i in range(n):
        sample = dataset[i]
        x = sample["x"]
        u = sample["u"]

        # Subsample to avoid huge memory use
        num = min(20000, x.shape[0])
        idx = torch.randperm(x.shape[0])[:num]

        xs.append(x[idx])
        us.append(u[idx])

    x_all = torch.cat(xs, dim=0)
    u_all = torch.cat(us, dim=0)

    x_mean = x_all.mean(dim=0)
    x_std = x_all.std(dim=0).clamp_min(1e-6)

    u_mean = u_all.mean(dim=0)
    u_std = u_all.std(dim=0).clamp_min(1e-6)

    return x_mean, x_std, u_mean, u_std


def normalize_x(x, x_mean, x_std):
    return (x - x_mean) / x_std


def normalize_u(u, u_mean, u_std):
    return (u - u_mean) / u_std


def main():
    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    root = "data/raw/airfrans/Dataset"
    output_dir = Path("checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / "coord_latent_decoder_airfrans.pt"

    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    dataset = AirfRANSDataset(
        root=root,
        split="train",
    )

    print("train size:", len(dataset))

    # Important:
    # Each AirfRANS sample has different node count,
    # so batch_size should stay small.
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_keep_list,
    )

    # --------------------------------------------------
    # Normalization stats
    # --------------------------------------------------
    print("computing normalization stats...")
    x_mean, x_std, u_mean, u_std = compute_train_stats(
        dataset,
        num_samples_for_stats=20,
    )

    x_mean = x_mean.to(device)
    x_std = x_std.to(device)
    u_mean = u_mean.to(device)
    u_std = u_std.to(device)

    print("x_mean:", x_mean)
    print("x_std:", x_std)
    print("u_mean:", u_mean)
    print("u_std:", u_std)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    latent_dim = 64
    hidden_dim = 256
    num_layers = 5
    num_frequencies = 6

    model = AutoDecoder(
        num_samples=len(dataset),
        coord_dim=2,
        latent_dim=latent_dim,
        out_dim=4,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_frequencies=num_frequencies,
        use_fourier=True,
    ).to(device)

    # Separate LR for decoder and latent codes
    optimizer = torch.optim.Adam(
        [
            {
                "params": model.decoder.parameters(),
                "lr": 1e-4,
            },
            {
                "params": model.latent_codes.parameters(),
                "lr": 1e-3,
            },
        ],
        weight_decay=0.0,
    )

    # --------------------------------------------------
    # Training settings
    # --------------------------------------------------
    epochs = 50
    num_points = 8192
    latent_reg = 1e-4
    print_every = 20
    save_every_epoch = 10

    global_step = 0
    best_loss = float("inf")

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(1, epochs + 1):
        model.train()

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_reg = 0.0
        count = 0

        for batch in dataloader:
            sample = batch[0]

            # Important:
            # DataLoader returns samples in shuffled order,
            # but we need the original dataset index for latent code.
            #
            # AirfRANSDataset currently does not return idx.
            # So we recover it by matching sample name.
            #
            # This is safe but slightly inefficient.
            sample_name = sample["name"]
            sample_id = dataset.names.index(sample_name)

            x = sample["x"].to(device)
            u = sample["u"].to(device)

            x, u = sample_nodes(
                x=x,
                u=u,
                num_points=num_points,
            )

            x_norm = normalize_x(x, x_mean, x_std)
            u_norm = normalize_u(u, u_mean, u_std)

            optimizer.zero_grad()

            u_pred = model(
                x=x_norm,
                sample_id=sample_id,
            )

            recon_loss = F.mse_loss(u_pred, u_norm)

            z = model.latent_codes.weight[sample_id]
            reg_loss = 0.5 * (z ** 2).mean()

            loss = recon_loss + latent_reg * reg_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_reg += reg_loss.item()
            count += 1
            global_step += 1

            if global_step % print_every == 0:
                print(
                    f"epoch={epoch:03d} | "
                    f"step={global_step:06d} | "
                    f"loss={loss.item():.6f} | "
                    f"recon={recon_loss.item():.6f} | "
                    f"reg={reg_loss.item():.6f}"
                )

        epoch_loss /= max(count, 1)
        epoch_recon /= max(count, 1)
        epoch_reg /= max(count, 1)

        print(
            f"[epoch {epoch:03d}] "
            f"loss={epoch_loss:.6f} | "
            f"recon={epoch_recon:.6f} | "
            f"reg={epoch_reg:.6f}"
        )

        # Save best checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss

            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "latent_dim": latent_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_frequencies": num_frequencies,
                "x_mean": x_mean.detach().cpu(),
                "x_std": x_std.detach().cpu(),
                "u_mean": u_mean.detach().cpu(),
                "u_std": u_std.detach().cpu(),
                "dataset_names": dataset.names,
            }

            torch.save(ckpt, ckpt_path)
            print("saved best checkpoint:", ckpt_path)

        # Periodic checkpoint
        if epoch % save_every_epoch == 0:
            periodic_path = output_dir / f"coord_latent_decoder_airfrans_epoch_{epoch:03d}.pt"

            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "latent_dim": latent_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_frequencies": num_frequencies,
                "x_mean": x_mean.detach().cpu(),
                "x_std": x_std.detach().cpu(),
                "u_mean": u_mean.detach().cpu(),
                "u_std": u_std.detach().cpu(),
                "dataset_names": dataset.names,
            }

            torch.save(ckpt, periodic_path)
            print("saved periodic checkpoint:", periodic_path)

    print("training finished")
    print("best loss:", best_loss)
    print("best checkpoint:", ckpt_path)

if __name__ == "__main__":
    main()