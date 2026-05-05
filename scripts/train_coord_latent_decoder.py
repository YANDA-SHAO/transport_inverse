import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.airfrans_dataset import AirfRANSDataset
from src.models.coord_latent_decoder import AutoDecoder


def collate_keep_list(batch):
    return batch


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_nodes(x, u, num_points: int):
    n = x.shape[0]
    if num_points <= 0 or num_points >= n:
        return x, u
    idx = torch.randperm(n, device=x.device)[:num_points]
    return x[idx], u[idx]


def compute_train_stats(dataset, num_samples_for_stats: int, max_points_per_sample: int):
    xs, us = [], []
    n = min(num_samples_for_stats, len(dataset))

    for i in range(n):
        sample = dataset[i]
        x = sample["x"]
        u = sample["u"]

        num = min(max_points_per_sample, x.shape[0])
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


def denormalize_u(u_norm, u_mean, u_std):
    return u_norm * u_std + u_mean


@torch.no_grad()
def decode_in_chunks(model, x_norm, z, chunk_size: int):
    outs = []
    for start in range(0, x_norm.shape[0], chunk_size):
        end = min(start + chunk_size, x_norm.shape[0])
        outs.append(model.decode(x_norm[start:end], z))
    return torch.cat(outs, dim=0)


@torch.no_grad()
def evaluate_train_reconstruction(
    model,
    dataset,
    name_to_idx,
    device,
    x_mean,
    x_std,
    u_mean,
    u_std,
    num_samples: int,
    num_points: int,
):
    model.eval()

    mse_list = []
    rel_l2_list = []

    n = min(num_samples, len(dataset))

    for i in range(n):
        sample = dataset[i]
        sample_id = name_to_idx[sample["name"]]

        x = sample["x"].to(device)
        u = sample["u"].to(device)
        x, u = sample_nodes(x, u, num_points)

        x_norm = normalize_x(x, x_mean, x_std)
        u_norm = normalize_u(u, u_mean, u_std)

        u_pred_norm = model(x_norm, sample_id)
        mse = F.mse_loss(u_pred_norm, u_norm).item()

        u_pred = denormalize_u(u_pred_norm, u_mean, u_std)

        rel_l2 = (
            torch.linalg.norm((u_pred - u).reshape(-1))
            / torch.linalg.norm(u.reshape(-1)).clamp_min(1e-12)
        ).item()

        mse_list.append(mse)
        rel_l2_list.append(rel_l2)

    return {
        "train_eval_mse_norm": float(sum(mse_list) / max(len(mse_list), 1)),
        "train_eval_rel_l2": float(sum(rel_l2_list) / max(len(rel_l2_list), 1)),
    }


def fit_test_oracle_latent(
    decoder_model,
    test_dataset,
    device,
    x_mean,
    x_std,
    u_mean,
    u_std,
    latent_dim: int,
    num_samples: int,
    z_steps: int,
    z_lr: float,
    z_num_points: int,
    latent_reg: float,
    eval_chunk_size: int,
    out_csv: Path,
):
    decoder_model.eval()

    rows = []
    n = min(num_samples, len(test_dataset))

    for sample_idx in range(n):
        sample = test_dataset[sample_idx]
        name = sample["name"]

        x_full = sample["x"].to(device)
        u_full = sample["u"].to(device)

        z = torch.zeros(latent_dim, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([z], lr=z_lr)

        for step in range(1, z_steps + 1):
            x_sub, u_sub = sample_nodes(x_full, u_full, z_num_points)

            x_norm = normalize_x(x_sub, x_mean, x_std)
            u_norm = normalize_u(u_sub, u_mean, u_std)

            optimizer.zero_grad(set_to_none=True)

            u_pred_norm = decoder_model.decode(x_norm, z)
            recon = F.mse_loss(u_pred_norm, u_norm)
            reg = 0.5 * (z ** 2).mean()
            loss = recon + latent_reg * reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            x_norm_full = normalize_x(x_full, x_mean, x_std)
            u_pred_norm_full = decode_in_chunks(
                decoder_model,
                x_norm_full,
                z.detach(),
                chunk_size=eval_chunk_size,
            )
            u_pred_full = denormalize_u(u_pred_norm_full, u_mean, u_std)

            rel_l2 = (
                torch.linalg.norm((u_pred_full - u_full).reshape(-1))
                / torch.linalg.norm(u_full.reshape(-1)).clamp_min(1e-12)
            ).item()

            mae = torch.mean(torch.abs(u_pred_full - u_full)).item()
            rmse = torch.sqrt(torch.mean((u_pred_full - u_full) ** 2)).item()

        row = {
            "sample_idx": sample_idx,
            "name": name,
            "oracle_rel_l2": rel_l2,
            "oracle_mae": mae,
            "oracle_rmse": rmse,
        }
        rows.append(row)

        print(
            f"[test oracle] {sample_idx + 1:03d}/{n:03d} | "
            f"rel_l2={rel_l2:.6f} | mae={mae:.6f} | rmse={rmse:.6f} | {name}"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_idx", "name", "oracle_rel_l2", "oracle_mae", "oracle_rmse"],
        )
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        mean_rel = sum(r["oracle_rel_l2"] for r in rows) / len(rows)
        mean_mae = sum(r["oracle_mae"] for r in rows) / len(rows)
        mean_rmse = sum(r["oracle_rmse"] for r in rows) / len(rows)
    else:
        mean_rel, mean_mae, mean_rmse = float("nan"), float("nan"), float("nan")

    print("=" * 100)
    print("TEST ORACLE LATENT FIT SUMMARY")
    print(f"mean_rel_l2 = {mean_rel:.6f}")
    print(f"mean_mae    = {mean_mae:.6f}")
    print(f"mean_rmse   = {mean_rmse:.6f}")
    print("saved:", out_csv)
    print("=" * 100)

    return {
        "test_oracle_mean_rel_l2": mean_rel,
        "test_oracle_mean_mae": mean_mae,
        "test_oracle_mean_rmse": mean_rmse,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    best_metric,
    config,
    x_mean,
    x_std,
    u_mean,
    u_std,
    dataset_names,
):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "config": config,
        "latent_dim": config["latent_dim"],
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "num_frequencies": config["num_frequencies"],
        "x_mean": x_mean.detach().cpu(),
        "x_std": x_std.detach().cpu(),
        "u_mean": u_mean.detach().cpu(),
        "u_std": u_std.detach().cpu(),
        "dataset_names": dataset_names,
    }
    torch.save(ckpt, path)


def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="data/raw/airfrans/Dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--run_name", type=str, default="coord_latent_decoder_airfrans_strong")

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_frequencies", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_points", type=int, default=16384)
    parser.add_argument("--latent_reg", type=float, default=1e-4)

    parser.add_argument("--decoder_lr", type=float, default=2e-4)
    parser.add_argument("--latent_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=2e-5)

    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--stats_samples", type=int, default=80)
    parser.add_argument("--stats_points", type=int, default=50000)

    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--eval_every_epoch", type=int, default=5)
    parser.add_argument("--save_every_epoch", type=int, default=25)

    parser.add_argument("--train_eval_samples", type=int, default=20)
    parser.add_argument("--train_eval_points", type=int, default=32768)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_amp", action="store_true")

    parser.add_argument("--run_test_oracle", action="store_true")
    parser.add_argument("--test_oracle_samples", type=int, default=20)
    parser.add_argument("--test_z_steps", type=int, default=1000)
    parser.add_argument("--test_z_lr", type=float, default=5e-2)
    parser.add_argument("--test_z_num_points", type=int, default=32768)
    parser.add_argument("--eval_chunk_size", type=int, default=65536)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)

    output_dir = Path(args.output_dir)
    run_dir = output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    log_csv = run_dir / "train_log.csv"
    config_json = run_dir / "config.json"

    config = vars(args).copy()
    config["device"] = str(device)
    config["use_amp"] = use_amp

    with open(config_json, "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 100)
    print("device:", device)
    print("use_amp:", use_amp)
    print("run_dir:", run_dir)
    print("=" * 100)

    train_dataset = AirfRANSDataset(root=args.root, split="train")
    print("train size:", len(train_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_keep_list,
        pin_memory=False,
    )

    name_to_idx = {name: i for i, name in enumerate(train_dataset.names)}

    if args.resume and last_path.exists():
        print("loading resume checkpoint:", last_path)
        ckpt = torch.load(last_path, map_location=device)

        x_mean = ckpt["x_mean"].to(device)
        x_std = ckpt["x_std"].to(device)
        u_mean = ckpt["u_mean"].to(device)
        u_std = ckpt["u_std"].to(device)
    else:
        print("computing normalization stats...")
        x_mean, x_std, u_mean, u_std = compute_train_stats(
            train_dataset,
            num_samples_for_stats=args.stats_samples,
            max_points_per_sample=args.stats_points,
        )
        x_mean = x_mean.to(device)
        x_std = x_std.to(device)
        u_mean = u_mean.to(device)
        u_std = u_std.to(device)

    print("x_mean:", x_mean)
    print("x_std :", x_std)
    print("u_mean:", u_mean)
    print("u_std :", u_std)

    model = AutoDecoder(
        num_samples=len(train_dataset),
        coord_dim=2,
        latent_dim=args.latent_dim,
        out_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_frequencies=args.num_frequencies,
        use_fourier=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.decoder.parameters(), "lr": args.decoder_lr},
            {"params": model.latent_codes.parameters(), "lr": args.latent_lr},
        ],
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    start_epoch = 1
    global_step = 0
    best_metric = float("inf")

    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device)

        try:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            if ckpt.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])

            start_epoch = int(ckpt["epoch"]) + 1
            global_step = int(ckpt.get("global_step", 0))
            best_metric = float(ckpt.get("best_metric", float("inf")))

            print(f"resumed from epoch {start_epoch - 1}, global_step={global_step}")
        except RuntimeError as e:
            print("WARNING: resume failed because architecture/config changed.")
            print(e)
            print("Starting from scratch.")

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_reg = 0.0
        count = 0

        for batch in train_loader:
            sample = batch[0]
            sample_name = sample["name"]
            sample_id = name_to_idx[sample_name]

            x = sample["x"].to(device, non_blocking=True)
            u = sample["u"].to(device, non_blocking=True)

            x, u = sample_nodes(x, u, args.num_points)

            x_norm = normalize_x(x, x_mean, x_std)
            u_norm = normalize_u(u, u_mean, u_std)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                u_pred = model(x=x_norm, sample_id=sample_id)
                recon_loss = F.mse_loss(u_pred, u_norm)

                z = model.latent_codes.weight[sample_id]
                reg_loss = 0.5 * (z ** 2).mean()

                loss = recon_loss + args.latent_reg * reg_loss

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_reg += reg_loss.item()
            count += 1
            global_step += 1

            if global_step % args.print_every == 0:
                print(
                    f"epoch={epoch:03d} | "
                    f"step={global_step:07d} | "
                    f"loss={loss.item():.6f} | "
                    f"recon={recon_loss.item():.6f} | "
                    f"reg={reg_loss.item():.6f} | "
                    f"lr_dec={optimizer.param_groups[0]['lr']:.2e} | "
                    f"lr_z={optimizer.param_groups[1]['lr']:.2e}"
                )

        scheduler.step()

        epoch_loss /= max(count, 1)
        epoch_recon /= max(count, 1)
        epoch_reg /= max(count, 1)

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "loss": epoch_loss,
            "recon": epoch_recon,
            "reg": epoch_reg,
            "lr_decoder": optimizer.param_groups[0]["lr"],
            "lr_latent": optimizer.param_groups[1]["lr"],
        }

        if epoch % args.eval_every_epoch == 0 or epoch == 1 or epoch == args.epochs:
            eval_row = evaluate_train_reconstruction(
                model=model,
                dataset=train_dataset,
                name_to_idx=name_to_idx,
                device=device,
                x_mean=x_mean,
                x_std=x_std,
                u_mean=u_mean,
                u_std=u_std,
                num_samples=args.train_eval_samples,
                num_points=args.train_eval_points,
            )
            row.update(eval_row)

            monitor = eval_row["train_eval_rel_l2"]
        else:
            monitor = epoch_recon

        append_csv(log_csv, row)

        print(
            f"[epoch {epoch:03d}] "
            f"loss={epoch_loss:.6f} | "
            f"recon={epoch_recon:.6f} | "
            f"reg={epoch_reg:.6f} | "
            f"monitor={monitor:.6f}"
        )

        save_checkpoint(
            path=last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_metric=best_metric,
            config=config,
            x_mean=x_mean,
            x_std=x_std,
            u_mean=u_mean,
            u_std=u_std,
            dataset_names=train_dataset.names,
        )

        if monitor < best_metric:
            best_metric = monitor
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                config=config,
                x_mean=x_mean,
                x_std=x_std,
                u_mean=u_mean,
                u_std=u_std,
                dataset_names=train_dataset.names,
            )
            print("saved best checkpoint:", best_path)

        if epoch % args.save_every_epoch == 0:
            periodic_path = run_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(
                path=periodic_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                config=config,
                x_mean=x_mean,
                x_std=x_std,
                u_mean=u_mean,
                u_std=u_std,
                dataset_names=train_dataset.names,
            )
            print("saved periodic checkpoint:", periodic_path)

    print("=" * 100)
    print("training finished")
    print("best_metric:", best_metric)
    print("best checkpoint:", best_path)
    print("last checkpoint:", last_path)
    print("log:", log_csv)
    print("=" * 100)

    if args.run_test_oracle:
        print("running test oracle latent fit...")
        test_dataset = AirfRANSDataset(root=args.root, split="test")

        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])

        oracle_csv = run_dir / "test_oracle_latent_fit.csv"

        fit_test_oracle_latent(
            decoder_model=model,
            test_dataset=test_dataset,
            device=device,
            x_mean=x_mean,
            x_std=x_std,
            u_mean=u_mean,
            u_std=u_std,
            latent_dim=args.latent_dim,
            num_samples=args.test_oracle_samples,
            z_steps=args.test_z_steps,
            z_lr=args.test_z_lr,
            z_num_points=args.test_z_num_points,
            latent_reg=args.latent_reg,
            eval_chunk_size=args.eval_chunk_size,
            out_csv=oracle_csv,
        )


if __name__ == "__main__":
    main()