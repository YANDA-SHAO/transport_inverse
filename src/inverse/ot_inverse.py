import torch
from typing import Callable, Dict, Optional


def build_local_candidates(
    s: torch.Tensor,
    x: torch.Tensor,
    k: int = 512,
    chunk_size: int = 64,
):
    all_idx = []

    for start in range(0, s.shape[0], chunk_size):
        end = min(start + chunk_size, s.shape[0])
        dist2 = torch.cdist(s[start:end], x) ** 2

        _, idx = torch.topk(
            dist2,
            k=min(k, x.shape[0]),
            dim=1,
            largest=False,
        )
        all_idx.append(idx)

    return torch.cat(all_idx, dim=0)  # [m, k]


def observation_aware_weights(
    s: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    idx_knn: torch.Tensor,
    tau: float = 1e-2,
    lambda_feat: float = 10.0,
    normalize_cost: bool = True,
):
    x_knn = x[idx_knn]  # [m, k, d]
    u_knn = u[idx_knn]  # [m, k, c]

    spatial_cost = ((s.unsqueeze(1) - x_knn) ** 2).sum(dim=-1)
    feature_cost = ((y.unsqueeze(1) - u_knn) ** 2).sum(dim=-1)

    if normalize_cost:
        spatial_cost = spatial_cost / spatial_cost.mean(dim=1, keepdim=True).clamp_min(1e-12)
        feature_cost = feature_cost / feature_cost.mean(dim=1, keepdim=True).clamp_min(1e-12)

    cost = spatial_cost + lambda_feat * feature_cost
    weights = torch.softmax(-cost / tau, dim=1)

    return weights, cost


def predict_observation_from_weights(
    u: torch.Tensor,
    idx_knn: torch.Tensor,
    weights: torch.Tensor,
):
    u_knn = u[idx_knn]  # [m, k, c]
    y_hat = torch.sum(weights.unsqueeze(-1) * u_knn, dim=1)
    return y_hat


def optimize_latent_inverse(
    decoder,
    y,
    s,
    x,
    z_init,
    mesh_data=None,
    k=512,
    tau=1e-2,
    lambda_feat=10.0,
    latent_reg=1e-4,
    lr=1e-2,
    n_iters=500,
    chunk_size=64,
    normalize_cost=True,
    verbose=True,
    detach_matching=False,
):
    with torch.no_grad():
        idx_knn = build_local_candidates(
            s=s,
            x=x,
            k=k,
            chunk_size=chunk_size,
        )

    z = z_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)

    history = {
        "loss": [],
        "data_loss": [],
        "reg_loss": [],
        "weights_max_mean": [],
    }

    for it in range(n_iters):
        optimizer.zero_grad()

        u = decoder(z, mesh_data)
        u_for_matching = u.detach() if detach_matching else u

        weights, _ = observation_aware_weights(
            s=s,
            y=y,
            x=x,
            u=u_for_matching,
            idx_knn=idx_knn,
            tau=tau,
            lambda_feat=lambda_feat,
            normalize_cost=normalize_cost,
        )

        y_hat = predict_observation_from_weights(
            u=u,
            idx_knn=idx_knn,
            weights=weights,
        )

        data_loss = ((y_hat - y) ** 2).mean()
        reg_loss = 0.5 * (z ** 2).mean()
        loss = data_loss + latent_reg * reg_loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            history["loss"].append(loss.item())
            history["data_loss"].append(data_loss.item())
            history["reg_loss"].append(reg_loss.item())
            history["weights_max_mean"].append(
                weights.max(dim=1).values.mean().item()
            )

        if verbose and (it % 50 == 0 or it == n_iters - 1):
            print(
                f"iter={it:04d} | "
                f"loss={loss.item():.6f} | "
                f"data={data_loss.item():.6f} | "
                f"reg={reg_loss.item():.6f} | "
                f"wmax={weights.max(dim=1).values.mean().item():.4f}"
            )

    with torch.no_grad():
        z_hat = z.detach()
        u_hat = decoder(z_hat, mesh_data)

        u_for_matching = u_hat.detach() if detach_matching else u_hat

        weights, cost = observation_aware_weights(
            s=s,
            y=y,
            x=x,
            u=u_for_matching,
            idx_knn=idx_knn,
            tau=tau,
            lambda_feat=lambda_feat,
            normalize_cost=normalize_cost,
        )

        y_hat = predict_observation_from_weights(
            u=u_hat,
            idx_knn=idx_knn,
            weights=weights,
        )

    return {
        "z_hat": z_hat,
        "u_hat": u_hat,
        "y_hat": y_hat,
        "idx_knn": idx_knn,
        "weights": weights,
        "cost": cost,
        "history": history,
    }