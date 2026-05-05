import torch
import torch.nn.functional as F


@torch.no_grad()
def decode_full_in_chunks(
    decoder_fn,
    z,
    mesh_data,
    chunk_size: int = 65536,
):
    """
    Decode full field in chunks to avoid GPU memory spikes.
    """
    x_norm = mesh_data["x_norm"]
    outs = []

    for start in range(0, x_norm.shape[0], chunk_size):
        end = min(start + chunk_size, x_norm.shape[0])
        sub_mesh = {"x_norm": x_norm[start:end]}
        outs.append(decoder_fn(z, sub_mesh))

    return torch.cat(outs, dim=0)


@torch.no_grad()
def update_latent_conditioned_weights(
    s,
    x_norm,
    y,
    u_hat_norm,
    k: int = 512,
    spatial_tau: float = 1e-3,
    value_tau: float = 1e-2,
    alpha_value: float = 0.05,
    chunk_size: int = 64,
):
    all_idx = []
    all_weights = []

    for start in range(0, s.shape[0], chunk_size):
        end = min(start + chunk_size, s.shape[0])

        s_chunk = s[start:end]
        y_chunk = y[start:end]

        dist2 = torch.cdist(s_chunk, x_norm) ** 2

        knn_dist2, knn_idx = torch.topk(
            dist2,
            k=k,
            dim=1,
            largest=False,
        )

        u_knn = u_hat_norm[knn_idx]
        value_dist2 = ((u_knn - y_chunk.unsqueeze(1)) ** 2).mean(dim=-1)

        logits = (
            -knn_dist2 / spatial_tau
            -alpha_value * value_dist2 / value_tau
        )

        weights = torch.softmax(logits, dim=1)

        all_idx.append(knn_idx)
        all_weights.append(weights)

    return torch.cat(all_idx, dim=0), torch.cat(all_weights, dim=0)


def fit_latent_conditioned_transport_inverse(
    decoder_fn,
    mesh_data,
    y,
    s,
    x_norm,
    idx_init,
    weights_init,
    u_norm,
    u,
    u_mean,
    u_std,
    latent_dim,
    device,
    lr: float = 1e-2,
    outer_iters: int = 3,
    inner_iters: int = 200,
    latent_reg: float = 1e-4,
    k: int = 512,
    spatial_tau: float = 1e-3,
    value_tau: float = 1e-2,
    alpha_value: float = 0.1,
    chunk_size: int = 64,
    full_decode_chunk_size: int = 65536,
    return_debug: bool = False,
):
    """
    Latent-conditioned transport inverse.

    This implements alternating inference:

        1. Fix correspondence P, optimize latent z.
        2. Decode current field u_hat = D(z, x).
        3. Update P using geometry and current decoded field.
        4. Repeat.

    This is Level 3 in the method hierarchy:

        Gaussian:
            P = P(s, x)

        Column-capped transport:
            P = P(s, x) with geometric mass constraint

        Latent-conditioned transport:
            P = P(s, x, D(z, x))

    Args:
        decoder_fn:
            Function decoder_fn(z, mesh_data) -> u_hat_norm.
        mesh_data:
            Dict containing {"x_norm": [N, 2]}.
        y:
            Observed normalized sensor values, [m, C].
        s:
            Sensor locations in normalized coordinate space, [m, 2].
        x_norm:
            Full normalized mesh coordinates, [N, 2].
        idx_init:
            Initial kNN indices, usually from Gaussian, [m, k].
        weights_init:
            Initial soft weights, usually Gaussian, [m, k].
        u_norm:
            Full normalized ground truth field, used only for final metrics.
        u:
            Full original-scale ground truth field, used only for final metrics.
        u_mean, u_std:
            Normalization statistics.
        latent_dim:
            Dimension of latent code z.
        device:
            torch device.
        lr:
            Learning rate for z optimization.
        outer_iters:
            Number of alternating updates.
        inner_iters:
            Number of z optimization steps per outer iteration.
        latent_reg:
            L2 regularization on z.
        k:
            Number of local candidates for P update.
        spatial_tau:
            Geometry temperature.
        value_tau:
            Value consistency temperature.
        alpha_value:
            Strength of decoded-field consistency.
            Keep small, e.g. 0.05 to 0.2.
        chunk_size:
            Sensor chunk size for correspondence update.
        full_decode_chunk_size:
            Chunk size for full-field decoding.
        return_debug:
            If True, returns selected diagnostic values.

    Returns:
        metrics dict containing rel_l2, mse, obs_mse, and optional debug.
    """
    z = torch.zeros(latent_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=lr)

    idx_knn = idx_init.detach()
    weights = weights_init.detach()

    last_obs_mse = None
    debug_rows = []

    for outer in range(outer_iters):
        # --------------------------------------------------
        # Step A: optimize z with fixed correspondence P
        # --------------------------------------------------
        m, kk = idx_knn.shape
        flat_idx = idx_knn.reshape(-1)

        unique_idx, inverse_flat = torch.unique(
            flat_idx,
            sorted=True,
            return_inverse=True,
        )
        inverse_knn = inverse_flat.reshape(m, kk)

        mesh_obs = {"x_norm": x_norm[unique_idx]}

        for _ in range(inner_iters):
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

            last_obs_mse = F.mse_loss(y_hat, y).item()

            if return_debug:
                debug_rows.append(
                    {
                        "outer": outer,
                        "obs_mse": last_obs_mse,
                        "weight_max": weights.max().item(),
                        "weight_mean": weights.mean().item(),
                    }
                )

        # --------------------------------------------------
        # Step B: update P using current decoded field
        # --------------------------------------------------
        if outer < outer_iters - 1:
            with torch.no_grad():
                u_hat_norm_full = decode_full_in_chunks(
                    decoder_fn=decoder_fn,
                    z=z.detach(),
                    mesh_data=mesh_data,
                    chunk_size=full_decode_chunk_size,
                )

                idx_knn, weights = update_latent_conditioned_weights(
                    s=s,
                    x_norm=x_norm,
                    y=y,
                    u_hat_norm=u_hat_norm_full,
                    k=k,
                    spatial_tau=spatial_tau,
                    value_tau=value_tau,
                    alpha_value=alpha_value,
                    chunk_size=chunk_size,
                )

                idx_knn = idx_knn.detach()
                weights = weights.detach()

    # --------------------------------------------------
    # Final full-field metrics
    # --------------------------------------------------
    with torch.no_grad():
        u_hat_norm = decode_full_in_chunks(
            decoder_fn=decoder_fn,
            z=z.detach(),
            mesh_data=mesh_data,
            chunk_size=full_decode_chunk_size,
        )

        u_hat = u_hat_norm * u_std + u_mean

        rel_l2_norm = (
            torch.linalg.norm((u_hat_norm - u_norm).reshape(-1))
            / torch.linalg.norm(u_norm.reshape(-1)).clamp_min(1e-12)
        ).item()

        rel_l2 = (
            torch.linalg.norm((u_hat - u).reshape(-1))
            / torch.linalg.norm(u.reshape(-1)).clamp_min(1e-12)
        ).item()

        mse_norm = F.mse_loss(u_hat_norm, u_norm).item()
        mse = F.mse_loss(u_hat, u).item()
        mae = torch.mean(torch.abs(u_hat - u)).item()

    metrics = {
        "rel_l2_norm": rel_l2_norm,
        "rel_l2": rel_l2,
        "mse_norm": mse_norm,
        "mse": mse,
        "obs_mse": last_obs_mse,
        "outer_iters": outer_iters,
        "inner_iters": inner_iters,
        "alpha_value": alpha_value,
        "value_tau": value_tau,
        "mae": mae,
    }

    if return_debug:
        metrics["debug_rows"] = debug_rows

    return metrics, z.detach()