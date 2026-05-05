import torch


@torch.no_grad()
def column_capped_transport_predict(
    s: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    k: int = 512,
    tau: float = 1e-3,
    column_cap: float = 0.25,
    n_sinkhorn_iters: int = 20,
    chunk_size: int = 64,
    eps: float = 1e-12,
    return_debug: bool = False,
):
    all_idx = []
    all_dist2 = []

    for start in range(0, s.shape[0], chunk_size):
        end = min(start + chunk_size, s.shape[0])
        s_chunk = s[start:end]

        dist2 = torch.cdist(s_chunk, x) ** 2
        knn_dist2, knn_idx = torch.topk(
            dist2,
            k=k,
            dim=1,
            largest=False,
        )

        all_idx.append(knn_idx)
        all_dist2.append(knn_dist2)

    idx_knn = torch.cat(all_idx, dim=0)
    knn_dist2 = torch.cat(all_dist2, dim=0)

    m = s.shape[0]
    n = x.shape[0]

    logits = -knn_dist2 / tau
    weights = torch.softmax(logits, dim=1)

    flat_idx = idx_knn.reshape(-1)

    def compute_debug(w, prefix):
        col_mass = torch.zeros(n, device=x.device, dtype=w.dtype)
        col_mass.scatter_add_(0, flat_idx, w.reshape(-1))

        return {
            f"{prefix}_weight_max": float(w.max().item()),
            f"{prefix}_weight_min": float(w.min().item()),
            f"{prefix}_weight_mean": float(w.mean().item()),
            f"{prefix}_row_sum_min": float(w.sum(dim=1).min().item()),
            f"{prefix}_row_sum_max": float(w.sum(dim=1).max().item()),
            f"{prefix}_col_mass_max": float(col_mass.max().item()),
            f"{prefix}_col_mass_mean": float(col_mass.mean().item()),
            f"{prefix}_num_capped_nodes": float((col_mass > column_cap).sum().item()),
        }

    debug = {}
    if return_debug:
        debug.update(compute_debug(weights, "before"))

    for _ in range(n_sinkhorn_iters):
        col_mass = torch.zeros(n, device=x.device, dtype=weights.dtype)
        col_mass.scatter_add_(0, flat_idx, weights.reshape(-1))

        scale = torch.clamp(
            column_cap / col_mass.clamp_min(eps),
            max=1.0,
        )

        weights = weights * scale[idx_knn]
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)

    if return_debug:
        debug.update(compute_debug(weights, "after"))

    u_knn = u[idx_knn]
    y_hat = torch.sum(weights.unsqueeze(-1) * u_knn, dim=1)

    if return_debug:
        return y_hat, idx_knn, weights, debug

    return y_hat, idx_knn, weights