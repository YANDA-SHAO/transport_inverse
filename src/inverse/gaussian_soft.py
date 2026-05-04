# src/inverse/gaussian_soft.py

import torch


@torch.no_grad()
def gaussian_soft_predict(
    s: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    k: int = 64,
    tau: float = 1e-4,
    chunk_size: int = 64,
):
    """
    Gaussian soft assignment with kNN restriction.

    Args:
        s: [m, 2] sensor locations
        x: [N, 2] mesh node coordinates
        u: [N, C] field values
        k: number of nearest candidate nodes
        tau: temperature / bandwidth
        chunk_size: number of sensors per chunk

    Returns:
        y_hat: [m, C]
        idx_knn: [m, k]
        weights: [m, k]
    """
    y_hats = []
    all_idx = []
    all_weights = []

    for start in range(0, s.shape[0], chunk_size):
        end = min(start + chunk_size, s.shape[0])

        s_chunk = s[start:end]              # [b, 2]
        dist2 = torch.cdist(s_chunk, x) ** 2 # [b, N]

        knn_dist2, knn_idx = torch.topk(
            dist2,
            k=k,
            dim=1,
            largest=False
        )                                  # [b, k]

        logits = -knn_dist2 / tau
        weights = torch.softmax(logits, dim=1)  # [b, k]

        u_knn = u[knn_idx]                 # [b, k, C]
        y_hat = torch.sum(weights.unsqueeze(-1) * u_knn, dim=1)

        y_hats.append(y_hat)
        all_idx.append(knn_idx)
        all_weights.append(weights)

    return (
        torch.cat(y_hats, dim=0),
        torch.cat(all_idx, dim=0),
        torch.cat(all_weights, dim=0),
    )