import torch

from src.inverse.sinkhorn import sinkhorn_log


@torch.no_grad()
def build_candidate_set(
    s: torch.Tensor,
    x: torch.Tensor,
    candidate_k: int = 16,
    chunk_size: int = 64,
):
    """
    Build a compact shared candidate set.

    For each sensor, take only candidate_k nearest mesh nodes,
    then union them into one shared candidate set.
    """
    if s.ndim != 2 or x.ndim != 2:
        raise ValueError(f"s and x must be 2D, got {s.shape}, {x.shape}")

    if s.shape[1] != x.shape[1]:
        raise ValueError(
            f"s and x coordinate dims mismatch: {s.shape[1]} vs {x.shape[1]}"
        )

    candidate_k = min(candidate_k, x.shape[0])

    all_idx = []

    for start in range(0, s.shape[0], chunk_size):
        end = min(start + chunk_size, s.shape[0])

        s_chunk = s[start:end]
        dist2 = torch.cdist(s_chunk, x) ** 2

        _, idx = torch.topk(
            dist2,
            k=candidate_k,
            dim=1,
            largest=False,
        )

        all_idx.append(idx.reshape(-1))

    candidate_idx = torch.unique(torch.cat(all_idx, dim=0), sorted=True)

    return candidate_idx


@torch.no_grad()
def ot_soft_predict(
    s: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    candidate_k: int = 16,
    epsilon: float = 1e-3,
    n_iters: int = 100,
    chunk_size: int = 64,
):
    """
    Compact shared-candidate Sinkhorn OT prediction.

    Args:
        s: [m, 2]
        x: [N, 2]
        u: [N, C]
        candidate_k: number of nearest nodes per sensor used to build shared candidate set
        epsilon: Sinkhorn entropy regularization
        n_iters: Sinkhorn iterations
        chunk_size: chunk size for kNN search

    Returns:
        y_hat: [m, C]
        candidate_idx: [L]
        P: [m, L]
        W: [m, L]
    """
    if s.ndim != 2:
        raise ValueError(f"s must be [m, d], got {s.shape}")

    if x.ndim != 2:
        raise ValueError(f"x must be [N, d], got {x.shape}")

    if u.ndim != 2:
        raise ValueError(f"u must be [N, C], got {u.shape}")

    if x.shape[0] != u.shape[0]:
        raise ValueError(
            f"x and u must have same N, got {x.shape[0]} and {u.shape[0]}"
        )

    candidate_idx = build_candidate_set(
        s=s,
        x=x,
        candidate_k=candidate_k,
        chunk_size=chunk_size,
    )

    x_candidate = x[candidate_idx]
    u_candidate = u[candidate_idx]

    cost = torch.cdist(s, x_candidate) ** 2

    P = sinkhorn_log(
        cost=cost,
        epsilon=epsilon,
        n_iters=n_iters,
    )

    row_mass = P.sum(dim=1, keepdim=True).clamp_min(1e-12)
    W = P / row_mass

    y_hat = W @ u_candidate

    return y_hat, candidate_idx, P, W