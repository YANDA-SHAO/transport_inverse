import torch


@torch.no_grad()
def feature_gaussian_soft_predict(
    s: torch.Tensor,
    y: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    k: int = 512,
    tau: float = 1e-3,
    lambda_feat: float = 1.0,
    chunk_size: int = 64,
    normalize_cost: bool = True,
):
    """
    Feature-aware Gaussian soft assignment.

    Cost:
        C_ij = spatial_cost_ij + lambda_feat * feature_cost_ij

    where:
        spatial_cost_ij = ||s_i - x_j||^2
        feature_cost_ij = ||y_i - u_j||^2

    Args:
        s: [m, 2] sensor locations
        y: [m, C] sensor observations
        x: [N, 2] mesh node coordinates
        u: [N, C] mesh field values
        k: number of nearest spatial candidates
        tau: softmax temperature
        lambda_feat: weight of feature cost
        chunk_size: number of sensors per chunk
        normalize_cost: normalize spatial and feature costs per sensor

    Returns:
        y_hat: [m, C]
        idx_knn: [m, k]
        weights: [m, k]
    """
    if s.ndim != 2:
        raise ValueError(f"s must be [m, d], got {s.shape}")

    if y.ndim != 2:
        raise ValueError(f"y must be [m, C], got {y.shape}")

    if x.ndim != 2:
        raise ValueError(f"x must be [N, d], got {x.shape}")

    if u.ndim != 2:
        raise ValueError(f"u must be [N, C], got {u.shape}")

    if s.shape[0] != y.shape[0]:
        raise ValueError(f"s and y must have same m, got {s.shape[0]} and {y.shape[0]}")

    if x.shape[0] != u.shape[0]:
        raise ValueError(f"x and u must have same N, got {x.shape[0]} and {u.shape[0]}")

    if s.shape[1] != x.shape[1]:
        raise ValueError(f"s and x coordinate dims mismatch: {s.shape[1]} vs {x.shape[1]}")

    if y.shape[1] != u.shape[1]:
        raise ValueError(f"y and u feature dims mismatch: {y.shape[1]} vs {u.shape[1]}")

    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    k = min(k, x.shape[0])

    y_hats = []
    all_idx = []
    all_weights = []

    for start in range(0, s.shape[0], chunk_size):
        end = min(start + chunk_size, s.shape[0])

        s_chunk = s[start:end]  # [b, d]
        y_chunk = y[start:end]  # [b, C]

        # spatial kNN first
        spatial_dist2_full = torch.cdist(s_chunk, x) ** 2  # [b, N]

        spatial_dist2, idx_knn = torch.topk(
            spatial_dist2_full,
            k=k,
            dim=1,
            largest=False,
        )  # [b, k]

        x_knn = x[idx_knn]  # [b, k, d]
        u_knn = u[idx_knn]  # [b, k, C]

        # feature distance only within spatial candidates
        feature_dist2 = ((y_chunk.unsqueeze(1) - u_knn) ** 2).sum(dim=-1)  # [b, k]

        if normalize_cost:
            spatial_scale = spatial_dist2.mean(dim=1, keepdim=True).clamp_min(1e-12)
            feature_scale = feature_dist2.mean(dim=1, keepdim=True).clamp_min(1e-12)

            spatial_cost = spatial_dist2 / spatial_scale
            feature_cost = feature_dist2 / feature_scale
        else:
            spatial_cost = spatial_dist2
            feature_cost = feature_dist2

        cost = spatial_cost + lambda_feat * feature_cost

        logits = -cost / tau
        weights = torch.softmax(logits, dim=1)

        y_hat = torch.sum(weights.unsqueeze(-1) * u_knn, dim=1)

        y_hats.append(y_hat)
        all_idx.append(idx_knn)
        all_weights.append(weights)

    return (
        torch.cat(y_hats, dim=0),
        torch.cat(all_idx, dim=0),
        torch.cat(all_weights, dim=0),
    )