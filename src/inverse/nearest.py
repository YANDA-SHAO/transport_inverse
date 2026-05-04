import torch


@torch.no_grad()
def nearest_node(
    s: torch.Tensor,
    x: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Find nearest mesh node for each sensor coordinate.

    Args:
        s: [m, d] nominal sensor coordinates
        x: [n, d] mesh node coordinates
        chunk_size: number of sensors processed per chunk

    Returns:
        idx: [m] nearest node indices
    """
    if s.ndim != 2:
        raise ValueError(f"s must have shape [m, d], got {s.shape}")
    if x.ndim != 2:
        raise ValueError(f"x must have shape [n, d], got {x.shape}")
    if s.shape[1] != x.shape[1]:
        raise ValueError(
            f"Dimension mismatch: s has dim {s.shape[1]}, x has dim {x.shape[1]}"
        )

    device = s.device
    x = x.to(device)

    idx_all = []

    for start in range(0, s.shape[0], chunk_size):
        end = min(start + chunk_size, s.shape[0])
        s_chunk = s[start:end]

        # [chunk, n]
        dist2 = torch.cdist(s_chunk, x, p=2) ** 2

        idx = torch.argmin(dist2, dim=1)
        idx_all.append(idx)

    return torch.cat(idx_all, dim=0)


@torch.no_grad()
def nearest_prediction(
    s: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    chunk_size: int = 64,
):
    """
    Nearest-node baseline prediction.

    Args:
        s: [m, d] nominal sensor coordinates
        x: [n, d] mesh node coordinates
        u: [n, c] full field

    Returns:
        y_hat: [m, c]
        idx: [m]
    """
    idx = nearest_node(s=s, x=x, chunk_size=chunk_size)
    y_hat = u[idx]
    return y_hat, idx