import torch


@torch.no_grad()
def sinkhorn_log(
    cost: torch.Tensor,
    epsilon: float = 1e-3,
    n_iters: int = 100,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
):
    """
    Log-domain Sinkhorn solver.

    Solves entropic OT:

        min_P <P, C> + epsilon * sum P_ij (log P_ij - 1)

    subject to:

        P 1 = a
        P^T 1 = b

    Args:
        cost: [m, n] cost matrix
        epsilon: entropy regularization
        n_iters: number of Sinkhorn iterations
        a: [m] source marginal. If None, uniform.
        b: [n] target marginal. If None, uniform.

    Returns:
        P: [m, n] transport plan
    """
    if cost.ndim != 2:
        raise ValueError(f"cost must be 2D [m, n], got {cost.shape}")

    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    m, n = cost.shape
    device = cost.device
    dtype = cost.dtype

    if a is None:
        a = torch.full((m,), 1.0 / m, device=device, dtype=dtype)
    else:
        a = a.to(device=device, dtype=dtype)

    if b is None:
        b = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
    else:
        b = b.to(device=device, dtype=dtype)

    if a.shape != (m,):
        raise ValueError(f"a must have shape [{m}], got {a.shape}")

    if b.shape != (n,):
        raise ValueError(f"b must have shape [{n}], got {b.shape}")

    if torch.any(a <= 0):
        raise ValueError("all entries of a must be positive")

    if torch.any(b <= 0):
        raise ValueError("all entries of b must be positive")

    a = a / a.sum()
    b = b / b.sum()

    log_a = torch.log(a)
    log_b = torch.log(b)

    log_K = -cost / epsilon

    u = torch.zeros_like(a)
    v = torch.zeros_like(b)

    for _ in range(n_iters):
        u = log_a - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)

    log_P = log_K + u.unsqueeze(1) + v.unsqueeze(0)
    P = torch.exp(log_P)

    return P


@torch.no_grad()
def check_transport_plan(
    P: torch.Tensor,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
):
    """
    Utility function for testing Sinkhorn output.

    Args:
        P: [m, n] transport plan
        a: optional source marginal
        b: optional target marginal

    Returns:
        dict of diagnostic errors
    """
    if P.ndim != 2:
        raise ValueError(f"P must be 2D [m, n], got {P.shape}")

    m, n = P.shape
    device = P.device
    dtype = P.dtype

    if a is None:
        a = torch.full((m,), 1.0 / m, device=device, dtype=dtype)
    else:
        a = a.to(device=device, dtype=dtype)
        a = a / a.sum()

    if b is None:
        b = torch.full((n,), 1.0 / n, device=device, dtype=dtype)
    else:
        b = b.to(device=device, dtype=dtype)
        b = b / b.sum()

    row_error = torch.abs(P.sum(dim=1) - a).max()
    col_error = torch.abs(P.sum(dim=0) - b).max()
    total_mass_error = torch.abs(P.sum() - 1.0)

    return {
        "row_error": row_error.item(),
        "col_error": col_error.item(),
        "total_mass_error": total_mass_error.item(),
    }