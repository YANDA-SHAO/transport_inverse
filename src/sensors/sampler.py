import torch


def sample_sensors(
    x: torch.Tensor,
    u: torch.Tensor,
    m: int,
    noise_std: float = 0.0,
    offset_std: float = 0.0,
):
    """
    Sample sparse sensors from full field.

    Args:
        x: [N,2]
        u: [N,4]
        m: number of sensors
    """

    N = x.shape[0]

    # sample indices
    idx = torch.randperm(N)[:m]

    # true positions
    x_true = x[idx]

    # observations
    y = u[idx]

    # add measurement noise
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)

    # nominal sensor coordinates (with offset)
    s = x_true.clone()

    if offset_std > 0:
        s = s + offset_std * torch.randn_like(s)

    return {
        "y": y,
        "s": s,
        "idx_true": idx,
    }