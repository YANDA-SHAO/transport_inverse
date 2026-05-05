import torch
import matplotlib.pyplot as plt


def subsample(x, u, max_points=30000):
    if x.shape[0] <= max_points:
        return x, u
    idx = torch.randperm(x.shape[0])[:max_points]
    return x[idx], u[idx]


def plot_reconstruction(
    x,
    u_gt,
    u_gauss,
    u_latent,
    sensors,
    title="",
    channel=2,   # 2 = pressure（推荐）
    save_path=None,
):
    """
    x: [N,2]
    u_gt / u_gauss / u_latent: [N,C]
    sensors: [m,2]
    """

    # subsample（关键）
    x_s, u_gt_s = subsample(x, u_gt)
    _, u_g_s = subsample(x, u_gauss)
    _, u_l_s = subsample(x, u_latent)

    x_np = x_s.cpu().numpy()
    gt = u_gt_s[:, channel].cpu().numpy()
    g = u_g_s[:, channel].cpu().numpy()
    l = u_l_s[:, channel].cpu().numpy()
    err = abs(l - gt)

    s = sensors.cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # GT
    sc0 = axes[0].scatter(x_np[:, 0], x_np[:, 1], c=gt, s=1, cmap="jet")
    axes[0].scatter(s[:, 0], s[:, 1], c="white", s=5)
    axes[0].set_title("Ground Truth")

    # Gaussian
    sc1 = axes[1].scatter(x_np[:, 0], x_np[:, 1], c=g, s=1, cmap="jet")
    axes[1].set_title("Gaussian")

    # Ours
    sc2 = axes[2].scatter(x_np[:, 0], x_np[:, 1], c=l, s=1, cmap="jet")
    axes[2].set_title("Ours")

    # Error
    sc3 = axes[3].scatter(x_np[:, 0], x_np[:, 1], c=err, s=1, cmap="hot")
    axes[3].set_title("Error (Ours)")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()