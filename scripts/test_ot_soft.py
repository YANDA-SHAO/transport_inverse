import torch

from src.data.airfrans_dataset import AirfRANSDataset
from src.sensors.sampler import sample_sensors
from src.inverse.nearest import nearest_node
from src.inverse.gaussian_soft import gaussian_soft_predict
from src.inverse.ot_soft import ot_soft_predict


def main():
    torch.manual_seed(0)

    dataset = AirfRANSDataset(
        root="data/raw/airfrans/Dataset",
        split="train",
    )

    sample = dataset[0]
    x = sample["x"]
    u = sample["u"]

    print("sample:", sample["name"])
    print("x:", x.shape)
    print("u:", u.shape)

    m = 512
    noise_std = 0.0
    offset_std = 0.002

    gaussian_k = 512
    tau = 1e-3

    candidate_k_list = [16, 32, 64, 128]
    epsilon_list = [1e-4, 1e-3, 1e-2]

    n_iters = 100
    chunk_size = 64

    obs = sample_sensors(
        x=x,
        u=u,
        m=m,
        noise_std=noise_std,
        offset_std=offset_std,
    )

    y = obs["y"]
    s = obs["s"]
    idx_true = obs["idx_true"]

    print("y:", y.shape)
    print("s:", s.shape)
    print()

    # --------------------------------------------------
    # Nearest
    # --------------------------------------------------
    idx_nearest = nearest_node(
        s=s,
        x=x,
        chunk_size=chunk_size,
    )
    y_hat_nearest = u[idx_nearest]

    nearest_mse = ((y - y_hat_nearest) ** 2).mean()
    nearest_acc = (idx_nearest == idx_true).float().mean()

    print("=" * 80)
    print("Nearest")
    print("=" * 80)
    print("nearest mse:", nearest_mse.item())
    print("nearest corr acc:", nearest_acc.item())

    # --------------------------------------------------
    # Gaussian
    # --------------------------------------------------
    y_hat_gaussian, idx_knn, weights = gaussian_soft_predict(
        s=s,
        x=x,
        u=u,
        k=gaussian_k,
        tau=tau,
        chunk_size=chunk_size,
    )

    gaussian_mse = ((y - y_hat_gaussian) ** 2).mean()
    true_in_knn_acc = (idx_knn == idx_true.unsqueeze(1)).any(dim=1).float().mean()
    weights_max_mean = weights.max(dim=1).values.mean()

    print()
    print("=" * 80)
    print("Gaussian")
    print("=" * 80)
    print("gaussian k:", gaussian_k)
    print("tau:", tau)
    print("gaussian mse:", gaussian_mse.item())
    print("true in kNN acc:", true_in_knn_acc.item())
    print("gaussian weights max mean:", weights_max_mean.item())

    # --------------------------------------------------
    # OT sweep
    # --------------------------------------------------
    print()
    print("=" * 80)
    print("OT sweep")
    print("=" * 80)

    best_result = None

    for candidate_k in candidate_k_list:
        print()
        print("-" * 80)
        print(f"candidate_k = {candidate_k}")
        print("-" * 80)

        for epsilon in epsilon_list:
            y_hat_ot, candidate_idx, P, W = ot_soft_predict(
                s=s,
                x=x,
                u=u,
                candidate_k=candidate_k,
                epsilon=epsilon,
                n_iters=n_iters,
                chunk_size=chunk_size,
            )

            ot_mse = ((y - y_hat_ot) ** 2).mean()

            true_in_candidate_acc = (
                candidate_idx.unsqueeze(0) == idx_true.unsqueeze(1)
            ).any(dim=1).float().mean()

            W_max_mean = W.max(dim=1).values.mean()
            W_row_min = W.sum(dim=1).min()
            W_row_max = W.sum(dim=1).max()

            P_row_min = P.sum(dim=1).min()
            P_row_max = P.sum(dim=1).max()
            P_col_min = P.sum(dim=0).min()
            P_col_max = P.sum(dim=0).max()

            result = {
                "candidate_k": candidate_k,
                "epsilon": epsilon,
                "ot_mse": ot_mse.item(),
                "candidate_size": candidate_idx.shape[0],
                "true_in_candidate_acc": true_in_candidate_acc.item(),
                "W_max_mean": W_max_mean.item(),
                "W_row_min": W_row_min.item(),
                "W_row_max": W_row_max.item(),
                "P_row_min": P_row_min.item(),
                "P_row_max": P_row_max.item(),
                "P_col_min": P_col_min.item(),
                "P_col_max": P_col_max.item(),
            }

            if best_result is None or result["ot_mse"] < best_result["ot_mse"]:
                best_result = result

            print(
                f"epsilon={epsilon:<8} | "
                f"ot_mse={result['ot_mse']:.4f} | "
                f"candidate_size={result['candidate_size']} | "
                f"true_in_candidate={result['true_in_candidate_acc']:.4f} | "
                f"W_max_mean={result['W_max_mean']:.4f} | "
                f"W_row=[{result['W_row_min']:.4f}, {result['W_row_max']:.4f}] | "
                f"P_row=[{result['P_row_min']:.6f}, {result['P_row_max']:.6f}] | "
                f"P_col=[{result['P_col_min']:.6f}, {result['P_col_max']:.6f}]"
            )

    print()
    print("=" * 80)
    print("Best OT result")
    print("=" * 80)
    print(best_result)

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print("nearest mse:", nearest_mse.item())
    print("gaussian mse:", gaussian_mse.item())
    print("best ot mse:", best_result["ot_mse"])


if __name__ == "__main__":
    main()