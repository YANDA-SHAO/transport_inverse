from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = Path("results/feature_gaussian_offset.csv")
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    summary = (
        df.groupby("offset_std")
        .agg(
            nearest_mean=("nearest_mse", "mean"),
            nearest_std=("nearest_mse", "std"),
            gaussian_mean=("gaussian_mse", "mean"),
            gaussian_std=("gaussian_mse", "std"),
            feature_mean=("feature_mse", "mean"),
            feature_std=("feature_mse", "std"),
            nearest_acc_mean=("nearest_acc", "mean"),
            true_knn_mean=("true_in_knn_acc", "mean"),
            gaussian_w_mean=("gaussian_w_max", "mean"),
            feature_w_mean=("feature_w_max", "mean"),
        )
        .reset_index()
    )

    summary_csv = output_dir / "feature_gaussian_offset_summary.csv"
    summary.to_csv(summary_csv, index=False)

    x = summary["offset_std"]

    # --------------------------------------------------
    # Figure 1: MSE vs offset
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))

    plt.errorbar(
        x,
        summary["nearest_mean"],
        yerr=summary["nearest_std"],
        marker="o",
        capsize=4,
        label="Nearest",
    )

    plt.errorbar(
        x,
        summary["gaussian_mean"],
        yerr=summary["gaussian_std"],
        marker="s",
        capsize=4,
        label="Spatial Gaussian",
    )

    plt.errorbar(
        x,
        summary["feature_mean"],
        yerr=summary["feature_std"],
        marker="^",
        capsize=4,
        label="Feature-aware Gaussian",
    )

    plt.xlabel("Offset standard deviation")
    plt.ylabel("Observation reconstruction MSE")
    plt.title("Robustness under correspondence uncertainty")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = output_dir / "offset_vs_mse.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # --------------------------------------------------
    # Figure 2: log-scale MSE
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))

    plt.plot(x, summary["nearest_mean"], marker="o", label="Nearest")
    plt.plot(x, summary["gaussian_mean"], marker="s", label="Spatial Gaussian")
    plt.plot(x, summary["feature_mean"], marker="^", label="Feature-aware Gaussian")

    plt.yscale("log")
    plt.xlabel("Offset standard deviation")
    plt.ylabel("Observation reconstruction MSE (log scale)")
    plt.title("Robustness under correspondence uncertainty")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = output_dir / "offset_vs_mse_log.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # --------------------------------------------------
    # Figure 3: ratio to nearest
    # --------------------------------------------------
    summary["gaussian_ratio"] = summary["gaussian_mean"] / summary["nearest_mean"]
    summary["feature_ratio"] = summary["feature_mean"] / summary["nearest_mean"]

    plt.figure(figsize=(7, 5))

    plt.plot(x, summary["gaussian_ratio"], marker="s", label="Spatial Gaussian / Nearest")
    plt.plot(x, summary["feature_ratio"], marker="^", label="Feature-aware Gaussian / Nearest")
    plt.axhline(1.0, linestyle="--", linewidth=1)

    plt.xlabel("Offset standard deviation")
    plt.ylabel("MSE ratio")
    plt.title("Relative performance compared with nearest")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = output_dir / "offset_vs_ratio.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # --------------------------------------------------
    # Figure 4: candidate coverage
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))

    plt.plot(x, summary["nearest_acc_mean"], marker="o", label="Nearest correspondence accuracy")
    plt.plot(x, summary["true_knn_mean"], marker="s", label="True node in kNN candidates")

    plt.xlabel("Offset standard deviation")
    plt.ylabel("Accuracy")
    plt.title("Recoverability under sensor offset")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = output_dir / "offset_vs_recoverability.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # --------------------------------------------------
    # Figure 5: weight sharpness
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))

    plt.plot(x, summary["gaussian_w_mean"], marker="s", label="Spatial Gaussian")
    plt.plot(x, summary["feature_w_mean"], marker="^", label="Feature-aware Gaussian")

    plt.xlabel("Offset standard deviation")
    plt.ylabel("Mean maximum assignment weight")
    plt.title("Assignment sharpness")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = output_dir / "offset_vs_weight_sharpness.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("Saved summary:", summary_csv)
    print("Saved figures to:", output_dir)
    print()
    print("Generated:")
    print(output_dir / "offset_vs_mse.png")
    print(output_dir / "offset_vs_mse_log.png")
    print(output_dir / "offset_vs_ratio.png")
    print(output_dir / "offset_vs_recoverability.png")
    print(output_dir / "offset_vs_weight_sharpness.png")


if __name__ == "__main__":
    main()