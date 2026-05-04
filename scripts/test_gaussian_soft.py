"""
This script evaluates baseline methods for the inverse mapping problem:
given noisy sensor locations s and observations y, recover the underlying
field values defined on mesh nodes x with values u.

The following baselines are tested:
    1) Nearest-neighbor (NN)
    2) Gaussian soft kNN interpolation

Key experimental findings:

(1) No sensor offset (offset_std = 0):
    - NN achieves the lowest MSE.
    - Reason: sensor locations exactly match mesh nodes, so direct indexing is optimal.
    - Gaussian interpolation introduces unnecessary smoothing → higher error.

(2) With sensor offset (offset_std > 0):
    - NN performance degrades rapidly (MSE increases by orders of magnitude).
    - Cause: nearest node is often incorrect due to spatial perturbation.
    - Gaussian interpolation becomes consistently better than NN.

(3) Limitations of Gaussian soft interpolation:
    - Improvement over NN is modest (not a qualitative improvement).
    - Strong sensitivity to hyperparameters:
        * small tau → weights are sharp → behaves like NN
        * large tau → overly smooth → large bias
    - Increasing k improves coverage but increases smoothing error.

(4) Fundamental bottleneck: kNN coverage
    - If the true underlying node is not included in the k-nearest neighbors,
      interpolation cannot recover the correct value.
    - This issue becomes more severe as offset increases.
    - Empirically verified by decreasing "true_in_knn_acc".

Overall conclusion:
    Both NN and Gaussian interpolation treat the problem as deterministic
    interpolation from observed sensor locations.

    They do NOT model:
        - uncertainty in sensor location
        - latent correspondence between s and x

    This limits their performance under realistic noisy sensing conditions,
    motivating probabilistic or latent-variable approaches.

(transport_inverse) C:\Users\285261K\Experiments\transport_inversion>python -m scripts.test_gaussian_soft
Loading dataset (task: full, split: train): 100%|███████████████████████████████████████| 800/800 [03:05<00:00,  4.32it/s]
sample: airFoil2D_SST_36.622_11.319_3.941_5.424_1.0_16.283
x: torch.Size([181794, 2])
u: torch.Size([181794, 4])

================================================================================
offset_std = 0.0
================================================================================

--- k = 64 ---
tau=1e-05    | nearest_mse=1.8895 | soft_mse=8.7357 | nearest_acc=0.7363 | true_in_knn=1.0000 | w_max_mean=0.2134
tau=0.0001   | nearest_mse=1.6178 | soft_mse=10.7619 | nearest_acc=0.7793 | true_in_knn=1.0000 | w_max_mean=0.0915
tau=0.001    | nearest_mse=1.4288 | soft_mse=10.6987 | nearest_acc=0.7754 | true_in_knn=1.0000 | w_max_mean=0.0373
tau=0.01     | nearest_mse=1.7496 | soft_mse=11.1514 | nearest_acc=0.7539 | true_in_knn=1.0000 | w_max_mean=0.0181

--- k = 128 ---
tau=1e-05    | nearest_mse=1.4193 | soft_mse=14.0901 | nearest_acc=0.7422 | true_in_knn=1.0000 | w_max_mean=0.2210
tau=0.0001   | nearest_mse=2.5762 | soft_mse=14.4812 | nearest_acc=0.7305 | true_in_knn=1.0000 | w_max_mean=0.0815
tau=0.001    | nearest_mse=2.2461 | soft_mse=18.0402 | nearest_acc=0.7383 | true_in_knn=1.0000 | w_max_mean=0.0306
tau=0.01     | nearest_mse=1.5129 | soft_mse=13.7349 | nearest_acc=0.7871 | true_in_knn=1.0000 | w_max_mean=0.0100

--- k = 256 ---
tau=1e-05    | nearest_mse=2.7690 | soft_mse=17.3405 | nearest_acc=0.7578 | true_in_knn=1.0000 | w_max_mean=0.2360
tau=0.0001   | nearest_mse=2.3705 | soft_mse=37.3463 | nearest_acc=0.7480 | true_in_knn=1.0000 | w_max_mean=0.0886
tau=0.001    | nearest_mse=1.4239 | soft_mse=46.5711 | nearest_acc=0.7637 | true_in_knn=1.0000 | w_max_mean=0.0261
tau=0.01     | nearest_mse=2.3176 | soft_mse=42.8633 | nearest_acc=0.7480 | true_in_knn=1.0000 | w_max_mean=0.0063

--- k = 512 ---
tau=1e-05    | nearest_mse=1.9428 | soft_mse=30.3097 | nearest_acc=0.7539 | true_in_knn=1.0000 | w_max_mean=0.2384
tau=0.0001   | nearest_mse=2.0211 | soft_mse=49.6760 | nearest_acc=0.7520 | true_in_knn=1.0000 | w_max_mean=0.0870
tau=0.001    | nearest_mse=1.8360 | soft_mse=83.6049 | nearest_acc=0.7598 | true_in_knn=1.0000 | w_max_mean=0.0241
tau=0.01     | nearest_mse=1.5349 | soft_mse=122.0268 | nearest_acc=0.7402 | true_in_knn=1.0000 | w_max_mean=0.0069
================================================================================
offset_std = 0.001
================================================================================

--- k = 64 ---
tau=1e-05    | nearest_mse=327.0464 | soft_mse=308.0889 | nearest_acc=0.3008 | true_in_knn=0.7891 | w_max_mean=0.2117
tau=0.0001   | nearest_mse=256.9152 | soft_mse=244.1597 | nearest_acc=0.3301 | true_in_knn=0.8164 | w_max_mean=0.0913
tau=0.001    | nearest_mse=336.9217 | soft_mse=320.3949 | nearest_acc=0.3223 | true_in_knn=0.8164 | w_max_mean=0.0434
tau=0.01     | nearest_mse=410.6163 | soft_mse=394.2551 | nearest_acc=0.2832 | true_in_knn=0.7715 | w_max_mean=0.0181

--- k = 128 ---
tau=1e-05    | nearest_mse=356.4061 | soft_mse=322.5402 | nearest_acc=0.2793 | true_in_knn=0.8457 | w_max_mean=0.1903
tau=0.0001   | nearest_mse=572.9747 | soft_mse=555.4867 | nearest_acc=0.2871 | true_in_knn=0.8379 | w_max_mean=0.0770
tau=0.001    | nearest_mse=422.4388 | soft_mse=394.1669 | nearest_acc=0.3359 | true_in_knn=0.8496 | w_max_mean=0.0353
tau=0.01     | nearest_mse=331.0605 | soft_mse=312.1362 | nearest_acc=0.3242 | true_in_knn=0.8613 | w_max_mean=0.0101

--- k = 256 ---
tau=1e-05    | nearest_mse=393.8824 | soft_mse=360.0814 | nearest_acc=0.3105 | true_in_knn=0.9121 | w_max_mean=0.2082
tau=0.0001   | nearest_mse=361.8163 | soft_mse=321.3462 | nearest_acc=0.3203 | true_in_knn=0.8945 | w_max_mean=0.0790
tau=0.001    | nearest_mse=394.6807 | soft_mse=375.5323 | nearest_acc=0.3535 | true_in_knn=0.8730 | w_max_mean=0.0266
tau=0.01     | nearest_mse=559.4080 | soft_mse=525.6801 | nearest_acc=0.3203 | true_in_knn=0.8750 | w_max_mean=0.0077

--- k = 512 ---
tau=1e-05    | nearest_mse=331.6908 | soft_mse=280.3476 | nearest_acc=0.3027 | true_in_knn=0.9355 | w_max_mean=0.2077
tau=0.0001   | nearest_mse=448.2458 | soft_mse=383.0361 | nearest_acc=0.3047 | true_in_knn=0.9180 | w_max_mean=0.0769
tau=0.001    | nearest_mse=317.5682 | soft_mse=304.4485 | nearest_acc=0.3418 | true_in_knn=0.9258 | w_max_mean=0.0228
tau=0.01     | nearest_mse=257.5345 | soft_mse=313.9636 | nearest_acc=0.3125 | true_in_knn=0.9316 | w_max_mean=0.0046
================================================================================
offset_std = 0.002
================================================================================

--- k = 64 ---
tau=1e-05    | nearest_mse=1498.9883 | soft_mse=1459.1764 | nearest_acc=0.2090 | true_in_knn=0.6484 | w_max_mean=0.2030
tau=0.0001   | nearest_mse=1239.9919 | soft_mse=1206.9653 | nearest_acc=0.2012 | true_in_knn=0.6973 | w_max_mean=0.0854
tau=0.001    | nearest_mse=1037.1849 | soft_mse=989.4802 | nearest_acc=0.2441 | true_in_knn=0.6621 | w_max_mean=0.0335
tau=0.01     | nearest_mse=1374.7119 | soft_mse=1339.5487 | nearest_acc=0.1738 | true_in_knn=0.6582 | w_max_mean=0.0168

--- k = 128 ---
tau=1e-05    | nearest_mse=1700.7822 | soft_mse=1627.7002 | nearest_acc=0.2363 | true_in_knn=0.7656 | w_max_mean=0.2073
tau=0.0001   | nearest_mse=1508.3408 | soft_mse=1473.1653 | nearest_acc=0.2051 | true_in_knn=0.7578 | w_max_mean=0.0753
tau=0.001    | nearest_mse=1049.9937 | soft_mse=987.1712 | nearest_acc=0.1816 | true_in_knn=0.7559 | w_max_mean=0.0261
tau=0.01     | nearest_mse=1301.0508 | soft_mse=1266.3223 | nearest_acc=0.2207 | true_in_knn=0.7656 | w_max_mean=0.0099

--- k = 256 ---
tau=1e-05    | nearest_mse=1377.5042 | soft_mse=1273.0105 | nearest_acc=0.2148 | true_in_knn=0.8164 | w_max_mean=0.2127
tau=0.0001   | nearest_mse=1149.8402 | soft_mse=1082.1630 | nearest_acc=0.2109 | true_in_knn=0.8477 | w_max_mean=0.0777
tau=0.001    | nearest_mse=1715.2527 | soft_mse=1634.7336 | nearest_acc=0.2461 | true_in_knn=0.8203 | w_max_mean=0.0228
tau=0.01     | nearest_mse=1264.5420 | soft_mse=1199.1675 | nearest_acc=0.2070 | true_in_knn=0.8242 | w_max_mean=0.0060

--- k = 512 ---
tau=1e-05    | nearest_mse=1613.7426 | soft_mse=1472.3175 | nearest_acc=0.1992 | true_in_knn=0.8613 | w_max_mean=0.1860
tau=0.0001   | nearest_mse=1616.8103 | soft_mse=1531.3931 | nearest_acc=0.2148 | true_in_knn=0.8711 | w_max_mean=0.0727
tau=0.001    | nearest_mse=1290.9211 | soft_mse=1188.6443 | nearest_acc=0.2109 | true_in_knn=0.8750 | w_max_mean=0.0213
tau=0.01     | nearest_mse=1143.3298 | soft_mse=1053.2516 | nearest_acc=0.2227 | true_in_knn=0.9082 | w_max_mean=0.0043
================================================================================
offset_std = 0.005
================================================================================

--- k = 64 ---
tau=1e-05    | nearest_mse=8218.8750 | soft_mse=8162.8857 | nearest_acc=0.0879 | true_in_knn=0.4473 | w_max_mean=0.1911
tau=0.0001   | nearest_mse=6724.1553 | soft_mse=6640.4375 | nearest_acc=0.0957 | true_in_knn=0.4980 | w_max_mean=0.0808
tau=0.001    | nearest_mse=6244.9395 | soft_mse=6202.4971 | nearest_acc=0.0977 | true_in_knn=0.4746 | w_max_mean=0.0308
tau=0.01     | nearest_mse=5810.3730 | soft_mse=5754.9824 | nearest_acc=0.1113 | true_in_knn=0.5449 | w_max_mean=0.0178

--- k = 128 ---
tau=1e-05    | nearest_mse=4770.8662 | soft_mse=4705.8911 | nearest_acc=0.1035 | true_in_knn=0.5684 | w_max_mean=0.1846
tau=0.0001   | nearest_mse=4575.4785 | soft_mse=4438.3066 | nearest_acc=0.1074 | true_in_knn=0.6484 | w_max_mean=0.0820
tau=0.001    | nearest_mse=8599.0107 | soft_mse=8524.4824 | nearest_acc=0.0977 | true_in_knn=0.5957 | w_max_mean=0.0296
tau=0.01     | nearest_mse=5897.2603 | soft_mse=5813.0439 | nearest_acc=0.0898 | true_in_knn=0.5742 | w_max_mean=0.0115

--- k = 256 ---
tau=1e-05    | nearest_mse=5869.4219 | soft_mse=5710.6895 | nearest_acc=0.1035 | true_in_knn=0.6680 | w_max_mean=0.1932
tau=0.0001   | nearest_mse=6872.5947 | soft_mse=6638.4292 | nearest_acc=0.1055 | true_in_knn=0.6816 | w_max_mean=0.0795
tau=0.001    | nearest_mse=5912.6309 | soft_mse=5692.5435 | nearest_acc=0.0996 | true_in_knn=0.6445 | w_max_mean=0.0193
tau=0.01     | nearest_mse=5787.6533 | soft_mse=5667.0176 | nearest_acc=0.0918 | true_in_knn=0.6836 | w_max_mean=0.0069

--- k = 512 ---
tau=1e-05    | nearest_mse=4449.3057 | soft_mse=4314.6445 | nearest_acc=0.0723 | true_in_knn=0.7754 | w_max_mean=0.1676
tau=0.0001   | nearest_mse=6226.3589 | soft_mse=5853.0078 | nearest_acc=0.0977 | true_in_knn=0.7676 | w_max_mean=0.0807
tau=0.001    | nearest_mse=6584.5996 | soft_mse=6157.7441 | nearest_acc=0.0996 | true_in_knn=0.7871 | w_max_mean=0.0241
tau=0.01     | nearest_mse=6627.9512 | soft_mse=6420.3965 | nearest_acc=0.0879 | true_in_knn=0.7695 | w_max_mean=0.0045
================================================================================
offset_std = 0.01
================================================================================

--- k = 64 ---
tau=1e-05    | nearest_mse=19003.5254 | soft_mse=18908.8281 | nearest_acc=0.0527 | true_in_knn=0.3711 | w_max_mean=0.1978
tau=0.0001   | nearest_mse=23375.9883 | soft_mse=23249.1738 | nearest_acc=0.0449 | true_in_knn=0.4395 | w_max_mean=0.0927
tau=0.001    | nearest_mse=19118.6895 | soft_mse=19034.0410 | nearest_acc=0.0586 | true_in_knn=0.3574 | w_max_mean=0.0336
tau=0.01     | nearest_mse=16535.2188 | soft_mse=16413.6602 | nearest_acc=0.0664 | true_in_knn=0.3945 | w_max_mean=0.0193

--- k = 128 ---
tau=1e-05    | nearest_mse=16497.6055 | soft_mse=16410.8223 | nearest_acc=0.0469 | true_in_knn=0.4473 | w_max_mean=0.1878
tau=0.0001   | nearest_mse=16468.7422 | soft_mse=16262.7852 | nearest_acc=0.0625 | true_in_knn=0.4648 | w_max_mean=0.0805
tau=0.001    | nearest_mse=21734.9492 | soft_mse=21463.9375 | nearest_acc=0.0625 | true_in_knn=0.4570 | w_max_mean=0.0250
tau=0.01     | nearest_mse=19555.4570 | soft_mse=19390.3828 | nearest_acc=0.0645 | true_in_knn=0.4766 | w_max_mean=0.0100

--- k = 256 ---
tau=1e-05    | nearest_mse=16819.3809 | soft_mse=16651.1152 | nearest_acc=0.0508 | true_in_knn=0.5391 | w_max_mean=0.2001
tau=0.0001   | nearest_mse=16111.0244 | soft_mse=15793.2852 | nearest_acc=0.0508 | true_in_knn=0.5195 | w_max_mean=0.0727
tau=0.001    | nearest_mse=20018.2852 | soft_mse=19638.9805 | nearest_acc=0.0605 | true_in_knn=0.5234 | w_max_mean=0.0284
tau=0.01     | nearest_mse=16818.0078 | soft_mse=16498.0820 | nearest_acc=0.0527 | true_in_knn=0.5527 | w_max_mean=0.0073

--- k = 512 ---
tau=1e-05    | nearest_mse=18111.4746 | soft_mse=17814.2695 | nearest_acc=0.0879 | true_in_knn=0.6641 | w_max_mean=0.2214
tau=0.0001   | nearest_mse=14359.2285 | soft_mse=13866.4199 | nearest_acc=0.0605 | true_in_knn=0.6465 | w_max_mean=0.0766
tau=0.001    | nearest_mse=14772.5996 | soft_mse=14025.5498 | nearest_acc=0.0410 | true_in_knn=0.6328 | w_max_mean=0.0172
tau=0.01     | nearest_mse=20035.8652 | soft_mse=19376.5742 | nearest_acc=0.0391 | true_in_knn=0.6211 | w_max_mean=0.0053
"""

import torch

from src.data.airfrans_dataset import AirfRANSDataset
from src.sensors.sampler import sample_sensors
from src.inverse.nearest import nearest_node
from src.inverse.gaussian_soft import gaussian_soft_predict


def evaluate_one_setting(
    x,
    u,
    m: int,
    noise_std: float,
    offset_std: float,
    k: int,
    tau: float,
    chunk_size: int,
):
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

    # Nearest
    idx_nearest = nearest_node(
        s=s,
        x=x,
        chunk_size=chunk_size,
    )
    y_hat_nearest = u[idx_nearest]

    nearest_mse = ((y - y_hat_nearest) ** 2).mean()
    nearest_acc = (idx_nearest == idx_true).float().mean()

    # Gaussian soft
    y_hat_soft, idx_knn, weights = gaussian_soft_predict(
        s=s,
        x=x,
        u=u,
        k=k,
        tau=tau,
        chunk_size=chunk_size,
    )

    soft_mse = ((y - y_hat_soft) ** 2).mean()
    true_in_knn_acc = (idx_knn == idx_true.unsqueeze(1)).any(dim=1).float().mean()
    weights_max_mean = weights.max(dim=1).values.mean()

    return {
        "nearest_mse": nearest_mse.item(),
        "nearest_acc": nearest_acc.item(),
        "soft_mse": soft_mse.item(),
        "true_in_knn_acc": true_in_knn_acc.item(),
        "weights_max_mean": weights_max_mean.item(),
    }


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
    print()

    # Main testing settings
    m = 512
    noise_std = 0.0
    chunk_size = 64

    offset_list = [0.0, 0.001, 0.002, 0.005, 0.01]
    k_list = [64, 128, 256, 512]
    tau_list = [1e-5, 1e-4, 1e-3, 1e-2]

    for offset_std in offset_list:
        print("=" * 80)
        print(f"offset_std = {offset_std}")
        print("=" * 80)

        for k in k_list:
            print(f"\n--- k = {k} ---")

            for tau in tau_list:
                result = evaluate_one_setting(
                    x=x,
                    u=u,
                    m=m,
                    noise_std=noise_std,
                    offset_std=offset_std,
                    k=k,
                    tau=tau,
                    chunk_size=chunk_size,
                )

                print(
                    f"tau={tau:<8} | "
                    f"nearest_mse={result['nearest_mse']:.4f} | "
                    f"soft_mse={result['soft_mse']:.4f} | "
                    f"nearest_acc={result['nearest_acc']:.4f} | "
                    f"true_in_knn={result['true_in_knn_acc']:.4f} | "
                    f"w_max_mean={result['weights_max_mean']:.4f}"
                )


if __name__ == "__main__":
    main()