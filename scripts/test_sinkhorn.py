import torch

from src.inverse.sinkhorn import sinkhorn_log, check_transport_plan


def main():
    torch.manual_seed(0)

    m = 5
    n = 8

    cost = torch.rand(m, n)

    P = sinkhorn_log(
        cost=cost,
        epsilon=1e-2,
        n_iters=200,
    )

    print("P shape:", P.shape)
    print("P sum:", P.sum().item())
    print("P min:", P.min().item())
    print("P max:", P.max().item())

    diag = check_transport_plan(P)

    print("row_error:", diag["row_error"])
    print("col_error:", diag["col_error"])
    print("total_mass_error:", diag["total_mass_error"])

    print("row sums:", P.sum(dim=1))
    print("col sums:", P.sum(dim=0))


if __name__ == "__main__":
    main()