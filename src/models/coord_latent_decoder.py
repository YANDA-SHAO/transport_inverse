import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """
    Fourier positional encoding for coordinates.
    """

    def __init__(self, in_dim: int = 2, num_frequencies: int = 6):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies

        freq_bands = 2.0 ** torch.arange(num_frequencies)
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, in_dim]

        Returns:
            encoded: [N, in_dim * (1 + 2 * num_frequencies)]
        """
        outs = [x]

        for freq in self.freq_bands:
            outs.append(torch.sin(freq * torch.pi * x))
            outs.append(torch.cos(freq * torch.pi * x))

        return torch.cat(outs, dim=-1)


class CoordLatentDecoder(nn.Module):
    """
    Coordinate-conditioned latent decoder.

    This is a lightweight GABI-style prior:

        u = D_psi(z, x)

    where:
        x: mesh node coordinates
        z: sample-level latent code
        u: field value at each node
    """

    def __init__(
        self,
        coord_dim: int = 2,
        latent_dim: int = 64,
        out_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 5,
        num_frequencies: int = 6,
        use_fourier: bool = True,
    ):
        super().__init__()

        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.use_fourier = use_fourier

        if use_fourier:
            self.coord_encoder = FourierFeatures(
                in_dim=coord_dim,
                num_frequencies=num_frequencies,
            )
            coord_feat_dim = coord_dim * (1 + 2 * num_frequencies)
        else:
            self.coord_encoder = nn.Identity()
            coord_feat_dim = coord_dim

        input_dim = coord_feat_dim + latent_dim

        layers = []
        dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        """
        Args:
            x: [N, coord_dim]
            z: [latent_dim] or [1, latent_dim]

        Returns:
            u: [N, out_dim]
        """
        if z.ndim == 1:
            z = z.unsqueeze(0)

        if z.ndim != 2 or z.shape[0] != 1:
            raise ValueError(f"z must be [latent_dim] or [1, latent_dim], got {z.shape}")

        x_feat = self.coord_encoder(x)

        z_expand = z.expand(x.shape[0], -1)
        inp = torch.cat([x_feat, z_expand], dim=-1)

        return self.net(inp)


class AutoDecoder(nn.Module):
    """
    Auto-decoder wrapper with one learnable latent code per training sample.
    """

    def __init__(
        self,
        num_samples: int,
        coord_dim: int = 2,
        latent_dim: int = 64,
        out_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 5,
        num_frequencies: int = 6,
        use_fourier: bool = True,
    ):
        super().__init__()

        self.decoder = CoordLatentDecoder(
            coord_dim=coord_dim,
            latent_dim=latent_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_frequencies=num_frequencies,
            use_fourier=use_fourier,
        )

        self.latent_codes = nn.Embedding(num_samples, latent_dim)
        nn.init.normal_(self.latent_codes.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor, sample_id: int):
        """
        Args:
            x: [N, coord_dim]
            sample_id: int

        Returns:
            u_hat: [N, out_dim]
        """
        sid = torch.tensor(sample_id, device=x.device, dtype=torch.long)
        z = self.latent_codes(sid)

        return self.decoder(x, z)

    def decode(self, x: torch.Tensor, z: torch.Tensor):
        return self.decoder(x, z)