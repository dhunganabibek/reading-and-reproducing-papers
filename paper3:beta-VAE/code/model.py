"""The fully connected beta-VAE of Table 1 (2D shapes), and a batched version that trains several copies at once."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BetaVAE(nn.Module):
    def __init__(self, n_latent=10, hidden=1200, n_pixels=64 * 64):
        super().__init__()
        self.n_latent = n_latent
        self.encoder = nn.Sequential(nn.Linear(n_pixels, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 2 * n_latent))
        self.decoder = nn.Sequential(nn.Linear(n_latent, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh(),
                                     nn.Linear(hidden, n_pixels))

    def encode(self, x):
        h = self.encoder(x.flatten(1))
        return h[:, :self.n_latent], h[:, self.n_latent:]   # mu, logvar

    def decode(self, z):
        return self.decoder(z)   # Bernoulli logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z), mu, logvar


def loss_terms(logits, x, mu, logvar):
    """-log p(x|z) summed over pixels, and the KL to N(0, I) per latent; works with or without a model axis."""
    recon = F.binary_cross_entropy_with_logits(logits, x.expand_as(logits), reduction="none").sum(-1)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    return recon, kl


class BatchedLinear(nn.Module):
    """S independent linear layers: (S, B, d_in) -> (S, B, d_out)."""
    def __init__(self, n_models, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_models, d_in, d_out))
        self.bias = nn.Parameter(torch.zeros(n_models, 1, d_out))

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)


class BatchedBetaVAE(nn.Module):
    """S independent BetaVAEs evaluated in one call; the copies share no parameters."""
    def __init__(self, n_models, n_latent=10, hidden=1200, n_pixels=64 * 64):
        super().__init__()
        self.n_models, self.n_latent = n_models, n_latent
        enc = [n_pixels, hidden, hidden, 2 * n_latent]
        dec = [n_latent, hidden, hidden, hidden, n_pixels]
        self.enc = nn.ModuleList(BatchedLinear(n_models, a, b) for a, b in zip(enc[:-1], enc[1:]))
        self.dec = nn.ModuleList(BatchedLinear(n_models, a, b) for a, b in zip(dec[:-1], dec[1:]))

    def layer_pairs(self, vae):
        """Each batched layer with the matching nn.Linear of a plain BetaVAE."""
        linears = [m for m in list(vae.encoder) + list(vae.decoder) if isinstance(m, nn.Linear)]
        return zip(list(self.enc) + list(self.dec), linears)

    @torch.no_grad()
    def set_member(self, s, vae):
        for batched, linear in self.layer_pairs(vae):
            batched.weight[s].copy_(linear.weight.t())
            batched.bias[s, 0].copy_(linear.bias)

    @torch.no_grad()
    def get_member(self, s):
        """Copy s as a plain BetaVAE on the CPU."""
        vae = BetaVAE(self.n_latent, self.enc[0].weight.shape[2], self.enc[0].weight.shape[1])
        for batched, linear in self.layer_pairs(vae):
            linear.weight.copy_(batched.weight[s].t().cpu())
            linear.bias.copy_(batched.bias[s, 0].cpu())
        return vae

    def forward(self, x):
        h = x.flatten(1).expand(self.n_models, -1, -1)   # (S, B, P) without a copy
        for i, layer in enumerate(self.enc):
            h = layer(h)
            if i < len(self.enc) - 1:
                h = F.relu(h)
        mu, logvar = h[..., :self.n_latent], h[..., self.n_latent:]
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        for i, layer in enumerate(self.dec):
            z = layer(z)
            if i < len(self.dec) - 1:
                z = torch.tanh(z)
        return z, mu, logvar
