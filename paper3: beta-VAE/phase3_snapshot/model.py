"""beta-VAE for the 2D shapes dataset, Table 1 of Higgins et al. (2017):
encoder FC 1200, 1200 (ReLU) -> 10 Gaussian latents; decoder FC 1200, 1200, 1200, 4096 (Tanh),
Bernoulli likelihood on the 4096 pixels.
"""
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
        mu, logvar = h[:, :self.n_latent], h[:, self.n_latent:]
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)                  # Bernoulli logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)      # reparameterisation
        return self.decode(z), mu, logvar


def loss_terms(logits, x, mu, logvar):
    """Per-sample negative reconstruction log-likelihood (sum over pixels) and KL(q(z|x) || N(0,I))
    per latent dimension (shape (B, n_latent)).  L_beta = -recon - beta * KL.sum(1)."""
    recon = F.binary_cross_entropy_with_logits(logits, x.flatten(1), reduction="none").sum(1)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    return recon, kl
