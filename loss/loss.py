import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VAELoss(nn.Module):
    def __init__(self, beta=1.0):
        
        super().__init__()
        self.beta = beta

    def forward(self, x, recon_x, mean, log_var,progress):
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
        kl_loss = kl_loss.mean()

        # Total loss
        total_loss = recon_loss + self.beta* kl_loss

        return total_loss, recon_loss, kl_loss
