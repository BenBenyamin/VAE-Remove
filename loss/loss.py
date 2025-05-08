import torch
import torch.nn as nn
import torch.nn.functional as F

class VAELoss(nn.Module):
    """
    Loss function for Variational Autoencoder (VAE).

    Combines Mean Squared Error (MSE) reconstruction loss with
    KL divergence loss to regularize the latent space.

    Args:
        beta (float): Scaling factor for the KL divergence term (default: 1.0).
                      Setting beta > 1 encourages disentangled representations.
    """
    def __init__(self, beta=1.0):
        """
        Initializes the VAE loss module.

        Args:
            beta (float): Weight for KL divergence loss component.
        """
        super().__init__()
        self.beta = beta

    def forward(self, x, recon_x, mean, log_var):
        """
        Computes the VAE loss.

        Args:
            x (Tensor): Original input image.
            recon_x (Tensor): Reconstructed image from the decoder.
            mean (Tensor): Mean of the latent Gaussian distribution.
            log_var (Tensor): Log variance of the latent Gaussian distribution.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: 
                - total_loss: Weighted sum of reconstruction and KL loss
                - recon_loss: Mean squared error reconstruction loss
                - kl_loss: KL divergence regularization term
        """
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
        kl_loss = kl_loss.mean()

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss
