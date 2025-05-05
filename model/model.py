import torch
import torch.nn as nn
from math import floor

class Reparameterize(nn.Module):

    def forward (self,mean,log_var):

        std = torch.exp(0.5*log_var) # Gaussian
        epsilon = torch.randn_like(std)

        return mean + epsilon*std

class LatentGaussian(nn.Module):

    def __init__(self,latent_size):

        super().__init__()

        self.latent_size = latent_size
        self.mean_layer = nn.LazyLinear(latent_size)
        self.log_var_layer = nn.LazyLinear(latent_size)

    def forward(self,z):

        z = z.view(z.size(0), -1)
        mean = self.mean_layer(z)
        log_var = self.log_var_layer(z)

        return mean, log_var


class Encoder(nn.Module):

    def __init__(self, 
    input_channels,
    latent_size,
    feature_sizes = [64,128,256,512]
    ):

        super().__init__()

        layers = [nn.Conv2d(input_channels,feature_sizes[0],kernel_size=4, stride=2,padding=1,bias=False),
                  nn.BatchNorm2d(feature_sizes[0]),
                  nn.LeakyReLU()]
        
        for i in range(1,len(feature_sizes)):
            layers.append(nn.Conv2d(feature_sizes[i-1],feature_sizes[i],kernel_size=4, stride=2,padding=1,bias=False))
            layers.append(nn.BatchNorm2d(feature_sizes[i]))
            layers.append(nn.LeakyReLU())
        
        layers.append(nn.Conv2d(feature_sizes[-1],latent_size,kernel_size=3, stride=1,padding=1,bias=False))
        layers.append(nn.BatchNorm2d(latent_size))

        self.layers = nn.Sequential(*layers)

        self.gaussian = LatentGaussian(latent_size)
        self.reparameterize = Reparameterize()

    

    def forward(self,x):
        
        x =  self.layers(x)
        mean , log_var = self.gaussian(x)

        z = self.reparameterize(mean,log_var)

        return z , mean, log_var



class Decoder(nn.Module):

    def __init__(self, 
    output_channels,
    latent_size,
    input_shape,
    feature_sizes = [64,128,256,512][::-1],
    ):

        super().__init__()

        self.input_shape = input_shape
        self.feature_sizes = feature_sizes

        self.bottleneck_size = feature_sizes[0] , floor(input_shape[1]/2**(len(feature_sizes))) ,  floor(input_shape[2]/2**(len(feature_sizes))) 

        self.upsample = nn.Linear(latent_size,self.bottleneck_size[0]*self.bottleneck_size[1]*self.bottleneck_size[2])
        
        layers = []
        
        for i in range(len(feature_sizes) - 1):
            layers.append(nn.ConvTranspose2d(feature_sizes[i], feature_sizes[i + 1], kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(feature_sizes[i + 1]))
            layers.append(nn.LeakyReLU())
        
        layers.append(nn.ConvTranspose2d(feature_sizes[-1], output_channels, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, z):

        batch = z.shape[0]
        x = self.upsample(z)
        x = x.view(batch, *self.bottleneck_size)

        x = self.layers(x)

        return x
    

class VAE(nn.Module):

    def __init__(self,
                 input_shape,
                 latent_size,
                 feature_sizes = [64,128,256,512]
                 ):
        
        super().__init__()
        
        self.input_size = input_shape
        self.latent_size = latent_size
        self.feature_sizes = feature_sizes

        self.encoder = Encoder(
            input_channels=input_shape[0],
            latent_size=latent_size,
            feature_sizes= feature_sizes
        )

        self.decoder = Decoder(
            output_channels=input_shape[0],
            feature_sizes= feature_sizes[::-1],
            latent_size=latent_size,
            input_shape=input_shape
        )
    
    def forward(self,x):

        z , mean , log_var = self.encoder(x)
        z = self.decoder(z)

        return z, mean, log_var
    
    def decode(self,z):

        return self.decoder(z)

# # ----- ENCODER TEST -----
# encoder = Encoder(input_channels=3, latent_size=141)
# x = torch.randn(8, 3, 128, 128)

# z  , mean , log_var = encoder(x)

# print("[ENCODER] Input shape:", x.shape)
# print("[ENCODER] Latent z shape:", z.shape)

# # ----- DECODER TEST -----
# decoder = Decoder(
#     output_channels=3,         
#     latent_size=141,           
#     input_shape=(3, 128, 128) 
# )

# recon_x = decoder(z)

# # Print shapes
# print("[DECODER] Decoder latent input shape:", z.shape)
# print("[DECODER] Reconstructed output shape:", recon_x.shape)

# # ----- VAE TEST -----
# vae = VAE(
#     input_shape=(3, 128, 128),
#     latent_size=141,
#     feature_sizes=[64, 128, 256, 512]
# )

# # Dummy input
# x = torch.randn(8, 3, 128, 128)

# # Pass through VAE
# recon_x, mean, log_var = vae(x)

# # Print shapes
# print("[VAE] Input shape:", x.shape)
# print("[VAE] Reconstructed output shape:", recon_x.shape)
# print("[VAE] Mean shape:", mean.shape)
# print("[VAE] Log variance shape:", log_var.shape)