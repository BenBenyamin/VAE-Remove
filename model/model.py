import torch
import torch.nn as nn

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

        layers = [nn.Conv2d(input_channels,feature_sizes[0],kernel_size=4, stride=2,padding=1,bias=False)]
        layers.append(nn.BatchNorm2d(feature_sizes[0]))
        layers.append(nn.LeakyReLU())
        
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

        return z , mean,log_var



class Decoder(nn.Module):
    pass


encoder = Encoder(input_channels=3, latent_size=128)
x = torch.randn(1, 3, 120, 120)  # batch of 1, 3-channel 64x64 image
out = encoder(x)
print("Input shape:", x.shape)
for o in out:
    print("Output shape:", o.shape)
