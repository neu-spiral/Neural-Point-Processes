import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    """usage: autoencoder = Autoencoder(num_kernels_encoder, num_kernels_decoder, input_channel=input_channel).to(device)"""
    def __init__(self, num_kernels_encoder, num_kernels_decoder, input_channel=3, deeper=False):
        super(Autoencoder, self).__init__()
        self.input_channel = input_channel
        self.encoder = self._build_encoder(num_kernels_encoder, deeper)
        self.decoder = self._build_decoder(num_kernels_decoder, num_kernels_encoder[-1], deeper)
        
    def _build_encoder(self, num_kernels, deeper):
        layers = []
        
        for out_channels in num_kernels:
            layers.append(nn.Conv2d(self.input_channel, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            if deeper: # Extra layer that won't change the size
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.input_channel = out_channels  # Update input_channel for the next layer
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self, num_kernels, input_channel, deeper):
        layers = []
        
        for i, out_channels in enumerate(num_kernels):
            if i == 0 and len(num_kernels) > 1:
                layers.append(nn.ConvTranspose2d(input_channel, out_channels, kernel_size=2, stride=2, output_padding=1))
            else: 
                layers.append(nn.ConvTranspose2d(input_channel, out_channels, kernel_size=2, stride=2))
            layers.append(nn.ReLU())
            if deeper: # Extra layer that won't change the size
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU())
            input_channel = out_channels
        
        layers.append(nn.ConvTranspose2d(input_channel, 1, kernel_size=2, stride=2))
        layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        inputHdim, inputWdim = x.size()[2], x.size()[3]
        x = self.encoder(x)
        x = self.decoder(x)
        assert x.size()[1:] == (1, inputHdim, inputWdim), f"Output dimensions do not match the expected size [1, inputHdim, inputWdim], got {x.size()[1:]}"
        return x


class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out

def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding

def _make_te(self, dim_in, dim_out):
    return nn.Sequential(
    nn.Linear(dim_in, dim_out),
    nn.SiLU(),
    nn.Linear(dim_out, dim_out)
    )


class UNet(nn.Module):
    def __init__(self, input_channels=1, shape=28, dims=[10,20,40,80], n_steps=1000, time_emb_dim=100):
        super(UNet, self).__init__()
        self.output_channels = input_channels
        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, input_channels)
        self.b1 = nn.Sequential(
            MyBlock((input_channels, shape, shape), input_channels, dims[0]),
            MyBlock((dims[0], shape, shape), dims[0], dims[0]),
            MyBlock((dims[0], shape, shape), dims[0], dims[0])
        )
        self.down1 = nn.Conv2d(dims[0], dims[0], 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, dims[0])
        self.b2 = nn.Sequential(
            MyBlock((dims[0], shape//2, shape//2), dims[0], dims[1]),
            MyBlock((dims[1], shape//2, shape//2), dims[1], dims[1]),
            MyBlock((dims[1], shape//2, shape//2), dims[1], dims[1])
        )
        self.down2 = nn.Conv2d(dims[1], dims[1], 4, 2, 1)
        
        self.te3 = self._make_te(time_emb_dim, dims[1])
        self.b3 = nn.Sequential(
            MyBlock((dims[1], shape//4, shape//4), dims[1], dims[2]),
            MyBlock((dims[2], shape//4, shape//4), dims[2], dims[2]),
            MyBlock((dims[2], shape//4, shape//4), dims[2], dims[2])
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(dims[2], dims[2], 2, 1),
            nn.SiLU(),
            nn.Conv2d(dims[2], dims[2], 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, dims[2])
        self.b_mid = nn.Sequential(
            MyBlock((dims[2], shape//8, shape//8), dims[2], dims[1]),
            MyBlock((dims[1], shape//8, shape//8), dims[1], dims[1]),
            MyBlock((dims[1], shape//8, shape//8), dims[1], dims[2])
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(dims[2], dims[2], 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(dims[2], dims[2], 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim, dims[3])
        self.b4 = nn.Sequential(
            MyBlock((dims[3],  shape//4, shape//4), dims[3], dims[2]),
            MyBlock((dims[2], shape//4, shape//4), dims[2], dims[1]),
            MyBlock((dims[1], shape//4, shape//4), dims[1], dims[1])
        )

        self.up2 = nn.ConvTranspose2d(dims[1], dims[1], 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, dims[2])
        self.b5 = nn.Sequential(
            MyBlock((dims[2], shape//2, shape//2), dims[2], dims[1]),
            MyBlock((dims[1], shape//2, shape//2), dims[1], dims[0]),
            MyBlock((dims[0], shape//2, shape//2), dims[0], dims[0])
        )

        self.up3 = nn.ConvTranspose2d(dims[0], dims[0], 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, dims[1])
        self.b_out = nn.Sequential(
            MyBlock((dims[1], shape, shape), dims[1], dims[0]),
            MyBlock((dims[0], shape, shape), dims[0], dims[0]),
            MyBlock((dims[0], shape, shape), dims[0], dims[0], normalize=False)
        )

        self.conv_out = nn.Conv2d(dims[0], self.output_channels, 3, 1, 1)

    def forward(self, x, t):
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 28, 28)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, 14, 14)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 7, 7)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)

        out = self.conv_out(out)
        
        # Store feature maps during forward pass
        self.feature_maps = {
            'out_mid': out_mid,
            'out4': out4,
            'out5': out5,
            'out': out
        }

        return out
    
    def feature_extract(self, feature_maps, original_size):
        # Upsample feature maps to the original input size
        original_size = original_size[1:]
        upsampled_out_mid = F.interpolate(feature_maps['out_mid'], size=original_size, mode='bilinear', align_corners=False)
        upsampled_out4 = F.interpolate(feature_maps['out4'], size=original_size, mode='bilinear', align_corners=False)
        upsampled_out5 = F.interpolate(feature_maps['out5'], size=original_size, mode='bilinear', align_corners=False)
        upsampled_out = F.interpolate(feature_maps['out'], size=original_size, mode='bilinear', align_corners=False)

        # Concatenate the upsampled feature maps along the channel dimension
        concatenated_feature_maps = torch.cat([upsampled_out_mid, upsampled_out4, upsampled_out5, upsampled_out], dim=1)

        return concatenated_feature_maps
    

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out))
    

# DDPM class
class DDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw=(1, 28, 28)):
        super(DDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        output = self.network(x, t)
        con_feature_maps = self.network.feature_extract(self.network.feature_maps, self.image_chw)
        return output, con_feature_maps
