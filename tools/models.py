import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """usage: autoencoder = Autoencoder(num_kernels_encoder, num_kernels_decoder, input_channel=input_channel).to(device)"""
    def __init__(self, num_kernels_encoder, num_kernels_decoder, input_channel=3):
        super(Autoencoder, self).__init__()
        self.input_channel = input_channel
        self.encoder = self._build_encoder(num_kernels_encoder)
        self.decoder = self._build_decoder(num_kernels_decoder, num_kernels_encoder[-1])
        
    def _build_encoder(self, num_kernels):
        layers = []
        
        for out_channels in num_kernels:
            layers.append(nn.Conv2d(self.input_channel, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.input_channel = out_channels  # Update input_channel for the next layer
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self, num_kernels, input_channel):
        layers = []
        
        for out_channels in num_kernels:
            layers.append(nn.ConvTranspose2d(input_channel, out_channels, kernel_size=2, stride=2))
            layers.append(nn.ReLU())
            input_channel = out_channels
        
        layers.append(nn.ConvTranspose2d(input_channel, 1, kernel_size=2, stride=2))
        layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
