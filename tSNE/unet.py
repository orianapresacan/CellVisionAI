import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.batchnorm(self.conv1(x)))
        x = self.relu(self.batchnorm(self.conv2(x)))
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Downscaling
        self.down_conv1 = ConvBlock(in_channels, 64)
        self.down_conv2 = ConvBlock(64, 128)
        self.down_conv3 = ConvBlock(128, 256)
        self.down_conv4 = ConvBlock(256, 512)
        
        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Upscaling
        self.up_conv3 = UpConv(512, 256)
        self.up_conv2 = UpConv(512, 128)
        self.up_conv1 = UpConv(256, 64) 

        # Final Convolution
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        # Downscale
        x1 = self.down_conv1(x)
        x2 = self.pool(x1)
        x2 = self.down_conv2(x2)
        x3 = self.pool(x2)
        x3 = self.down_conv3(x3)
        x4 = self.pool(x3)
        x4 = self.down_conv4(x4)

        # Upscale and concatenate
        x = self.up_conv3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv1(x)
        x = torch.cat([x, x1], dim=1)

        # Final convolution
        x = self.final_conv(x)
        return x


class UNet_mod(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=4, initial_features=64):
        super(UNet_mod, self).__init__()

        # Dynamic creation of downsampling and upsampling layers
        self.down_convs = []
        self.up_convs = []

        # Downscaling
        features = initial_features
        for _ in range(num_blocks):
            down_conv = ConvBlock(in_channels, features)
            self.down_convs.append(down_conv)
            in_channels = features
            features *= 2
        self.down_convs = nn.ModuleList(self.down_convs)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Upscaling
        features = in_channels
        for _ in range(num_blocks - 1):
            features //= 2  # Halving the number of features for upconv
            up_conv = UpConv(features * 2, features)
            self.up_convs.append(up_conv)
        self.up_convs = nn.ModuleList(self.up_convs)

        # Final Convolution
        self.final_conv = nn.Conv2d(initial_features * 2, out_channels, kernel_size=1)

    def forward(self, x):
        # Downscaling
        skip_connections = []
        for down_conv in self.down_convs:
            x = down_conv(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Upscaling and concatenating with skip connections
        for i, up_conv in enumerate(self.up_convs):
            x = up_conv(x)
            skip_connection = skip_connections[-(i + 2)]

            # Cropping if necessary
            if x.size() != skip_connection.size():
                diffY = skip_connection.size()[2] - x.size()[2]
                diffX = skip_connection.size()[3] - x.size()[3]
                skip_connection = skip_connection[:, :, diffY // 2: skip_connection.size()[2] - diffY // 2, diffX // 2: skip_connection.size()[3] - diffX // 2]

            x = torch.cat([x, skip_connection], dim=1)

        x = self.final_conv(x)
        return x


