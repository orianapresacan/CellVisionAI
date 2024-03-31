import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(12, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.fc_mu = nn.Linear(256*4*4, 512)
        self.fc_logvar = nn.Linear(256*4*4, 512)

    def forward(self, x):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv4_out = F.relu(self.conv4(conv3_out))
        flattened = conv4_out.view(-1, 256*4*4)
        mu = self.fc_mu(flattened)
        logvar = self.fc_logvar(flattened)
        return mu, logvar, [conv1_out, conv2_out, conv3_out, conv4_out]

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(512, 256*4*4)
        self.conv_trans1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv_trans2 = nn.ConvTranspose2d(128+128, 64, 4, 2, 1)  # Note the channel concatenation
        self.conv_trans3 = nn.ConvTranspose2d(64+64, 32, 4, 2, 1)    # from skip connections
        self.conv_trans4 = nn.ConvTranspose2d(32+32, 3, 4, 2, 1)

    def forward(self, z, skip_connections):
        z = self.fc(z)
        z = z.view(-1, 256, 4, 4)
        z = F.relu(self.conv_trans1(z))
        z = torch.cat((z, skip_connections[2]), 1)  # Skip connection
        z = F.relu(self.conv_trans2(z))
        z = torch.cat((z, skip_connections[1]), 1)  # Skip connection
        z = F.relu(self.conv_trans3(z))
        z = torch.cat((z, skip_connections[0]), 1)  # Skip connection
        reconstruction = torch.sigmoid(self.conv_trans4(z))
        return reconstruction

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar, skip_connections = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z, skip_connections)
        return reconstruction, mu, logvar
