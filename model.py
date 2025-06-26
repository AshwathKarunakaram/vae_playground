import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        # Encoder
        self.enc_fc1 = nn.Linear(28 * 28, 400)
        self.enc_fc_mu = nn.Linear(400, latent_dim)  # Mean
        self.enc_fc_logvar = nn.Linear(400, latent_dim)  # Log variance
        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, 400)
        self.dec_fc2 = nn.Linear(400, 28 * 28)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, x):
        h1 = self.relu(self.enc_fc1(x))
        mu = self.enc_fc_mu(h1)
        logvar = self.enc_fc_logvar(h1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h2 = self.relu(self.dec_fc1(z))
        return self.sigmoid(self.dec_fc2(h2))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 28 * 28))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar