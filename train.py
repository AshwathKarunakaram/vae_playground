import torch
from torch import optim, nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import VAE

batch_size = 128
epochs = 10
latent_dim = 20
lr = 1e-3

transform = transforms.ToTensor()
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 28*28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Model, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
model.train()
for epoch in range(1, epochs+1):
    train_loss = 0
    for batch, _ in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(batch)
        loss = loss_function(recon, batch, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f'Epoch {epoch}, Loss: {train_loss/len(train_loader.dataset):.4f}')

# Save model
torch.save(model.state_dict(), 'vae_mnist.pth')