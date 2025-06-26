import streamlit as st
import torch
import numpy as np
from model import VAE

# Load model
latent_dim = 20
model = VAE(latent_dim)
model.load_state_dict(torch.load('vae_mnist.pth', map_location='cpu'))
model.eval()

st.title("VAE Playground")

# Sidebar sliders for latent variables
z = np.zeros((latent_dim,))
for i in range(latent_dim):
    z[i] = st.sidebar.slider(f"z[{i}]",   # label
                        -3.0,         # min_value (float)
                         3.0,         # max_value (float)
                         0.0)         # default value (float)

# Generate image
with torch.no_grad():
    z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
    recon = model.decode(z_tensor).view(28,28).numpy()

st.image(recon, width=200)