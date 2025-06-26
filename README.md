# VAE Playground

An interactive Variational Autoencoder demo that takes you from raw MNIST digits to a live, browser-based “slider” interface for exploring a 20-dimensional latent space. Perfect for anyone who wants to see foundational ML concepts in action without writing a single line of PyTorch.

---

## 🚀 Why It’s Cool

- **Core ML in your browser**  
  No black-box APIs. You build the encoder, decoder, and ELBO loss from scratch—and then expose it as a live demo.

- **Latent space intuition**  
  Twenty sliders correspond to the VAE’s hidden features (e.g. “thickness,” “loopiness,” “tilt”). Drag one and watch a digit morph before your eyes.

- **Full pipeline**  
  From data download to model training to interactive UI and cloud deploy—all in one lightweight project.

---

## 📦 Contents

- **model.py** – VAE definition (encoder, reparameterization trick, decoder)  
- **train.py** – Training loop on MNIST, minimizing ELBO = BCE + KL divergence  
- **app.py**   – Streamlit app with 20 latent-variable sliders and real-time reconstructions  
- **requirements.txt** – Pinning `torch`, `torchvision`, `streamlit`  
- **README.md** – This document

---

## 🔧 Installation

```bash
git clone https://github.com/yourusername/vae-playground.git
cd vae-playground
python3 -m venv venv
source venv/bin/activate       # or `venv\Scripts\activate` on Windows
pip install --upgrade pip
pip install -r requirements.txt
