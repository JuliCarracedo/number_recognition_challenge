# app.py

import streamlit as st
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

device = 'cpu'
latent_dim = 100
num_classes = 10

# Define same Generator as during training
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 28*28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat((z, self.label_emb(labels)), -1)
        return self.model(x).view(-1, 1, 28, 28)

@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("generator.pt", map_location=device))
    model.eval()
    return model

def generate_images(model, digit, n=5):
    z = torch.randn(n, latent_dim)
    labels = torch.tensor([digit] * n)
    with torch.no_grad():
        images = model(z, labels).squeeze().numpy()
    return images

st.title("🧠 Handwritten Digit Generator")
digit = st.selectbox("Select a digit (0–9):", list(range(10)))

if st.button("Generate"):
    model = load_model()
    images = generate_images(model, digit)
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].axis("off")
    st.pyplot(fig)
