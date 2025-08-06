import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load data
data = np.load("mario_sequences_augmented.npy")  # shape (7229, 50, 7)
print(f"Loaded data shape: {data.shape}")
data_tensor = torch.tensor(data, dtype=torch.float32)
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# LSTM Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hn, _) = self.encoder(x)
        latent = self.hidden_to_latent(hn[-1])
        hidden = self.latent_to_hidden(latent).unsqueeze(0)
        c0 = torch.zeros_like(hidden)
        decoded, _ = self.decoder(x, (hidden, c0))
        reconstructed = self.output_layer(decoded)
        return reconstructed, latent

# Initialize and train model
model = LSTMAutoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("\nTraining LSTM Autoencoder...")
model.train()
for epoch in range(10):
    total_loss = 0
    for batch in dataloader:
        x = batch[0]
        reconstructed, _ = model(x)
        loss = criterion(reconstructed, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# Extract latent vectors
print("\nExtracting latent vectors...")
model.eval()
latent_vectors = []
with torch.no_grad():
    for batch in DataLoader(dataset, batch_size=256):
        _, z = model(batch[0])
        latent_vectors.append(z)
latent_vectors = torch.cat(latent_vectors).numpy()
np.save("latent_vectors.npy", latent_vectors)
print(f"Saved latent vectors with shape: {latent_vectors.shape}")

# Clustering
print("\nRunning K-Means and GMM clustering...")
kmeans = KMeans(n_clusters=4, random_state=0).fit(latent_vectors)
gmm = GaussianMixture(n_components=4, random_state=0).fit(latent_vectors)
kmeans_labels = kmeans.labels_
gmm_labels = gmm.predict(latent_vectors)

# Evaluation
sil_kmeans = silhouette_score(latent_vectors, kmeans_labels)
db_kmeans = davies_bouldin_score(latent_vectors, kmeans_labels)
sil_gmm = silhouette_score(latent_vectors, gmm_labels)
db_gmm = davies_bouldin_score(latent_vectors, gmm_labels)

print("\nEvaluation Results:")
print(f"K-Means:     Silhouette = {sil_kmeans:.4f}, Davies–Bouldin = {db_kmeans:.4f}")
print(f"GMM:         Silhouette = {sil_gmm:.4f}, Davies–Bouldin = {db_gmm:.4f}")

# Visualize PCA and t-SNE
pca = PCA(n_components=2).fit_transform(latent_vectors)
tsne = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(latent_vectors)

# Plot PCA
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("PCA of Latent Space (K-Means)")
plt.scatter(pca[:, 0], pca[:, 1], c=kmeans_labels, cmap="viridis", s=10)
plt.xlabel("PC 1")
plt.ylabel("PC 2")

# Plot t-SNE
plt.subplot(1, 2, 2)
plt.title("t-SNE of Latent Space (K-Means)")
plt.scatter(tsne[:, 0], tsne[:, 1], c=kmeans_labels, cmap="viridis", s=10)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")

plt.tight_layout()
plt.savefig("latent_space_visualizations.png")
plt.show()
