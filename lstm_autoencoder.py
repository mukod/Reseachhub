
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        _, (hn, _) = self.encoder(x)
        z = self.latent(hn[-1])
        dec_input = self.decoder_input(z).unsqueeze(1).repeat(1, seq_len, 1)
        output, _ = self.decoder(dec_input)
        return output

def train(model, dataloader, num_epochs=20, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            output = model(x)
            loss = criterion(output, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

def load_data(path, batch_size=64):
    data = np.load(path)
    tensor = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to .npy data file")
    parser.add_argument("--input_dim", type=int, required=True, help="Dimensionality of input features")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = load_data(args.data, args.batch_size)
    model = LSTMAutoencoder(args.input_dim, args.hidden_dim, args.latent_dim).to(device)
    train(model, dataloader, args.epochs)
