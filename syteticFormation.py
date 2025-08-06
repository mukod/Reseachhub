import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tqdm import tqdm

# === Load data ===
df = pd.read_csv("runebound/runebound_depths_synthetic_10000.csv")

# === Config ===
FEATURE_COLS = ["x", "y", "hp", "powerups", "coins", "enemies_defeated"]
LABEL_COL = "playstyle"
GROUP_COL = "episode_id"
WINDOW_SIZE = 50
STRIDE = 10  # overlap step

# === Encode labels ===
labels_df = df[[GROUP_COL, LABEL_COL]].drop_duplicates().sort_values(GROUP_COL)
label_encoder = LabelEncoder()
labels_df["encoded_label"] = label_encoder.fit_transform(labels_df[LABEL_COL])
label_map = dict(zip(labels_df[GROUP_COL], labels_df["encoded_label"]))

# === Normalize + Slice sequences ===
X_sequences = []
y_labels = []

for episode_id, group in tqdm(df.groupby(GROUP_COL)):
    values = group[FEATURE_COLS].values

    # Normalize per episode
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    # Sliding window
    for start in range(0, len(scaled_values) - WINDOW_SIZE + 1, STRIDE):
        window = scaled_values[start:start + WINDOW_SIZE]
        if window.shape[0] == WINDOW_SIZE:
            X_sequences.append(window)
            y_labels.append(label_map[episode_id])

# === Convert to arrays ===
X = np.array(X_sequences)            # Shape: (num_sequences, 50, 6)
y = np.array(y_labels)               # Shape: (num_sequences,)
print("✅ Final shapes:", X.shape, y.shape)

# === Save to .npy files ===
np.save("synthetic_supervised_sequences.npy", X)
np.save("synthetic_supervised_labels.npy", y)
print("🎉 Saved synthetic_supervised_sequences.npy and synthetic_supervised_labels.npy")
