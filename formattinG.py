import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tqdm import tqdm

# === Load datasets ===
mario_df = pd.read_csv("MarioPCGStudy/combined_trajectories.csv")
synthetic_df = pd.read_csv("runebound/runebound_depths_synthetic_10000.csv")


# === Feature columns ===
mario_features = ["jumps", "kills", "right_moves", "left_moves", "coins_collected", "deaths", "powerups"]
synthetic_features = ["x", "y", "hp", "powerups", "coins", "enemies_defeated"]

# === Normalization function ===
def normalize_grouped(df, group_col, feature_cols):
    scaler = MinMaxScaler()
    sequences = []
    for _, group in df.groupby(group_col):
        values = group[feature_cols].values
        scaled = scaler.fit_transform(values)
        sequences.append(pd.DataFrame(scaled, columns=feature_cols))
    return sequences

# === Padding function ===
def pad_sequence(seq, max_len, num_features):
    arr = np.zeros((max_len, num_features))
    length = min(len(seq), max_len)
    arr[:length] = seq[:length]
    return arr

# === Config ===
MAX_LEN = 50

# === Normalize and pad ===
mario_sequences = normalize_grouped(mario_df, "participant_id", mario_features)
synthetic_sequences = normalize_grouped(synthetic_df, "episode_id", synthetic_features)

mario_data = np.array([pad_sequence(seq.values, MAX_LEN, len(mario_features)) for seq in mario_sequences])
synthetic_data = np.array([pad_sequence(seq.values, MAX_LEN, len(synthetic_features)) for seq in synthetic_sequences])

# === Encode synthetic labels ===
labels = synthetic_df[["episode_id", "playstyle"]].drop_duplicates().sort_values("episode_id")
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels["playstyle"])

# === Save output ===
np.save("mario_sequences.npy", mario_data)
np.save("synthetic_sequences.npy", synthetic_data)
np.save("synthetic_labels.npy", encoded_labels)
print("[✓] Saved mario_sequences.npy, synthetic_sequences.npy, and synthetic_labels.npy")
