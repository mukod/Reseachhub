import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# === Load the combined dataset ===
df = pd.read_csv("MarioPCGStudy/combined_trajectories.csv")

# === Config ===
FEATURE_COLUMNS = ["jumps", "kills", "right_moves", "left_moves", "coins_collected", "deaths", "powerups"]
MAX_LEN = 50  # Fixed sequence length
STRIDE = 10   # Sliding window step

# === Normalize grouped data ===
def normalize_grouped(df, group_cols, feature_cols):
    scaler = MinMaxScaler()
    sequences = []

    # Group by participant_id and filename (i.e., unique game sessions)
    for _, group in df.groupby(group_cols):
        values = group[feature_cols].values
        if len(values) < 2:  # Skip too-short sessions
            continue
        scaled = scaler.fit_transform(values)
        sequences.append(pd.DataFrame(scaled, columns=feature_cols))

    return sequences

# === Sliding window function ===
def sliding_window_sequences(seq, max_len=50, stride=10):
    windows = []
    for start in range(0, len(seq) - max_len + 1, stride):
        window = seq[start:start+max_len]
        if len(window) == max_len:
            windows.append(window)
    return windows

# === Pad a sequence shorter than max_len (optional) ===
def pad_sequence(seq, max_len, num_features):
    arr = np.zeros((max_len, num_features))
    length = min(len(seq), max_len)
    arr[:length] = seq[:length]
    return arr

# === Normalize and generate windowed trajectories ===
group_cols = ["participant_id", "filename"]
normalized_seqs = normalize_grouped(df, group_cols, FEATURE_COLUMNS)

# Generate sliding windows
all_sequences = []
for seq in tqdm(normalized_seqs, desc="Processing sessions"):
    windows = sliding_window_sequences(seq.values, MAX_LEN, STRIDE)
    for window in windows:
        all_sequences.append(window)

# Convert to NumPy array
mario_data = np.array(all_sequences)

# === Save the final .npy ===
np.save("mario_sequences_augmented.npy", mario_data)
print(f"[✓] Saved mario_sequences_augmented.npy with shape {mario_data.shape}")
