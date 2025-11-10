import pandas as pd
import numpy as np

# Set the path to your CSV
csv_path = "MarioAllPlayerTrajectories.csv"

# List only the feature columns you care about, matching the CSV structure
feature_cols = [
    "little", "large", "fire", "death", "coin", "enemy_defeated",
    "jump", "run", "duck", "left", "right"
]

# Load the CSV
df = pd.read_csv(csv_path)

# Group episodes by PlayerID and Episode (as these columns were set when saving the CSV)
groups = df.groupby(["PlayerID", "Episode"], sort=False)

trajs = []       # List of np.arrays, shape = (timesteps, features) per episode
player_ids = []  # Parallel list of player IDs for each episode
episodes = []    # Parallel list of episode numbers

for (player_id, episode_num), grp in groups:
    arr = grp[feature_cols].to_numpy(dtype=np.float32)
    trajs.append(arr)
    player_ids.append(player_id)
    episodes.append(episode_num)

# Save the trajectories and their metadata for later ML work
np.save("Mario_trajs.npy", np.array(trajs, dtype=object))        # ragged (object) array: 1 per episode
np.save("Mario_player_ids.npy", np.array(player_ids))
np.save("Mario_episode_ids.npy", np.array(episodes))
print(f"[✓] Saved numpy arrays: {len(trajs)} episodes")
print(f"[✓] Mario_trajs.npy shape: {np.array(trajs, dtype=object).shape}")
