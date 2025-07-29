import os
import pandas as pd
from collections import Counter

# Define your dataset directory
root_dir = "C:/Users/mukos/Desktop/MarioPCGStudy (1)/MarioPCGStudy/AnonymizedDirectory"

# Define relevant events for each category
EVENT_CATEGORIES = {
    "jumps": ["JumpStart"],
    "right_moves": ["RightMoveStart"],
    "left_moves": ["LeftMoveStart"],
    "kills": [
        "StompKillGoomba", "ShellKillGoomba", "FireKillGoomba",
        "StompKillGreenKoopa", "FireKillGreenKoopa", "StompKillBulletBill"
    ],
    "powerups": ["BlockPowerDestroy", "FireStateStart", "LargeStateStart"],
    "coins_collected": ["BlockCoinDestroy", "CollectCoin"],
    "deaths": ["DeathByGap", "DieByGoomba", "DieByGreenKoopa", "LostLevel"],
}

# Build an event vocabulary (for encoding the last event)
event_vocab = set()
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith("InfoTimeline.csv"):
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)
            if 'Event' in df.columns:
                event_vocab.update(df['Event'].unique())

event_to_index = {event: idx for idx, event in enumerate(sorted(event_vocab))}

# Function to extract a single trajectory vector
def extract_trajectory(csv_path):
    df = pd.read_csv(csv_path)
    if 'Event' not in df.columns:
        return None
    
    event_counts = Counter(df['Event'])
    
    trajectory = {
        "jumps": sum(event_counts.get(e, 0) for e in EVENT_CATEGORIES["jumps"]),
        "right_moves": sum(event_counts.get(e, 0) for e in EVENT_CATEGORIES["right_moves"]),
        "left_moves": sum(event_counts.get(e, 0) for e in EVENT_CATEGORIES["left_moves"]),
        "kills": sum(event_counts.get(e, 0) for e in EVENT_CATEGORIES["kills"]),
        "powerups": sum(event_counts.get(e, 0) for e in EVENT_CATEGORIES["powerups"]),
        "coins_collected": sum(event_counts.get(e, 0) for e in EVENT_CATEGORIES["coins_collected"]),
        "deaths": sum(event_counts.get(e, 0) for e in EVENT_CATEGORIES["deaths"]),
        "event_code": event_to_index.get(df['Event'].iloc[-1], -1)  # last event encoded, -1 if unknown
    }
    
    return trajectory

# Loop through all files to extract trajectories
trajectories = []
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith("InfoTimeline.csv"):
            file_path = os.path.join(root, file)
            try:
                traj = extract_trajectory(file_path)
                if traj:
                    traj['filename'] = file
                    trajectories.append(traj)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

# Convert to DataFrame for inspection and saving
trajectories_df = pd.DataFrame(trajectories)

# Reorder columns for readability
trajectories_df = trajectories_df[[
    'filename', 'jumps', 'right_moves', 'left_moves',
    'kills', 'powerups', 'coins_collected', 'deaths', 'event_code'
]]

# Show first few rows as verification
print(trajectories_df.head())

# Optional: Save for ML usage
trajectories_df.to_csv("MarioPCG_trajectories_summary.csv", index=False)
