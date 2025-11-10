import os
import pandas as pd
import numpy as np

feature_names = [
    "little", "large", "fire", "death", "coin", "enemy_defeated",
    "jump", "run", "duck", "left", "right"
]

def mario_event_to_trajectory_full(df):
    traj = []
    little, large, fire = 0, 0, 0
    jump, run, duck, left, right = 0, 0, 0, 0, 0
    for i, row in df.iterrows():
        evt = row['Event']

        # Powerup: mutually exclusive
        if evt == 'LittleStateStart': little, large, fire = 1, 0, 0
        elif evt == 'LargeStateStart': little, large, fire = 0, 1, 0
        elif evt == 'FireStateStart': little, large, fire = 0, 0, 1
        elif evt == 'LittleStateEnd': little = 0
        elif evt == 'LargeStateEnd': large = 0
        elif evt == 'FireStateEnd': fire = 0

        # Movement states
        if evt == 'JumpStart': jump = 1
        elif evt == 'JumpEnd': jump = 0
        if evt == 'RunStateStart': run = 1
        elif evt == 'RunStateEnd': run = 0
        if evt == 'DuckStart': duck = 1
        elif evt == 'DuckEnd': duck = 0
        if evt == 'LeftMoveStart': left = 1
        elif evt == 'LeftMoveEnd': left = 0
        if evt == 'RightMoveStart': right = 1
        elif evt == 'RightMoveEnd': right = 0

        # Binary event features (set = 1 only for that row)
        death = int('Death' in evt or 'DieBy' in evt)
        coin = int('Coin' in evt)
        enemy_defeated = int('Kill' in evt)

        traj.append([little, large, fire, death, coin, enemy_defeated,
                     jump, run, duck, left, right])
    return np.array(traj)

root_dir = "C:/Users/mukos/Desktop/MarioPCGStudy (1)/MarioPCGStudy/AnonymizedDirectory"
all_traj = []

for player_folder in os.listdir(root_dir):
    player_path = os.path.join(root_dir, player_folder)
    if not os.path.isdir(player_path):
        continue
    episode_count = 0
    for filename in sorted(os.listdir(player_path)):
        if filename.endswith(".csv"):
            filepath = os.path.join(player_path, filename)
            try:
                df = pd.read_csv(filepath)
                if 'Event' in df.columns:
                    traj = mario_event_to_trajectory_full(df)
                    traj_df = pd.DataFrame(traj, columns=feature_names)
                    traj_df['Time'] = df['Time'].values
                    traj_df['PlayerID'] = player_folder
                    episode_count += 1
                    traj_df['Episode'] = episode_count
                    all_traj.append(traj_df)
            except Exception as e:
                print(f"❌ Failed to process {filepath}: {e}")

full_df = pd.concat(all_traj, ignore_index=True)
full_df.to_csv("MarioAllPlayerTrajectories.csv", index=False)
print("✅ Saved MarioAllPlayerTrajectories.csv")
