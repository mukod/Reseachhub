import random
import pandas as pd
from tqdm import tqdm

# === Configuration ===
PLAYSTYLES = ["Runner", "Magician", "Aggressor", "Explorer", "Balanced"]
ACTIONS = ["move_left", "move_right", "jump", "run", "attack", "use_powerup", "wait"]
MAX_TIMESTEPS = 100
NUM_EPISODES = 10000  # You can increase this if needed

# === Playstyle Action Biases ===
def select_action(playstyle):
    if playstyle == "Runner":
        return random.choices(ACTIONS, weights=[1, 3, 1, 5, 1, 1, 1])[0]
    elif playstyle == "Magician":
        return random.choices(ACTIONS, weights=[1, 1, 1, 1, 1, 6, 1])[0]
    elif playstyle == "Aggressor":
        return random.choices(ACTIONS, weights=[1, 1, 1, 2, 6, 1, 1])[0]
    elif playstyle == "Explorer":
        return random.choices(ACTIONS, weights=[3, 3, 1, 1, 1, 1, 3])[0]
    elif playstyle == "Balanced":
        return random.choices(ACTIONS, weights=[2]*len(ACTIONS))[0]
    return random.choice(ACTIONS)

# === Episode Simulation ===
def simulate_episode(episode_id, playstyle):
    x, y = 0, 0
    hp = 100
    powerups = 0
    coins = 0
    enemies_defeated = 0

    trajectory = []
    for t in range(random.randint(20, MAX_TIMESTEPS)):
        action = select_action(playstyle)
        if action == "move_left":
            x -= 1
        elif action == "move_right":
            x += 1
        elif action == "jump":
            y += 1
        elif action == "run":
            x += 2
        elif action == "attack":
            if random.random() < 0.5:
                enemies_defeated += 1
        elif action == "use_powerup":
            powerups += 1
            hp = min(100, hp + 10)
        elif action == "wait":
            hp = max(0, hp - 1)

        if random.random() < 0.2:
            coins += 1

        trajectory.append({
            "episode_id": episode_id,
            "timestep": t,
            "x": x,
            "y": y,
            "hp": hp,
            "powerups": powerups,
            "coins": coins,
            "enemies_defeated": enemies_defeated,
            "action": action,
            "playstyle": playstyle
        })
    return trajectory

# === Generate Full Dataset ===
all_data = []
for episode_id in tqdm(range(NUM_EPISODES), desc="Generating Synthetic Trajectories"):
    style = random.choice(PLAYSTYLES)
    all_data.extend(simulate_episode(episode_id, style))

# === Save to CSV ===
df = pd.DataFrame(all_data)
df.to_csv("runebound_depths_synthetic_10000.csv", index=False)
print("[✓] Dataset saved to runebound_depths_synthetic_10000.csv")
