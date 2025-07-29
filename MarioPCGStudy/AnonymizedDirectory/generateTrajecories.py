import os
import zipfile
import pandas as pd
from tqdm import tqdm

# --- 1. UNZIP ---
def unzip_data(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[✓] Extracted to {extract_to}")

# --- EVENT MAPPER FOR ALL UNIQUE EVENTS ---

def map_event_to_code(event):
    mapping = {
        "BlockCoinDestroy": 1,
        "BlockCoinDestroyBulletBill": 2,
        "BlockPowerDestroy": 3,
        "BlockPowerDestroyBulletBill": 4,
        "CollectCoin": 5,
        "CollectCoinBulletBill": 6,
        "DuckEnd": 7,
        "DuckStart": 8,
        "FireKillGoomba": 9,
        "FireKillGreenKoopa": 10,
        "FireStateEnd": 11,
        "FireStateStart": 12,
        "JumpEnd": 13,
        "JumpStart": 14,
        "LargeStateEnd": 15,
        "LargeStateStart": 16,
        "LeftMoveEnd": 17,
        "LeftMoveStart": 18,
        "LittleStateEnd": 19,
        "LittleStateStart": 20,
        "RightMoveEnd": 21,
        "RightMoveStart": 22,
        "RunStateEnd": 23,
        "RunStateStart": 24,
        "ShellKillBulletBill": 25,
        "ShellKillGoomba": 26,
        "ShellKillGreenKoopa": 27,
        "StartLevel": 28,
        "StompKillBulletBill": 29,
        "StompKillGoomba": 30,
        "StompKillGreenKoopa": 31,
        "UnleashShell": 32,

        # --- Terminal / Outcome Events ---
        "DeathByBulletBill": 99,
        "DeathByGap": 98,
        "DeathByShell": 97,
        "DieByGoomba": 96,
        "DieByGreenKoopa": 95,
        "LostLevel": 94,
        "WonLevel": 93,
    }
    return mapping.get(event, 0)  # Returns 0 if event not found


# --- 3. PROCESS A SINGLE FILE ---
def process_csv(file_path, participant_id, filename):
    df = pd.read_csv(file_path)
    jumps = right = left = kills = powerups = coins = deaths = wins = 0
    trajectory = []
    powerup_events = {
    "BlockPowerDestroy",
    "BlockPowerDestroyBulletBill",
    "LargeStateStart",
    "FireStateStart",
}
    for _, row in df.iterrows():
        event = row['Event']
        code = map_event_to_code(event)

        # Count jumps
        if event == 'JumpStart':
            jumps += 1

        # Count right moves
        elif event == 'RightMoveStart':
            right += 1

        # Count left moves
        elif event == 'LeftMoveStart':
            left += 1

        # Count kills (any event with 'Kill')
        elif 'Kill' in event:
            kills += 1

        # Count power-ups
        elif event in powerup_events:
            powerups += 1

        # Count coins
        elif 'Coin' in event:
            coins += 1

        # Count deaths (all death or die events)
        elif 'Death' in event or event.startswith('Die'):
            deaths += 1

        # Count wins
        elif event == 'WonLevel':
            wins += 1

        trajectory.append([
            participant_id,
            filename,
            jumps,
            right,
            left,
            kills,
            powerups,
            coins,
            deaths,
            wins,
            code
        ])

    return trajectory

# --- 4. MAIN PROCESSING ---
def process_all_to_single_csv(input_root, output_csv_path):
    all_trajectories = []

    for dirpath, _, filenames in os.walk(input_root):
        for fname in filenames:
            if fname.endswith(".csv"):
                full_path = os.path.join(dirpath, fname)
                relative_path = os.path.relpath(full_path, input_root)

                participant_id = os.path.basename(os.path.dirname(full_path))
                trajectory = process_csv(full_path, participant_id, fname)
                all_trajectories.extend(trajectory)

    df = pd.DataFrame(all_trajectories, columns=[
        'participant_id', 'filename', 'jumps', 'right_moves', 'left_moves',
        'kills', 'powerups', 'coins_collected', 'deaths', 'wins', 'event_code'
    ])
    df.to_csv(output_csv_path, index=False)
    print(f"[✓] Combined CSV saved to: {output_csv_path}")

# --- 5. RUN ---
if __name__ == "__main__":
    zip_file = "AnonymizedDirectory.zip"
    extracted_folder = "AnonymizedDirectory"
    output_combined_csv = "combined_trajectories.csv"

    if not os.path.exists(extracted_folder):
        unzip_data(zip_file, extracted_folder)

    print("[✓] Generating single combined trajectory file...")
    process_all_to_single_csv(extracted_folder, output_combined_csv)
    print("[✓] Done.")
