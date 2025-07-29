import pandas as pd
import os

# List of your CSV file paths (make sure these point to the actual locations on your machine)
file_paths = [
    "C:/Users/mukos/Desktop/MarioPCGStudy (1)/MarioPCGStudy/AnonymizedDirectory/0007113/MarioLevelInfoTimeline.csv",
    "C:/Users/mukos/Desktop/MarioPCGStudy (1)/MarioPCGStudy/AnonymizedDirectory/0007113/TestLevel1InfoTimeline.csv",
    "C:/Users/mukos/Desktop/MarioPCGStudy (1)/MarioPCGStudy/AnonymizedDirectory/0007113/TestLevel2InfoTimeline.csv"
]


# Dictionary to hold the results
unique_events_summary = {}

# Loop through and extract unique event types from each file
for file_path in file_paths:
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if 'Event' in df.columns:
            unique_events = df['Event'].unique().tolist()
            unique_events_summary[file_path] = unique_events
        else:
            unique_events_summary[file_path] = "❌ 'Event' column not found"
    else:
        unique_events_summary[file_path] = "❌ File not found"

# Display the results
for file, events in unique_events_summary.items():
    print(f"\n📂 {file}")
    if isinstance(events, list):
        print("🔍 Unique Events:")
        for event in events:
            print(f" - {event}")
    else:
        print(events)
