import os
import pandas as pd

# Set the root directory where all anonymized folders live
root_dir = "C:/Users/mukos/Desktop/MarioPCGStudy (1)/MarioPCGStudy/AnonymizedDirectory"

# Dictionary to store results
unique_events_summary = {}

# Walk through all folders and files under the root directory
for foldername, subfolders, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(".csv"):
            filepath = os.path.join(foldername, filename)
            try:
                df = pd.read_csv(filepath)
                if 'Event' in df.columns:
                    events = df['Event'].unique().tolist()
                    unique_events_summary[filepath] = events
                else:
                    unique_events_summary[filepath] = "❌ No 'Event' column found"
            except Exception as e:
                unique_events_summary[filepath] = f"❌ Failed to load: {e}"

# Display the summary
for path, result in unique_events_summary.items():
    print(f"\n📂 {path}")
    if isinstance(result, list):
        print("🔍 Unique Events:")
        for evt in result:
            print(f" - {evt}")
    else:
        print(result)
