"""
Download and combine all PPG cholesterol data from Figshare
"""

import json
import requests
import pandas as pd
from pathlib import Path
import time

# Create directory
data_dir = Path("datasets/ppg_cholesterol/all_subjects")
data_dir.mkdir(exist_ok=True, parents=True)

print("Fetching file list from Figshare...")

# Get all CSV files
response = requests.get("https://api.figshare.com/v2/articles/27132552/files")
files = json.loads(response.text)

print(f"Found {len(files)} CSV files to download")

# Download each file
all_data = []
for i, file_info in enumerate(files):
    filename = file_info['name']
    url = file_info['download_url']

    print(f"Downloading {i+1}/{len(files)}: {filename}...")

    # Download file
    response = requests.get(url)
    filepath = data_dir / filename

    with open(filepath, 'wb') as f:
        f.write(response.content)

    # Read and append to combined data
    try:
        df = pd.read_csv(filepath)
        all_data.append(df)
        print(f"  Added {len(df)} rows from {filename}")
    except Exception as e:
        print(f"  Error reading {filename}: {e}")

    # Be nice to the server
    time.sleep(0.5)

# Combine all data
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    output_file = Path("datasets/ppg_cholesterol/all_ppg_signals.csv")
    combined_df.to_csv(output_file, index=False)

    print(f"\n✅ Combined {len(all_data)} files into {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Unique subjects: {combined_df[['Edad', 'Sexo', 'Colesterol']].drop_duplicates().shape[0]}")

    # Show summary
    print("\nData summary:")
    print(combined_df.groupby(['Edad', 'Sexo', 'Colesterol']).size().head(10))
else:
    print("No data downloaded")