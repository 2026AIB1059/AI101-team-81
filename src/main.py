import os
import pandas as pd

def load_and_verify_data():
    """Loads and verifies F1 dataset files."""
    print("==========================================")
    print("   AI101 Team 81: F1 Race Predictor Pipeline   ")
    print("==========================================")
    
    # Path to primary dataset directory
    dataset_dir = "Formula 1 Dataset"
    results_path = os.path.join(dataset_dir, "results.csv")
    races_path = os.path.join(dataset_dir, "races.csv")
    drivers_path = os.path.join(dataset_dir, "drivers.csv")
    
    # Check file existence
    for path in [results_path, races_path, drivers_path]:
        if os.path.exists(path):
            print(f"[SUCCESS] Verified file: {path}")
        else:
            print(f"[WARNING] File not found: {path}")

    # Load primary data frames
    if os.path.exists(results_path) and os.path.exists(races_path):
        results_df = pd.read_csv(results_path)
        races_df = pd.read_csv(races_path)
        print(f"\n[INFO] Loaded Results: {results_df.shape[0]} rows, {results_df.shape[1]} columns")
        print(f"[INFO] Loaded Races: {races_df.shape[0]} rows, {races_df.shape[1]} columns")
        print("[SUCCESS] Data verification pipeline complete!")
    else:
        print("\n[ERROR] Dataset files missing. Please check data directory.")

if __name__ == "__main__":
    load_and_verify_data()