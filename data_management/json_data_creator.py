import os
import glob
import json
import numpy as np
from typing import List, Dict, Tuple

# --- Configuration for Manifest Creation ---
OUTPUT_JSON_FILE = "data_splits.json"
# The root path containing all your independent datasets (e.g., ../../data/datasetA, ../../data/datasetB)
DATA_ROOT_PATH = "../../data" 
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # (train, val, test)
SHUFFLE_SEED = 42 # Use a fixed seed for reproducible splits
# -------------------------------------------


def create_data_manifest(folders: List[str], splits: Tuple[float, float, float], shuffle_seed: int, output_file: str):
    """
    Scans the specified folders (which are the independent datasets), 
    gathers image file paths, performs splitting, and saves the results to a JSON file (unsupervised format).
    """
    t, v, te = splits
    if abs(t + v + te - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")

    # Data entries only store the image path (no label)
    all_data_entries: List[Dict[str, str]] = []
    
    # 1. Discover all image files
    for folder in folders:
        # Replicate the exact file discovery pattern: search for sub-* within the current dataset folder
        pattern = os.path.join(folder, "sub-*", "**", "anat", "*.nii.gz")
        found_images = sorted(glob.glob(os.path.join(pattern), recursive=True))
        
        # Replicate the exact exclusion filtering
        valid_images = [f for f in found_images if "ax" not in f.lower() and "cor" not in f.lower() and "preproc" not in f.lower()]

        for img_path in valid_images:
            # Store full image path only
            all_data_entries.append({
                'image': img_path
            })

    total = len(all_data_entries)
    if total == 0:
        # Added clarity to the error message
        print("-" * 60)
        raise RuntimeError(f"No valid image files found using BIDS-like pattern in the discovered folders.")

    # 2. Perform the Split
    indices = list(range(total))
    
    # Replicate shuffle seed logic
    rng = np.random.RandomState(shuffle_seed)
    rng.shuffle(indices)

    print(f"Total files found: {total}. Splitting into train/val/test with ratios {t}/{v}/{te}.")
    n_train = int(total * t)
    n_val = int(total * v)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    # The rest go to test
    test_indices = indices[n_train + n_val:] 

    # 3. Assemble the final data splits dictionary
    data_splits = {
        "TRAINING": [all_data_entries[i] for i in train_indices],
        "VALIDATION": [all_data_entries[i] for i in val_indices],
        "TEST": [all_data_entries[i] for i in test_indices]
    }

    # 4. Save to JSON
    with open(output_file, 'w') as f:
        json.dump(data_splits, f, indent=4)

    print("-" * 60)
    print(f"Data manifest successfully saved to {output_file}")
    print(f"Training set size: {len(data_splits['TRAINING'])}")
    print(f"Validation set size: {len(data_splits['VALIDATION'])}")
    print(f"Test set size: {len(data_splits['TEST'])}")
    print("-" * 60)

if __name__ == "__main__":
    
    # --- FIX: Dynamically discover all dataset folders inside the root path ---
    
    # Resolve the absolute path to make sure os.path.isdir works correctly
    root_path_abs = os.path.abspath(DATA_ROOT_PATH) 

    if not os.path.isdir(root_path_abs):
        raise FileNotFoundError(f"Data root directory not found at: {root_path_abs}. Please check your relative path configuration ('../../data').")

    # Find all immediate subdirectories (these are the independent datasets like 'ADNI', 'PPMI', etc.)
    discovered_folders = [
        os.path.join(root_path_abs, d) 
        for d in os.listdir(root_path_abs) 
        if os.path.isdir(os.path.join(root_path_abs, d))
    ]

    if not discovered_folders:
        raise RuntimeError(f"No dataset sub-folders found inside: {root_path_abs}. Please ensure your BIDS-style datasets are immediate subdirectories of '{DATA_ROOT_PATH}'.")

    print(f"Discovered {len(discovered_folders)} dataset folders: {discovered_folders}")
    
    # Pass the discovered list of folders to the manifest creator
    create_data_manifest(discovered_folders, SPLIT_RATIOS, SHUFFLE_SEED, OUTPUT_JSON_FILE)