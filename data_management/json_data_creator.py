import os
import glob
import json
import numpy as np
from typing import List, Dict, Tuple




def create_data_manifest(data_path, splits: Tuple[float, float, float], shuffle_seed: int, output_file: str):
    """
    Scans the specified folders (which are the independent datasets), 
    gathers image file paths, performs splitting, and saves the results to a JSON file (unsupervised format).
    """
    root_path_abs = os.path.abspath(data_path) 

    if not os.path.isdir(root_path_abs):
        raise FileNotFoundError(f"Data root directory not found at: {root_path_abs}. Please check your relative path configuration ('../../data').")

    # Find all immediate subdirectories (these are the independent datasets like 'ADNI', 'PPMI', etc.)
    discovered_folders = [
        os.path.join(root_path_abs, d) 
        for d in os.listdir(root_path_abs) 
        if os.path.isdir(os.path.join(root_path_abs, d))
    ]

    if not discovered_folders:
        raise RuntimeError(f"No dataset sub-folders found inside: {root_path_abs}.")

    print(f"\nDiscovered {len(discovered_folders)} dataset folders:")
    
    # Pass the discovered list of folders to the manifest creator
    t, v, te = splits


    # Data entries only store the image path (no label)
    all_data_entries: List[Dict[str, str]] = []
    
    # 1. Discover all image files
    for folder in discovered_folders:
        mask_count=0
        # Replicate the exact file discovery pattern: search for sub-* within the current dataset folder
        pattern = os.path.join(folder, "sub-*", "**", "anat", "*.nii.gz")
        found_images = sorted(glob.glob(pattern, recursive=True))
        
        valid_images = [f for f in found_images if "preproc" not in f.lower()]
        for f in valid_images:
            mask,count = get_mask(folder,f)
            if count:
                mask_count+=1
            dict = {'image': f}
            #dict['label'] = mask
            # Store full image path only
            all_data_entries.append(dict)
        print(f"Dataset '{os.path.basename(folder)}': found {len(valid_images)} images.")
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

    print(f"\nTotal files found: {total}. Splitting into train/val/test with ratios {t}/{v}/{te}.")
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


    if output_file is not False:
        print(f"Data manifest located at {output_file}\n")
    print(f"Training set size: {len(data_splits['TRAINING'])}.")
    print(f"Validation set size: {len(data_splits['VALIDATION'])}.")
    print(f"Test set size: {len(data_splits['TEST'])}.")
  
    return data_splits

def get_mask(folder, img_file):

    base = os.path.basename(img_file)
    base_noext = base
    for ext in ('.nii.gz', '.nii'):
        if base_noext.endswith(ext):
            base_noext = "_".join(base_noext[:-len(ext)].split("_")[:-1])
            break


    labels_dir = os.path.join(folder, 'derivatives', 'labels',base_noext)
    pattern = os.path.join(labels_dir, '**', f"{base_noext}*SC_seg*.nii*")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return matches[0],True
    return "./data_management/dummy/dummy_mask.nii.gz", False

