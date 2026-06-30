import os
import csv
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    ToTensord, SpatialPadd, CenterSpatialCropd, Spacingd,
    NormalizeIntensityd,
)
from monai.networks.nets import ResNet
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score


def parse_args():
    parser = argparse.ArgumentParser(description="SCS inference script.")
    parser.add_argument('--data',       required=True,  help="Path to the 3D data directory.")
    parser.add_argument('--model_path', required=True,  help="Path to the folder containing claude_scs16.pth.")
    parser.add_argument('--output_csv', required=True,  help="Path for the output CSV.")
    parser.add_argument('--fold_csv',   default=None,   help="fold_split_RSNA.json — filter to is_test=1 subjects.")
    parser.add_argument('--gt_dir',     default=None,   help="RSNA_patches_scs dir with 0/1/2 subdirs for ground truth.")
    return parser.parse_args()


def get_transforms():
    return Compose([
        LoadImaged(keys=['T2']),
        EnsureChannelFirstd(keys=['T2']),
        Spacingd(keys=['T2'], pixdim=(4, 0.4, 0.4), mode='bilinear'),
        SpatialPadd(keys=['T2'], spatial_size=(6, 80, 80)),
        CenterSpatialCropd(keys=['T2'], roi_size=(6, 80, 80)),
        ScaleIntensityd(keys=['T2']),
        NormalizeIntensityd(keys=['T2'], nonzero=True, channel_wise=True),
        ToTensord(keys=['T2']),
    ])


def load_test_subjects(fold_csv):
    import pandas as pd
    df = pd.read_csv(fold_csv)
    return set(df[df['is_test'] == 1]['subject_id'].tolist())


def prepare_data(data_dir, transform, test_subjects=None):
    data = []
    for subject in sorted(os.listdir(data_dir)):
        if test_subjects is not None and subject not in test_subjects:
            continue
        subject_dir = os.path.join(data_dir, subject, 'anat')
        if not os.path.isdir(subject_dir):
            continue
        for file in sorted(os.listdir(subject_dir)):
            # SCS uses axial patches (acq-ax)
            if '_patch.nii.gz' not in file or 'acq-ax' not in file:
                continue
            image_path = os.path.join(subject_dir, file)
            parts      = file.split('_')
            # last parts: ..._L5_S1_patch.nii.gz → parts[-3]=L5, parts[-2]=S1
            disk_level = f"{parts[-3]}_{parts[-2]}"
            label_key  = f"{subject}_spinal_canal_stenosis_{disk_level.lower()}"
            data.append({"T2": image_path, "label": label_key})

    print(f"Loaded {len(data)} samples from {data_dir}")
    return Dataset(data=data, transform=transform)


def build_gt_dict(gt_dir):
    """Build {(subject_id, level_norm): label_int} from RSNA_patches_scs 0/1/2 dirs.

    NPZ filenames: sub-XXXXX_acq-sag_rec-YYY_T2w_desc-L2L3_label-SpinalCanalStenosis_label.npz
    Level extracted from desc-L2L3 → l2_l3.
    """
    gt = {}
    for label_int in range(3):
        label_dir = Path(gt_dir) / str(label_int)
        if not label_dir.exists():
            continue
        for npz in label_dir.glob("*.npz"):
            name = npz.stem
            # extract subject: first token before first underscore after "sub-"
            subject = name.split("_acq")[0]  # e.g. sub-1698156042
            # extract level from desc-L2L3
            desc_part = [p for p in name.split("_") if p.startswith("desc-")]
            if not desc_part:
                continue
            raw_level = desc_part[0].replace("desc-", "")  # e.g. L2L3
            # insert underscore: L2L3 → l2_l3
            import re
            m = re.match(r"([A-Za-z]+\d+)([A-Za-z]+\d+)", raw_level)
            if not m:
                continue
            level_norm = f"{m.group(1).lower()}_{m.group(2).lower()}"  # l2_l3
            gt[(subject, level_norm)] = label_int
    return gt


def load_model(model_path, filename, device, layers=(2, 2, 2, 2)):
    model = ResNet(
        block            = 'basic',
        layers           = list(layers),
        block_inplanes   = [64, 128, 256, 512],
        spatial_dims     = 3,
        n_input_channels = 1,
        num_classes      = 3,
    ).to(device)
    ckpt = os.path.join(model_path, filename)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    print(f"Loaded {ckpt}")
    return model


def inference(device, data_dir, model_path, test_subjects=None, batch_size=16, layers=(2, 2, 2, 2)):
    dataset     = prepare_data(data_dir, get_transforms(), test_subjects)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    models = [load_model(model_path, 'claude_scs16.pth', device, layers)]

    predictions = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Inference"):
            inputs = batch["T2"].to(device)
            labels = batch["label"]
            logits = sum(m(inputs) for m in models) / len(models)
            probs  = logits.softmax(dim=1).cpu().numpy()
            for label_key, prob in zip(labels, probs):
                predictions.append((label_key, prob.tolist()))

    return sorted(predictions, key=lambda x: x[0])


def write_csv(predictions, output_csv):
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "pathology", "level",
                         "Normal/Mild", "Moderate", "Severe"])
        for label_key, probs in predictions:
            parts     = label_key.split("_")
            subject   = parts[0]
            pathology = "_".join(parts[1:-2])
            level     = "_".join(parts[-2:])
            writer.writerow([subject, pathology, level,
                             round(probs[0], 4), round(probs[1], 4), round(probs[2], 4)])
    print(f"Saved predictions → {output_csv}")


def compute_metrics(predictions, gt_dict):
    label_map = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
    all_probs, all_labels = [], []
    skipped = 0

    for label_key, probs in predictions:
        parts   = label_key.split("_")
        subject = parts[0]
        level   = "_".join(parts[-2:])  # already lowercased
        gt = gt_dict.get((subject, level))
        if gt is None:
            skipped += 1
            continue
        all_probs.append(probs)
        all_labels.append(gt)

    if not all_labels:
        print("No matching ground truth found — skipping metrics.")
        return

    if skipped:
        print(f"[warn] {skipped} samples skipped (no GT match)")

    probs_arr  = np.array(all_probs, dtype=np.float64)
    labels_arr = np.array(all_labels, dtype=np.int64)
    preds_arr  = probs_arr.argmax(axis=1)

    auc = roc_auc_score(labels_arr, probs_arr, multi_class="ovr", average="macro")
    acc = accuracy_score(labels_arr, preds_arr)
    qwk = cohen_kappa_score(labels_arr, preds_arr, weights="quadratic")

    print(f"\n{'='*40}")
    print(f"  claude_scs16 — test set  (n={len(labels_arr)})")
    print(f"{'='*40}")
    print(f"  AUC  : {auc:.4f}")
    print(f"  Acc  : {acc:.4f}")
    print(f"  QWK  : {qwk:.4f}")
    print(f"{'='*40}\n")


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model folder not found: {args.model_path}")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data directory not found: {args.data}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_subjects = None
    if args.fold_csv:
        test_subjects = load_test_subjects(args.fold_csv)
        print(f"Test subjects from CSV: {len(test_subjects)}")

    predictions = inference(
        device        = device,
        data_dir      = args.data,
        model_path    = args.model_path,
        test_subjects = test_subjects,
        batch_size    = 16,
        layers        = (2, 2, 2, 2),
    )

    write_csv(predictions, args.output_csv)

    if args.gt_dir:
        gt_dict = build_gt_dict(args.gt_dir)
        print(f"Ground truth entries loaded: {len(gt_dict)}")
        compute_metrics(predictions, gt_dict)


if __name__ == "__main__":
    main()
