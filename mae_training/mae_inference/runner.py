import os
import time
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import json
import csv
from collections import defaultdict
import matplotlib.pyplot as plt


from model.build import build_model
from mae_training.utils import load_json_param
from mae_training.utils import load_checkpoint
from mae_training.utils import collate_fn
from torch.utils.data import DataLoader
from data_management.build import build_datasets
from mae_training.augment import GPUResampleAug3D
from mae_training.utils import plot_6_middle_slices



def build_model_from_ckpt(ckpt_path, device):
    ckpt = load_checkpoint(ckpt_path, device)
    model_name = ckpt["model_name"]
    mp = dict(ckpt["model_params"])

    model = build_model(model_name, mp, rank=0).to(device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model


class InferenceRunner:
    def __init__(self, args):
        self.args = args
        
        conf= load_json_param(args.config)
        model_params = conf["Model"]
        data_params = conf["Data"]
        training_params = conf["Training"]

        self.ckpt_path = args.model_ckpt
        self.outdir = args.outdir

        self.model_params = model_params
        self.data_params = data_params
       
        self.model_name=model_params["model_name"]
        self.img_size=tuple(model_params["img_size"])
        self.img_resolution=tuple(model_params["img_resolution"])


        self.batch_size = data_params["batch_size"]      
        self.data_path = data_params["data_path"]
        self.json_manifest = data_params.get("json_manifest", None)
        self.only_validation = data_params.get("only_validation", False)

        self.num_workers = training_params["num_workers"]
        self.no_cuda = training_params["no_cuda"]
        self.tqdm_disable = training_params["tqdm_disable"]

        os.makedirs(self.outdir, exist_ok=True)

        self.device = torch.device('cuda' if (torch.cuda.is_available() and not self.no_cuda) else 'cpu')
        print("\nDEVICE :\n")
        print(f"Using device: {self.device}")

        mp = dict(model_params)
        mp.pop("model_name", None)
        mp.pop("img_resolution", None)
        self.model = build_model(self.model_name, mp).to(self.device)
        ckpt = load_checkpoint(self.ckpt_path, self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.gpu_tf = GPUResampleAug3D(img_size=self.img_size, target_res=self.img_resolution, inference=True).to(self.device)


        data_path = self.data_params['data_path']
        json_manifest = self.data_params.get('json_manifest', None)

        val_ds= build_datasets(
            data_path=None,json_path=self.json_manifest,splits=None,shuffle_seed=None,rank=0,inference=True,only_validation=self.only_validation
        )
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn)

    def save_6_middle_slices(self, image: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor, out_path: str):
        fig = plot_6_middle_slices(image, gt, pred)
        fig.savefig(out_path)
        plt.close(fig)
    def _dataset_from_path_5th_after_home(self, full_path: str) -> str:
        parts = os.path.abspath(full_path).split(os.sep)
        # parts: ["", "home", "xxx", "yyy", ...]
        try:
            home_idx = parts.index("home")
            target_idx = home_idx + 5  # 5e entité en partant de /home/
            if target_idx < len(parts) and parts[target_idx]:
                return parts[target_idx]
        except ValueError:
            pass
        return "unknown_dataset"

    def _per_sample_l1(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: (B, C, D, H, W) ou (B, 1, D, H, W)
        # retourne (B,)
        dims = tuple(range(1, pred.ndim))
        return (pred - target).abs().mean(dim=dims)

    @torch.no_grad()
    def _infer_batch(self, batch):
        images = [b['image'].to(self.device, non_blocking=True) for b in batch]
        spacings = [torch.as_tensor(b['image'].meta['spacing_dhw'], dtype=torch.float32, device=self.device) for b in batch]
        x,infos = self.gpu_tf(images, spacings)
        pred = self.model(x)
        per_sample_loss = self._per_sample_l1(pred, x)
        return x, pred, infos, per_sample_loss

    def _get_dataset_name(self, meta):
        full_path = os.path.abspath(meta["filename_or_obj"])
        root = os.path.abspath(self.data_path)
        try:
            rel = os.path.relpath(full_path, root)
            parts = rel.split(os.sep)
            return parts[0] if len(parts) > 0 else "unknown_dataset"
        except Exception:
            return "unknown_dataset"

    def _reconstruct_volume(self, pred_vol: torch.Tensor, info: dict, meta: dict) -> np.ndarray:
        # pred_vol : (D_t, H_t, W_t) sur img_size
        pred = pred_vol.detach().cpu()
        Dz, Dh, Dw = info["resampled_shape_dhw"]
        pw0, pw1, ph0, ph1, pz0, pz1 = info["img_pad"]
        z0, z1, y0, y1, x0, x1 = info["img_crop_slices"]
        D2 = Dz + pz0 + pz1
        H2 = Dh + ph0 + ph1
        W2 = Dw + pw0 + pw1
        vol_pad = torch.zeros((D2, H2, W2), dtype=pred.dtype)
        vol_pad[z0:z1, y0:y1, x0:x1] = pred
        vol_res = vol_pad[pz0:D2 - pz1, ph0:H2 - ph1, pw0:W2 - pw1]
        m = info["norm_mean"]
        s = info["norm_std"]
        vol_res = vol_res * s + m
        orig_shape = tuple(meta["orig_shape_dhw"])
        vol_res = vol_res.unsqueeze(0).unsqueeze(0)
        vol_orig = torch.nn.functional.interpolate(vol_res, size=orig_shape, mode="trilinear", align_corners=False).squeeze(0).squeeze(0).cpu().numpy()
        return vol_orig


    def _save_pred(self, batch, x, pred, infos, split: str, idx_start: int):
        out_dir = os.path.join(self.outdir, split)
        os.makedirs(out_dir, exist_ok=True)
        B = x.shape[0]
        root = os.path.abspath(self.data_path)
        for b in range(B):
            vol_in = x[b, 0]
            vol_pred = pred[b, 0]
            rec = batch[b]
            meta = rec["image"].meta
            full_path = os.path.abspath(meta["filename_or_obj"])
            try:
                rel = os.path.relpath(full_path, root)
                parts = rel.split(os.sep)
                dataset_name = parts[0] if len(parts) > 0 else "unknown_dataset"
            except Exception:
                dataset_name = "unknown_dataset"
            fname = os.path.basename(full_path)
            stem = fname[:-7] if fname.endswith(".nii.gz") else os.path.splitext(fname)[0]
            # 1) PNG de slices (optionnel)
            out_name_png = f"{dataset_name}__{stem}_slices.png"
            out_path_png = os.path.join(out_dir, out_name_png)
            self.save_6_middle_slices(vol_in, vol_in, vol_pred, out_path_png)
            # 2) Copier le GT en _GT.nii.gz
            out_gt = os.path.join(out_dir, f"{dataset_name}__{stem}_GT.nii.gz")
            if not os.path.exists(out_gt):
                shutil.copy2(full_path, out_gt)
            # 3) Reconstruire le NIfTI de la prédiction en _RECON.nii.gz
            info = infos[b]
            recon_np = self._reconstruct_volume(vol_pred, info, meta)
            if "orig_affine" in meta:
                affine = np.asarray(meta["orig_affine"], dtype=float)
            else:
                affine = nib.load(full_path).affine
            recon_img = nib.Nifti1Image(recon_np.astype(np.float32), affine)
            out_recon = os.path.join(out_dir, f"{dataset_name}__{stem}_RECON.nii.gz")
            nib.save(recon_img, out_recon)

    def summarize(self,v):
        v = np.asarray(v)
        return {
            "n": int(len(v)),
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()),
            "median": float(np.median(v)),
            "max": float(v.max()),
        }


    def run(self):
        split_name = "val"
        loader = self.val_loader
        losses_by_dataset = defaultdict(list)
        for i, batch in tqdm(enumerate(loader), total=len(loader), desc=f"Infer {split_name}"):
            x, pred, infos, per_sample_loss = self._infer_batch(batch)
            for b in range(len(batch)):
                meta = batch[b]["image"].meta
                dataset_name = self._get_dataset_name(meta)
                l = float(per_sample_loss[b].cpu().item())
                losses_by_dataset[dataset_name].append(l)
            idx_start = i * self.val_loader.batch_size
            #self._save_pred(batch, x, pred, infos, split_name, idx_start)
    
        stats = {k: self.summarize(v) for k, v in losses_by_dataset.items()}
        datasets = sorted(losses_by_dataset.keys())
        means, stds, mins, maxs = [], [], [], []

        for d in datasets:
            v = np.asarray(losses_by_dataset[d], dtype=np.float32)
            means.append(v.mean())
            stds.append(v.std(ddof=1) if v.size > 1 else 0.0)
            mins.append(v.min())
            maxs.append(v.max())

        x = np.arange(len(datasets))

        plt.figure(figsize=(max(6, 0.8 * len(datasets)), 4))

        # barres mean ± std
        plt.bar(
            x,
            means,
            yerr=stds,
            capsize=6,
            alpha=0.8,
            label="mean ± std"
        )

        # triangles min / max par dataset
        plt.scatter(x, mins, marker="v", s=60, label="min")
        plt.scatter(x, maxs, marker="^", s=60, label="max")

        plt.xticks(x, datasets, rotation=45, ha="right")
        plt.ylabel("Loss (L1)")
        plt.title("Loss statistics per dataset")
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(self.outdir, "loss_bar_by_dataset.png"), dpi=150)
plt.close()