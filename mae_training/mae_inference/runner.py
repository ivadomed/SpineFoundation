import os
import time
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from model.build import build_model
from mae_training.augment import GPUResampleAug3D
from mae_training.utils import load_json_param
from mae_training.utils import load_checkpoint
from mae_training.utils import collate_fn
from torch.utils.data import DataLoader
from data_management.build import build_datasets
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
        

        model_params = load_json_param(args.model_params)
        data_params = load_json_param(args.data_params)
        training_params = load_json_param(args.training_params)

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

        self.gpu_tf = GPUResampleAug3D(img_size=self.img_size, target_res=self.img_resolution).to(self.device)

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

    @torch.no_grad()
    def _infer_batch(self, batch):
        images = [b['image'].to(self.device, non_blocking=True) for b in batch]
        spacings = [torch.as_tensor(b['image'].meta['spacing_dhw'], dtype=torch.float32, device=self.device) for b in batch]
        x = self.gpu_tf(images, spacings)
        pred = self.model(x)
        return x, pred

    def _save_pred(self, batch, x, pred, split: str, idx_start: int):
        out_dir = os.path.join(self.outdir, split)
        os.makedirs(out_dir, exist_ok=True)

        B = x.shape[0]
        root = os.path.abspath(self.data_path)

        for b in range(B):
            vol_in   = x[b, 0]
            vol_gt   = x[b, 0]
            vol_pred = pred[b, 0]

            rec = batch[b]
            full_path = os.path.abspath(rec["image"].meta["filename_or_obj"])

            # chemin relatif par rapport à data_path
            try:
                rel = os.path.relpath(full_path, root)
                parts = rel.split(os.sep)
                dataset_name = parts[0] if len(parts) > 0 else "unknown_dataset"
            except Exception:
                dataset_name = "unknown_dataset"

            fname = os.path.basename(full_path)          # sub-XXX_T2w.nii.gz
            stem = fname[:-7] if fname.endswith(".nii.gz") else os.path.splitext(fname)[0]

            out_name = f"{dataset_name}__{stem}_slices.png"
            out_path = os.path.join(out_dir, out_name)

            self.save_6_middle_slices(vol_in, vol_gt, vol_pred, out_path)



    def run(self):

        split_name = "val"
        loader = self.val_loader

        for i, batch in tqdm(enumerate(loader), total=len(loader), desc=f"Infer {split_name}"):
            x, pred = self._infer_batch(batch)
            idx_start = i * self.val_loader.batch_size
            self._save_pred(batch, x, pred, split_name, idx_start)
