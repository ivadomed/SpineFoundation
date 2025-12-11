import os
import time
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm

from model.build import build_model
from training.augment import GPUResampleAug3D
from training.utils import load_json_param
from training.utils import load_checkpoint
from training.utils import collate_fn
from torch.utils.data import DataLoader
from data_management.build import build_datasets

class InferenceRunner:
    def __init__(self, args):
        self.args = args
        
        model_params = load_json_param(args.model_params)
        data_params = load_json_param(args.data_params)
        training_params = load_json_param(args.training_params)

        self.model_params = model_params
        self.data_params = data_params
       
        self.model_name=model_params["model_name"]
        self.in_channels=model_params["in_channels"]
        self.img_size=tuple(model_params["img_size"])
        self.img_resolution=tuple(model_params["img_resolution"])
        self.patch_size=model_params["patch_size"]
        self.enc_embed_dim=model_params["enc_embed_dim"]
        self.enc_num_heads=model_params["enc_num_heads"]
        self.enc_layers=model_params["enc_layers"]
        self.enc_mlp_dim=model_params["enc_mlp_dim"]
        self.dropout=model_params["dropout"]
        self.mask_ratio=model_params["mask_ratio"]
        self.dec_embed_dim=model_params["dec_embed_dim"]
        self.dec_layers=model_params["dec_layers"]
        self.dec_num_heads=model_params["dec_num_heads"]
        self.dec_mlp_dim=model_params["dec_mlp_dim"]

        self.batch_size = data_params["batch_size"]      
        self.train_ratio = data_params["train_ratio"]
        self.val_ratio = data_params["val_ratio"]
        self.test_ratio = data_params["test_ratio"]
        self.seed = data_params["seed"]
        self.data_path = data_params["data_path"]
        self.json_manifest = data_params.get("json_manifest", None)
        

        self.global_step = 0
        
        self.epochs = training_params["epochs"]
        self.work_dir = training_params["work_dir"]
        self.num_workers = training_params["num_workers"]
        self.wandb = training_params["wandb"]
        self.log_image_interval = training_params["log_image_interval"]
        self.lr = training_params["lr"]
        self.weight_decay = training_params["weight_decay"]
        self.amp = training_params["amp"]
        self.no_cuda = training_params["no_cuda"]
        self.resume = training_params["resume"]
        self.tqdm_disable = training_params["tqdm_disable"]

        os.makedirs(self.outdir, exist_ok=True)

        self.device = torch.device('cuda' if (torch.cuda.is_available() and not self.no_cuda) else 'cpu')

        # Model
        mp = dict(self.model_params)
        model_name = mp.pop('model_name')
        img_resolution = tuple(mp.pop('img_resolution'))
        self.model = build_model(model_name, mp).to(self.device)
        ckpt = load_checkpoint(self.ckpt_path, self.device)
        try:
            self.model.load_state_dict(ckpt['model'])
        except Exception:
            # allow loading from DDP-saved checkpoints
            self.model.load_state_dict(ckpt)
        self.model.eval()

        # GPU transform
        img_size = tuple(self.model_params['img_size'])
        self.gpu_tf = GPUResampleAug3D(img_size=img_size, target_res=img_resolution).to(self.device)

        # Data
        train_ratio = self.data_params['train_ratio']
        val_ratio = self.data_params['val_ratio']
        test_ratio = self.data_params['test_ratio']
        seed = self.data_params['seed']
        data_path = self.data_params['data_path']
        json_manifest = self.data_params.get('json_manifest', None)

        # Use validation + test splits for inference; you can change this to train if needed
        train_ds, val_ds, test_ds = build_datasets(
            data_path=data_path,
            json_path=json_manifest,
            splits=(train_ratio, val_ratio, test_ratio),
            shuffle_seed=seed,
        )
        self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn)

    @torch.no_grad()
    def _infer_batch(self, batch):
        images = [b['image'].to(self.device, non_blocking=True) for b in batch]
        spacings = [torch.as_tensor(b['image'].meta['spacing_dhw'], dtype=torch.float32, device=self.device) for b in batch]
        x = self.gpu_tf(images, spacings)
        pred = self.model(x)
        return x, pred

    def _save_pred(self, rec, x, pred, split: str, idx: int):
        # Save as .npy for now; NIfTI saving would need original affine/shape mapping
        base = os.path.join(self.outdir, split)
        os.makedirs(base, exist_ok=True)
        np.save(os.path.join(base, f"input_{idx}.npy"), x.detach().cpu().numpy())
        np.save(os.path.join(base, f"pred_{idx}.npy"), pred.detach().cpu().numpy())

    def run(self):
        for split_name, loader in [('val', self.val_loader), ('test', self.test_loader)]:
            for i, batch in tqdm(enumerate(loader, start=0), total=len(loader), desc=f"Infer {split_name}"):
                x, pred = self._infer_batch(batch)
                self._save_pred(batch[0], x[0, 0], pred[0, 0], split_name, i)
