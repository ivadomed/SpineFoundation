import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from model.build import build_model
from data_management.dataloader import build_dataloaders
from .utils import patchify, save_checkpoint, load_checkpoint, load_json_param, list_child_folders


class Trainer:
    def __init__(self, args):
        self.args = args

        model_params = load_json_param(args.model_params)
        data_params = load_json_param(args.data_params)

        self.model_params = model_params
        self.data_params = data_params
       
        self.model_name=model_params["model_name"]
        self.in_channels=model_params["in_channels"]
        self.img_size=model_params["img_size"]
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
        self.num_workers = data_params["num_workers"]
        self.train_ratio = data_params["train_ratio"]
        self.val_ratio = data_params["val_ratio"]
        self.test_ratio = data_params["test_ratio"]
        self.seed = data_params["seed"]
        self.work_dir = data_params["work_dir"]
        self.epochs = data_params["epochs"]
        self.data_path = data_params["data_path"]
        self.lr = data_params["lr"]
        self.weight_decay = data_params["weight_decay"]
        self.amp = data_params["amp"]
        self.no_cuda = data_params["no_cuda"]
        self.resume = data_params["resume"]

        self.device = torch.device('cuda' if (torch.cuda.is_available() and not self.no_cuda) else 'cpu') 
        
        self.model = build_model(self.model_name, data_params.pop("model_name", None)
)
        self.model.to(self.device)


        self.optimizer = AdamW(self.model.parameters(),lr=self.lr,weight_decay=self.weight_decay,)
        self.scaler = GradScaler(enabled=self.amp)
        self.criterion = nn.L1Loss()

        folders = list_child_folders(self.data_path)

        splits=(self.train_ratio, self.val_ratio, self.test_ratio)
        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
                                                                img_size=self.img_size,
                                                                batch_size=self.batch_size,
                                                                folders=folders,
                                                                num_workers=self.num_workers,
                                                                shuffle_seed=self.seed,
                                                                splits=splits,
                                                            )
        
        self.start_epoch = 0
        self.best_val = float('inf')
        if self.resume:
            ckpt = load_checkpoint(self.resume, self.device)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.start_epoch = ckpt.get('epoch', 0) + 1
            self.best_val = ckpt.get('val_loss', float('inf'))
            print(f"Resumed from {self.resume} at epoch {self.start_epoch}")


    def train_step(self, batch):
        self.model.train()

        x = batch["image"].to(self.device)
        if x.ndim == 4:  # (B, D, H, W) -> (B, 1, D, H, W)
            x = x.unsqueeze(1)

        with autocast(enabled=self.amp):
            pred = self.model(x)

            target = x

            loss = self.criterion(pred, target)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def train_one_epoch(self, epoch: int):
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        for i, batch in enumerate(pbar, start=1):
            loss = self.train_step(batch)
            running_loss += loss
            pbar.set_postfix({'loss': running_loss / i})

    def validate(self, epoch: int):
        self.model.eval()
        total = 0.0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["image"].to(self.device)
                if x.ndim == 4:
                    x = x.unsqueeze(1)

                with autocast(enabled=self.amp):
                    pred = self.model(x)
                    if pred.shape == x.shape:
                        target = x
                    else:
                        target = patchify(x, self.patch_size)

                    loss = self.criterion(pred, target)

                total += loss.item() * x.shape[0]
                count += x.shape[0]

        avg = total / max(1, count)
        print(f"Validation loss (epoch {epoch}): {avg:.6f}")
        return avg

   
    def fit(self):
        for epoch in range(self.start_epoch, self.epochs):
            self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)

            is_best = val_loss < self.best_val
            self.best_val = min(self.best_val, val_loss)

            ckpt = {
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'val_loss': val_loss,
            }

            save_checkpoint(
                ckpt,
                os.path.join(self.work_dir, f'ckpt_epoch_{epoch}.pt'),
            )
            if is_best:
                save_checkpoint(
                    ckpt,
                    os.path.join(self.work_dir, 'best.ckpt'),
                )


