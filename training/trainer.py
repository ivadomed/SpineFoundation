import os
import json
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

import wandb
import matplotlib.pyplot as plt

from model.build import build_model
from data_management.build import build_datasets


from .utils import patchify, save_checkpoint, load_checkpoint, load_json_param, list_child_folders, plot_6_middle_slices, plot_6_uniform_slices
from .loss import MSEwloss

TIME_CHECK = True 


class Trainer:
    def __init__(self, args):
        self.args = args

        model_params = load_json_param(args.model_params)
        data_params = load_json_param(args.data_params)
        training_params = load_json_param(args.training_params)

        self.model_params = model_params
        self.data_params = data_params
       
        self.model_name=model_params["model_name"]
        self.in_channels=model_params["in_channels"]
        self.img_size=model_params["img_size"]
        self.img_resolution=model_params["img_resolution"]
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


        self.device = torch.device('cuda' if (torch.cuda.is_available() and not self.no_cuda) else 'cpu') 
        print("\n========== DEVICE ==========")
        print(f"Using device: {self.device}")
        model_params.pop("model_name", None)
        model_params.pop("img_resolution", None)
        
        self.model = build_model(self.model_name, model_params)
        self.model.to(self.device)


        self.optimizer = AdamW(self.model.parameters(),lr=self.lr,weight_decay=self.weight_decay,)
        self.scaler = GradScaler(device=self.device, enabled=self.amp)
        self.criterion = MSEwloss()

        

        self.train_loader, self.val_loader, self.test_loader = build_datasets(
                                                                data_path=self.data_path,
                                                                json_path=self.json_manifest,
                                                                splits=(self.train_ratio, self.val_ratio, self.test_ratio),
                                                                img_size=self.img_size,
                                                                img_resolution=self.img_resolution,
                                                                batch_size=self.batch_size,
                                                                num_workers=self.num_workers,
                                                                shuffle_seed=self.seed,
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


    def train_step(self, batch, iteration: int, epoch: int):
        self.global_step += 1
        self.model.train()

        def now():
            if TIME_CHECK and self.device.type == "cuda":
                torch.cuda.synchronize()
            return time.time()

        timings = None
        if TIME_CHECK:
            t0 = now()

        # -------- transfer ----------
        x = batch["image"].to(self.device)
        mask = batch["label"].to(self.device)
        if x.ndim == 4:
            x = x.unsqueeze(1)

        if TIME_CHECK:
            t1 = now()

        # -------- forward + loss ----------
        with autocast(device_type=self.device.type, enabled=self.amp):
            pred = self.model(x)
            target = x

            if self.wandb and self.global_step % self.log_image_interval == 0:
                fig = plot_6_middle_slices(
                    image=x[0, 0].cpu(),
                    gt=target[0, 0].cpu(),
                    pred=pred[0, 0].cpu(),
                )
                wandb.log({"Train/Images": wandb.Image(fig)}, step=self.global_step)
                plt.close(fig)

            loss = self.criterion(pred, target, weight=mask)

        if TIME_CHECK:
            t2 = now()

        # -------- backward + optimizer ----------
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if TIME_CHECK:
            t3 = now()
            timings = {
                "transfer": t1 - t0,
                "forward_loss": t2 - t1,
                "backward_step": t3 - t2,
                "total": t3 - t0,
            }

        return loss.item(), timings



    def train_one_epoch(self, epoch: int):
        running_loss = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch}",
            disable=self.tqdm_disable
        )

        if TIME_CHECK:
            sums = {"transfer": 0, "forward_loss": 0, "backward_step": 0, "total": 0}
            last_timings = None

        for i, batch in enumerate(pbar, start=1):
            loss, timings = self.train_step(batch, i, epoch)
            running_loss += loss

            postfix = {'loss': running_loss / i}

            if TIME_CHECK and timings:
                last_timings = timings
                for k in sums:
                    sums[k] += timings[k]

                postfix.update({
                    "t_tot": f"{timings['total']:.3f}",
                    "t_fwd": f"{timings['forward_loss']:.3f}",
                    "t_bwd": f"{timings['backward_step']:.3f}",
                })

            pbar.set_postfix(postfix)

        epoch_loss = running_loss / len(self.train_loader)

        if TIME_CHECK and last_timings:
            n = len(self.train_loader)
            avg = {k: sums[k] / n for k in sums}

            print(f"\n[TimeCheck] Epoch {epoch}")
            print(f"  Avg transfer      : {avg['transfer']:.4f} s")
            print(f"  Avg forward+loss  : {avg['forward_loss']:.4f} s")
            print(f"  Avg backward+step : {avg['backward_step']:.4f} s")
            print(f"  Avg total/batch   : {avg['total']:.4f} s")
            print(f"  Last batch times  : {last_timings}\n")

        return epoch_loss

        
    def validate(self, epoch: int):
        self.model.eval()
        total = 0.0
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x = batch["image"].to(self.device)
                mask = batch["label"].to(self.device)
                if x.ndim == 4:
                    x = x.unsqueeze(1)

                with autocast(device_type=self.device.type, enabled=self.amp):
                    pred = self.model(x)
                    if pred.shape == x.shape:
                        target = x
                    else:
                        target = patchify(x, self.patch_size)

                    loss = self.criterion(pred, target, weight=mask)

                total += loss.item() * x.shape[0]
                count += x.shape[0]

        avg = total / max(1, count)
        print(f"Validation loss (epoch {epoch}): {avg:.6f}")
        return avg


   
    def fit(self):

        if self.wandb:
            wandb.init(project="SpineMAE", config={
                "model_name": self.model_name,
                "in_channels": self.in_channels,
                "img_size": self.img_size,
                "patch_size": self.patch_size,
                "enc_embed_dim": self.enc_embed_dim,
                "enc_num_heads": self.enc_num_heads,
                "enc_layers": self.enc_layers,
                "enc_mlp_dim": self.enc_mlp_dim,
                "dropout": self.dropout,
                "mask_ratio": self.mask_ratio,
                "dec_embed_dim": self.dec_embed_dim,
                "dec_layers": self.dec_layers,
                "dec_num_heads": self.dec_num_heads,
                "dec_mlp_dim": self.dec_mlp_dim,
                "batch_size": self.batch_size,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "epochs": self.epochs,
            })
            wandb.watch(self.model, log="all")

        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self.train_one_epoch(epoch)
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

            if self.wandb:
                wandb.log({
                    'Train/Loss': train_loss,
                    'Val/Loss': val_loss,
                    'Epoch': epoch,
                }, step=self.global_step)
        if self.wandb:
            wandb.finish()
            


