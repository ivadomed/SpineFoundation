import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from .model.build import build_model
from .training.data import build_dataloaders
from .training.utils import patchify, save_checkpoint, load_checkpoint


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')

        model_params = dict(
            in_channels=args.in_channels,
            img_size=args.img_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.enc_layers,
            mlp_dim=args.enc_mlp_dim,
            dropout_rate=args.dropout,
            mask_ratio=args.mask_ratio,
            decoder_embed_dim=args.dec_embed_dim,
            decoder_num_layers=args.dec_layers,
            decoder_num_heads=args.dec_num_heads,
            decoder_mlp_dim=args.dec_mlp_dim)

        data_params = dict(
            img_size=args.img_size,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            work_dir=args.work_dir)

        if args.data_params:
            try:
                user_data = json.loads(args.data_params)
                data_params.update(user_data)
            except Exception as e:
                raise RuntimeError(f'Failed to parse --data-params: {e}')

        if args.model_params:
            try:
                user_params = json.loads(args.model_params)
                model_params.update(user_params)
            except Exception as e:
                raise RuntimeError(f'Failed to parse --model-params: {e}')

        self.model = build_model(args.model_name, model_params)
        self.model.to(self.device)


        self.optimizer = AdamW(self.model.parameters(),lr=args.lr,weight_decay=args.weight_decay,)
        self.scaler = GradScaler(enabled=args.amp)
        self.criterion = nn.L1Loss()
        folders = list_child_folders(args.data_path)
        splits=(data_params["train_ratio"], data_params["val_ratio"], data_params["test_ratio"])
        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
                                                                img_size=data_params["img_size"],
                                                                batch_size=data_params["batch_size"],
                                                                num_workers=data_params["num_workers"],
                                                                shuffle_seed=data_params["seed"],
                                                                splits=splits,
                                                            )
        
        self.start_epoch = 0
        self.best_val = float('inf')
        if args.resume:
            ckpt = load_checkpoint(args.resume, self.device)
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.start_epoch = ckpt.get('epoch', 0) + 1
            self.best_val = ckpt.get('val_loss', float('inf'))
            print(f"Resumed from {args.resume} at epoch {self.start_epoch}")


    def train_step(self, batch):
        self.model.train()

        x = batch.to(self.device)
        if x.ndim == 4:  # (B, D, H, W) -> (B, 1, D, H, W)
            x = x.unsqueeze(1)

        with autocast(enabled=self.args.amp):
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
                x = batch.to(self.device)
                if x.ndim == 4:
                    x = x.unsqueeze(1)

                with autocast(enabled=self.args.amp):
                    pred = self.model(x)
                    if pred.shape == x.shape:
                        target = x
                    else:
                        target = patchify(x, self.args.patch_size)

                    loss = self.criterion(pred, target)

                total += loss.item() * x.shape[0]
                count += x.shape[0]

        avg = total / max(1, count)
        print(f"Validation loss (epoch {epoch}): {avg:.6f}")
        return avg

   
    def fit(self):
        for epoch in range(self.start_epoch, self.args.epochs):
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
                os.path.join(self.args.work_dir, f'ckpt_epoch_{epoch}.pt'),
            )
            if is_best:
                save_checkpoint(
                    ckpt,
                    os.path.join(self.args.work_dir, 'best.ckpt'),
                )


