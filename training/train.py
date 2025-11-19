
import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from SpineFoundation.model.build import build_model, list_models
from SpineFoundation.training.data import build_dataloaders
from SpineFoundation.training.utils import patchify, save_checkpoint, load_checkpoint
import json


def train(args):
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')


    enc_params = dict(
        in_channels=args.in_channels,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.enc_layers,
        mlp_dim=args.enc_mlp_dim,
        dropout_rate=args.dropout,
    )
    dec_params = dict(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        decoder_embed_dim=args.dec_embed_dim,
        num_layers=args.dec_layers,
        num_heads=args.dec_num_heads,
        in_channels=args.in_channels,
    )

    # merge with JSON params if provided
    if args.encoder_params:
        try:
            user_enc = json.loads(args.encoder_params)
            if not isinstance(user_enc, dict):
                raise ValueError('encoder_params must be a JSON object')
            enc_params.update(user_enc)
        except Exception as e:
            raise RuntimeError(f'Failed to parse --encoder-params: {e}')
    if args.decoder_params:
        try:
            user_dec = json.loads(args.decoder_params)
            if not isinstance(user_dec, dict):
                raise ValueError('decoder_params must be a JSON object')
            dec_params.update(user_dec)
        except Exception as e:
            raise RuntimeError(f'Failed to parse --decoder-params: {e}')

    encoder = build_model(args.encoder_name, enc_params)
    decoder = build_model(args.decoder_name, dec_params)

    encoder.to(device)
    decoder.to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    criterion = nn.L1Loss()

    train_loader, val_loader = build_dataloaders(args.img_size, args.batch_size, num_workers=args.num_workers)

    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, device)
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
        opt.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1

    best_val = float('inf')
    for epoch in range(start_epoch, args.epochs):
        encoder.train(); decoder.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        for batch in pbar:
            x = batch.to(device).unsqueeze(1) if batch.ndim == 4 else batch.to(device)
            with autocast(enabled=args.amp):
                z, ids_restore = encoder(x, mask_ratio=args.mask_ratio)
                pred = decoder.forward(z, ids_restore, return_patches=True)
                target = patchify(x, args.patch_size)
                loss = criterion(pred, target)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

        # validation
        val_loss = validate(encoder, decoder, val_loader, criterion, device, args)
        is_best = val_loss < best_val
        best_val = min(best_val, val_loss)

        ckpt = {'epoch': epoch, 'encoder': encoder.state_dict(), 'decoder': decoder.state_dict(), 'optimizer': opt.state_dict(), 'val_loss': val_loss}
        save_checkpoint(ckpt, os.path.join(args.work_dir, f'ckpt_epoch_{epoch}.pt'))
        if is_best:
            save_checkpoint(ckpt, os.path.join(args.work_dir, 'best.ckpt'))


def validate(encoder, decoder, val_loader, criterion, device, args):
    encoder.eval(); decoder.eval()
    total = 0.0; count = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch.to(device).unsqueeze(1) if batch.ndim == 4 else batch.to(device)
            z, ids_restore = encoder(x, mask_ratio=args.mask_ratio)
            pred = decoder.forward(z, ids_restore, return_patches=True)
            target = patchify(x, args.patch_size)
            loss = criterion(pred, target)
            total += loss.item() * x.shape[0]
            count += x.shape[0]
    avg = total / max(1, count)
    print(f"Validation loss: {avg:.6f}")
    return avg


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--encoder-name', type=str, default='spine_encoder', help='Model name for encoder (registry)')
    p.add_argument('--decoder-name', type=str, default='spine_decoder', help='Model name for decoder (registry)')
    p.add_argument('--encoder-params', type=str, default='', help='JSON string of encoder constructor params')
    p.add_argument('--decoder-params', type=str, default='', help='JSON string of decoder constructor params')
    p.add_argument('--img-size', nargs=3, type=int, default=(32,32,32))
    p.add_argument('--patch-size', nargs=3, type=int, default=(8,8,8))
    p.add_argument('--in-channels', type=int, default=1)
    p.add_argument('--embed-dim', type=int, default=64)
    p.add_argument('--enc-layers', type=int, default=2)
    p.add_argument('--enc-mlp-dim', type=int, default=128)
    p.add_argument('--num-heads', type=int, default=4)
    p.add_argument('--dec-embed-dim', type=int, default=32)
    p.add_argument('--dec-layers', type=int, default=1)
    p.add_argument('--dec-num-heads', type=int, default=4)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-2)
    p.add_argument('--mask-ratio', type=float, default=0.5)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--no-cuda', action='store_true')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--work-dir', type=str, default='./training_runs')
    p.add_argument('--num-workers', type=int, default=2)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.img_size = tuple(map(int, args.img_size))
    args.patch_size = tuple(map(int, args.patch_size))
    os.makedirs(args.work_dir, exist_ok=True)
    train(args)
