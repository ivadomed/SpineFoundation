"""
Optimisation des hyperparamètres avec Optuna pour segmentation_hf.

Usage:
    python -m segmentation_hf.tune \
        --model_dir   /path/to/curia \
        --npz_train_dir segmentation_hf/data/fold_0/train_npz \
        --npz_val_dir   segmentation_hf/data/fold_0/val_npz \
        --output_dir    outputs_seg/tune_fold0 \
        --n_trials 30 --trial_epochs 30 --final_epochs 300 \
        [--amp] [--batch_size 256] [--num_workers 4] [--seed 42]

Le study Optuna est stocké dans output_dir/study.db (SQLite).
Chaque trial produit output_dir/trial_NNN/{best.pt, history.csv}.
Le run final est dans output_dir/final/.
Les meilleurs params sont dans output_dir/best_params.json.
"""

import argparse
import gc
import json
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from .config import TrainConfig
from .trainer import train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",     required=True)
    p.add_argument("--npz_train_dir", required=True)
    p.add_argument("--npz_val_dir",   required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--n_trials",      type=int, default=30)
    p.add_argument("--trial_epochs",  type=int, default=30)
    p.add_argument("--final_epochs",  type=int, default=300)
    p.add_argument("--batch_size",    type=int, default=256)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--image_size",    type=int, default=224)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--amp",           action="store_true")
    p.add_argument("--patch_token_key", default="patch_tokens")
    p.add_argument("--in_channels", type=int, default=None,
                   help="Override backbone hidden size for non-HF backbones (e.g. MRICore=256).")
    return p.parse_args()


def build_trial_cfg(trial: optuna.Trial, args: argparse.Namespace, epochs: int, output_dir: str) -> TrainConfig:
    bce_weight = trial.suggest_float("bce_weight", 0.1, 0.9)

    return TrainConfig(
        model_dir       = args.model_dir,
        train_images    = "",
        train_masks     = "",
        val_images      = "",
        val_masks       = "",
        npz_train_dir   = args.npz_train_dir,
        npz_val_dir     = args.npz_val_dir,
        patch_token_key = args.patch_token_key,
        in_channels     = args.in_channels,
        output_dir      = output_dir,
        image_size      = args.image_size,
        epochs          = epochs,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        seed            = args.seed,
        amp             = args.amp,
        save_every      = epochs + 1,   # ne pas sauvegarder pendant les trials (sauf best)
        skip_train_eval = True,
        augment         = True,
        use_scheduler   = True,
        use_wandb       = False,
        wandb_mode      = "disabled",
        seg_head_channels = trial.suggest_categorical("seg_head_channels", [64, 128, 256]),
        seg_head_depth    = trial.suggest_int("seg_head_depth", 2, 4),
        seg_head_dropout  = trial.suggest_float("seg_head_dropout", 0.0, 0.3),
        seg_head_norm     = trial.suggest_categorical("seg_head_norm", ["batch", "group", "instance"]),
        seg_head_nonlin   = trial.suggest_categorical("seg_head_nonlin", ["gelu", "leakyrelu"]),
        lr                = trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        weight_decay      = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        bce_weight        = bce_weight,
        dice_weight       = round(1.0 - bce_weight, 6),
        bce_pos_weight    = trial.suggest_float("bce_pos_weight", 0.1, 10.0, log=True),
    )


def main() -> None:
    args     = parse_args()
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    storage  = f"sqlite:///{out_dir / 'study.db'}"
    study_name = out_dir.name

    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    sampler = TPESampler(seed=args.seed)

    study = optuna.create_study(
        study_name  = study_name,
        storage     = storage,
        direction   = "maximize",
        pruner      = pruner,
        sampler     = sampler,
        load_if_exists = True,
    )

    def objective(trial: optuna.Trial) -> float:
        trial_dir = str(out_dir / f"trial_{trial.number:03d}")
        cfg = build_trial_cfg(trial, args, args.trial_epochs, trial_dir)
        try:
            result = train(cfg, trial=trial)
        except (RuntimeError, Exception) as e:
            msg = str(e)
            if any(k in msg for k in ("INT_MAX", "elements", "CUDA error", "cudaError", "invalid configuration")):
                raise optuna.exceptions.TrialPruned(f"CUDA: {msg[:120]}")
            raise
        finally:
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        return result

    print(f"Study         : {study_name}")
    print(f"Storage       : {storage}")
    print(f"Trials        : {args.n_trials} × {args.trial_epochs} epochs")
    print(f"Final run     : {args.final_epochs} epochs")
    print(f"Output        : {out_dir}")

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # ── Best params ───────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\nMeilleur trial : #{best.number}  val_dice={best.value:.4f}")
    print("Params :")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    best_params = dict(best.params)
    best_params["val_dice"] = best.value
    (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))

    # ── Final run with best params ────────────────────────────────────────────
    print(f"\nRun final ({args.final_epochs} epochs) avec les meilleurs params…")
    bce_w = best.params["bce_weight"]

    final_cfg = TrainConfig(
        model_dir       = args.model_dir,
        train_images    = "",
        train_masks     = "",
        val_images      = "",
        val_masks       = "",
        npz_train_dir   = args.npz_train_dir,
        npz_val_dir     = args.npz_val_dir,
        patch_token_key = args.patch_token_key,
        in_channels     = args.in_channels,
        output_dir      = str(out_dir / "final"),
        image_size      = args.image_size,
        epochs          = args.final_epochs,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        seed            = args.seed,
        amp             = args.amp,
        save_every      = args.final_epochs + 1,
        skip_train_eval = True,
        augment         = True,
        use_wandb       = False,
        wandb_mode      = "disabled",
        seg_head_channels = best.params["seg_head_channels"],
        seg_head_depth    = best.params["seg_head_depth"],
        seg_head_dropout  = best.params["seg_head_dropout"],
        seg_head_norm     = best.params["seg_head_norm"],
        seg_head_nonlin   = best.params["seg_head_nonlin"],
        lr                = best.params["lr"],
        weight_decay      = best.params["weight_decay"],
        bce_weight        = bce_w,
        dice_weight       = round(1.0 - bce_w, 6),
        bce_pos_weight    = best.params["bce_pos_weight"],
    )

    final_dice = train(final_cfg)
    best_params["final_val_dice"] = final_dice
    (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))

    print(f"\nTerminé. Final val_dice={final_dice:.4f}")
    print(f"Modèle final : {out_dir / 'final' / 'best.pt'}")
    print(f"Params       : {out_dir / 'best_params.json'}")


if __name__ == "__main__":
    main()
