"""
Optimisation des hyperparamètres avec Optuna pour classification_hf.

Prend un fichier de config YAML comme base et surcharge les hyperparamètres
à chaque trial. Supporte tous les chemins (resnet, cached linear).

Usage:
    python -m classification_hf.tune \
        --config classification_hf/configs/rsna_scs_crop4cm_resnet.yaml \
        --output_dir outputs_cls/tune_scs_crop4cm \
        --n_trials 30 --trial_epochs 20 --final_epochs 100 \
        [--seed 42] [--metric auc_ovr_macro]

Le study Optuna est stocké dans output_dir/study.db (SQLite, résumable).
Chaque trial produit output_dir/trial_NNN/...
Le run final est dans output_dir/final/.
Les meilleurs params sont dans output_dir/best_params.json.
"""

import argparse
import json
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from omegaconf import OmegaConf

from .trainer import main as train_main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",        required=True,
                   help="Base YAML config (hyperparams will be overridden per trial)")
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--n_trials",      type=int, default=30)
    p.add_argument("--trial_epochs",  type=int, default=20,
                   help="Epochs per trial (short, for fast HP search)")
    p.add_argument("--final_epochs",  type=int, default=100,
                   help="Epochs for the final run with best params")
    p.add_argument("--metric",        default="val_loss",
                   choices=["val_loss", "wbce", "auc_ovr_macro", "auc_ovr_weighted", "qwk", "f1_macro", "accuracy"],
                   help="Validation metric (val_loss/wbce are minimised, others are maximised)")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--set", nargs="*", default=[],
                   help="Key=value overrides applied to base config, e.g. --set fold_column=regime_all_split_1_set")
    return p.parse_args()


def build_trial_config(base_cfg, trial: optuna.Trial, output_dir: str, epochs: int):
    """Override HP fields in the base config with Optuna suggestions."""
    freeze_bb = bool(OmegaConf.select(base_cfg, "freeze_backbone", default=True))
    bs_choices = [16, 32, 64] if not freeze_bb else [128, 256, 512]
    overrides = {
        "output_dir":   output_dir,
        "epochs":       epochs,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "weight_decay":  trial.suggest_float("weight_decay",  1e-5, 1.0,  log=True),
        "batch_size":    trial.suggest_categorical("batch_size", bs_choices),
        "use_class_weights": trial.suggest_categorical("use_class_weights", [True, False]),
    }

    head_type = OmegaConf.select(base_cfg, "model.head_type", default="linear") or "linear"
    if head_type == "resnet":
        overrides["model"] = OmegaConf.merge(
            base_cfg.model,
            OmegaConf.create({
                "proj_dim": trial.suggest_categorical("proj_dim", [64, 128, 256]),
                "n_blocks": trial.suggest_int("n_blocks", 1, 3),
            }),
        )

    return OmegaConf.merge(base_cfg, OmegaConf.create(overrides))


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = OmegaConf.load(args.config)
    if args.set:
        base_cfg = OmegaConf.merge(base_cfg, OmegaConf.from_dotlist(args.set))

    storage    = f"sqlite:///{out_dir / 'study.db'}"
    study_name = out_dir.name

    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    sampler = TPESampler(seed=args.seed)

    direction = "minimize" if args.metric in ("val_loss", "wbce") else "maximize"

    study = optuna.create_study(
        study_name     = study_name,
        storage        = storage,
        direction      = direction,
        pruner         = pruner,
        sampler        = sampler,
        load_if_exists = True,
    )

    def objective(trial: optuna.Trial) -> float:
        trial_dir = str(out_dir / f"trial_{trial.number:03d}")
        cfg = build_trial_config(base_cfg, trial, trial_dir, args.trial_epochs)
        try:
            result = train_main(cfg, trial=trial)
            return result if result is not None else float("nan")
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            msg = str(e)
            if any(k in msg for k in ("CUDA error", "cudaError", "out of memory", "INT_MAX")):
                raise optuna.exceptions.TrialPruned(f"CUDA: {msg[:120]}")
            raise

    print(f"Study      : {study_name}")
    print(f"Storage    : {storage}")
    print(f"Metric     : {args.metric}  ({direction})")
    print(f"Trials     : {args.n_trials} × {args.trial_epochs} epochs")
    print(f"Final run  : {args.final_epochs} epochs")
    print(f"Output     : {out_dir}\n")

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # ── Best params ───────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\nMeilleur trial : #{best.number}  {args.metric}={best.value:.4f}")
    print("Params :")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    best_params = dict(best.params)
    best_params[f"best_{args.metric}"] = best.value
    (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))

    # ── Final run with best params ────────────────────────────────────────────
    print(f"\nRun final ({args.final_epochs} epochs) avec les meilleurs params…")
    final_cfg = build_trial_config(
        base_cfg,
        _DictTrial(best.params),
        str(out_dir / "final"),
        args.final_epochs,
    )
    final_metric = train_main(final_cfg)
    best_params[f"final_{args.metric}"] = final_metric
    (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))

    print(f"\nTerminé. Final {args.metric}={final_metric:.4f}")
    print(f"Params : {out_dir / 'best_params.json'}")


class _DictTrial:
    """Thin wrapper so build_trial_config works with a plain dict (final run)."""
    def __init__(self, params: dict):
        self._p = params

    def suggest_float(self, name, *args, **kwargs):
        return self._p[name]

    def suggest_categorical(self, name, *args, **kwargs):
        return self._p[name]

    def suggest_int(self, name, *args, **kwargs):
        return self._p[name]


if __name__ == "__main__":
    main()
