import argparse
import json
import math
import os
import tempfile
from copy import deepcopy

import matplotlib.pyplot as plt
from training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-params", type=str, required=True)
    p.add_argument("--data-params", type=str, required=True)
    p.add_argument("--training-params", type=str, required=True)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--lr-max", type=float, default=1e-2)
    p.add_argument("--num-lr", type=int, default=7)
    p.add_argument("--epochs-per-lr", type=int, default=4)
    return p.parse_args()


def generate_lrs(lr_min: float, lr_max: float, num_lr: int):
    if num_lr == 1:
        return [lr_min]
    log_min = math.log10(lr_min)
    log_max = math.log10(lr_max)
    step = (log_max - log_min) / (num_lr - 1)
    return [10 ** (log_min + i * step) for i in range(num_lr)]


def run_for_lr(base_args, base_train_cfg, lr_value, epochs_per_lr: int):
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"training_params_lr_{lr_value:.1e}.json")

    cfg = deepcopy(base_train_cfg)
    cfg["lr"] = lr_value

    for key in ["epochs"]:
        if key in cfg:
            cfg[key] = epochs_per_lr

    with open(tmp_path, "w") as f:
        json.dump(cfg, f, indent=2)

    run_args = argparse.Namespace(
        model_params=base_args.model_params,
        data_params=base_args.data_params,
        training_params=tmp_path,
    )

    trainer = Trainer(run_args)
    best_val = trainer.fit()
    return best_val


def main():
    args = parse_args()

    with open(args.training_params, "r") as f:
        base_train_cfg = json.load(f)

    lrs = generate_lrs(args.lr_min, args.lr_max, args.num_lr)
    results = []

    for lr in lrs:
        print(f"LR = {lr:.2e}")
        best_val = run_for_lr(args, base_train_cfg, lr, args.epochs_per_lr)
        results.append((lr, best_val))
        print(f"best_val = {best_val}")

    results_sorted = sorted(results, key=lambda t: t[1])
    best_lr, best_val = results_sorted[0]
    lr_up = best_lr
    lr_down = best_lr / 10

    print("\nRésumé trié :")
    for lr, val in results_sorted:
        print(f"{lr:.2e}\t{val}")

    print("\nlr_up   =", lr_up)
    print("lr_down =", lr_down)

    lrs_plot = [lr for lr, _ in results]
    vals_plot = [v for _, v in results]

    plt.figure(figsize=(6, 4))
    plt.semilogx(lrs_plot, vals_plot, marker="o")
    plt.scatter([best_lr], [best_val], s=60)
    plt.xlabel("Learning rate (log scale)")
    plt.ylabel("best_val")
    plt.title("LR sweep")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("SpineFoundation/scripts/lr_sweep2.png", dpi=150)

    print("\nPlot sauvegardé : lr_sweep.png")


if __name__ == "__main__":
    main()
