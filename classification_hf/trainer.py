"""
Classification trainer — mirrors curia/trainer.py exactly, adapted for local data.

Instead of load_dataset("raidium/CuriaBench", ...) we build a DatasetDict from a
local directory tree.  Everything else (Classifier, SGD, LR scaling, HF Trainer,
feature caching, metrics) is a direct copy of the curia pipeline.
"""

from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoImageProcessor, Dinov2Model, Trainer, TrainingArguments

from .dataset import extract_features_fn, load_local_dataset, preprocess_function
from .model import Classifier


# ── Metrics (verbatim from curia/trainer.py) ─────────────────────────────────


def _extract_predictions_and_labels(eval_pred):
    if hasattr(eval_pred, "predictions"):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    return predictions, labels


def compute_classification_metrics(eval_pred):
    logits, labels = _extract_predictions_and_labels(eval_pred)
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)

    proba = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    if proba.shape[1] == 2:
        proba = proba[:, 1]

    try:
        auc = roc_auc_score(labels, proba, multi_class="ovr")
    except ValueError:
        auc = float("nan")

    return {"accuracy": acc, "auc": auc}


# ── LR scaling (verbatim from curia/trainer.py) ───────────────────────────────


def scale_lr(learning_rate: float, batch_size: int) -> float:
    return learning_rate * batch_size / 256.0


# ── Save head (verbatim from curia/trainer.py) ────────────────────────────────


def save_head(model, output_dir: str):
    output_path = Path(output_dir) / "head.pt"
    output_path.parent.mkdir(exist_ok=True, parents=True)
    payload = {"classifier": model.linear.state_dict()}
    if model.attention_module is not None:
        payload["attention"] = model.attention_module.state_dict()
    torch.save(payload, output_path)
    print(f"Saved classifier head to {output_path}")


# ── instantiate_cache_model_and_dataset (adapted from curia) ─────────────────


def instantiate_cache_model_and_dataset(config, train_dataset, val_dataset):
    """
    Extract CLS tokens with the frozen backbone (once), cache them in the HF
    Dataset, then build a Classifier that trains on those cached features.
    Mirrors curia's instantiate_cache_model_and_dataset for the 2-D no-mask case.
    """
    model_name = config.model.model_name
    backbone = Dinov2Model.from_pretrained(model_name)
    backbone.cuda()
    backbone.eval()
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

    _extract = partial(extract_features_fn, processor=processor, backbone=backbone)

    train_dataset = train_dataset.map(
        _extract, batched=True, batch_size=config.batch_size, num_proc=0
    )
    val_dataset = val_dataset.map(
        _extract, batched=True, batch_size=config.batch_size, num_proc=0
    )

    attention_cfg = OmegaConf.select(config, "model.attention_cfg")
    model = Classifier(
        backbone.config.hidden_size,
        config.model.num_classes,
        regression=False,
        attention_cfg=attention_cfg,
    )
    nn.init.normal_(model.linear.weight, mean=0.0, std=0.01)
    nn.init.zeros_(model.linear.bias)

    for split in [train_dataset, val_dataset]:
        split.set_format(type="torch", columns=["pixel_values", "labels"])

    return model, train_dataset, val_dataset


# ── instantiate_model_and_dataset (full backbone, no caching) ────────────────


def instantiate_model_and_dataset(config, train_dataset, val_dataset):
    """
    Pre-process images with the curia AutoImageProcessor and fine-tune with the
    full backbone frozen — no feature caching.
    """
    from transformers import AutoModelForImageClassification

    model_name = config.model.model_name
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

    _preprocess = partial(preprocess_function, processor=processor)

    train_dataset = train_dataset.map(
        _preprocess, batched=True, batch_size=config.batch_size,
        num_proc=config.num_workers,
    )
    val_dataset = val_dataset.map(
        _preprocess, batched=True, batch_size=config.batch_size,
        num_proc=config.num_workers,
    )

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=config.model.num_classes,
        ignore_mismatched_sizes=True,
        trust_remote_code=True,
    )
    model.base_model.requires_grad_(False)

    for split in [train_dataset, val_dataset]:
        split.set_format(type="torch", columns=["pixel_values", "labels"])

    return model, train_dataset, val_dataset


# ── Main entry point (mirrors curia/trainer.py main()) ───────────────────────


def main(config) -> None:
    # ── Load local dataset ────────────────────────────────────────────────────
    ds = load_local_dataset(
        config.data_dir,
        val_split=float(OmegaConf.select(config, "val_split", default=0.15)),
        seed=int(OmegaConf.select(config, "seed", default=42)),
    )
    train_dataset = ds["train"]
    val_dataset   = ds["val"]

    print(f"Dataset loaded from {config.data_dir}")
    print(f"  train: {len(train_dataset)}  val: {len(val_dataset)}")

    # ── Build model + preprocessed datasets ───────────────────────────────────
    use_feature_caching = bool(OmegaConf.select(config, "use_feature_caching", default=False))
    if use_feature_caching:
        model, train_dataset, val_dataset = instantiate_cache_model_and_dataset(
            config, train_dataset, val_dataset
        )
    else:
        model, train_dataset, val_dataset = instantiate_model_and_dataset(
            config, train_dataset, val_dataset
        )

    # ── Optimizer + scheduler (verbatim from curia) ───────────────────────────
    steps_per_epoch = max(1, len(train_dataset) // config.batch_size)
    max_steps = steps_per_epoch * config.epochs

    scaled_lr = scale_lr(config.learning_rate, config.batch_size)
    print(f"LR: {config.learning_rate} → scaled: {scaled_lr:.6f}  (batch_size={config.batch_size})")

    optimizer = SGD(model.parameters(), lr=scaled_lr, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0)

    # ── HF TrainingArguments (verbatim from curia) ────────────────────────────
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        logging_strategy="steps",
        logging_steps=max(10, steps_per_epoch // 10),
        save_strategy="no",
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        dataloader_num_workers=config.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_classification_metrics,
        optimizers=(optimizer, scheduler),
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    save_head(trainer.model, config.output_dir)

    print("\n--- Val Set Evaluation ---")
    val_results = trainer.evaluate(eval_dataset=val_dataset)
    print(val_results)
    with (Path(config.output_dir) / "val_results.txt").open("w") as f:
        f.write(str(val_results))
