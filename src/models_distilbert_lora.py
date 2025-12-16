import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, TaskType, get_peft_model

from src.data_loader import load_train_test_data
from src.eval_utils import (
    classification_report,
    count_parameters,
    plot_confusion_matrix,
    plot_pr_curve,
    save_metrics_json,
    save_predictions_csv,
    set_seed,
)


DEFAULT_MODEL_NAME = "distilbert-base-uncased"


@dataclass
class TrainConfig:
    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = 64
    batch_size: int = 32
    lr: float = 2e-4
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    seed: int = 42


class TitleDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = [str(t) for t in texts]
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def _apply_lora(model, lora_cfg: TrainConfig):
    target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"]
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_cfg.lora_r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    try:
        lora_model = get_peft_model(model, config)
        return lora_model
    except ValueError as e:
        print("[WARN] LoRA target modules did not match. Available module names:")
        names = sorted(set([n.split(".")[-1] for n, _ in model.named_modules() if hasattr(_, "weight")]))
        print(names)
        raise e


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.from_numpy(logits)
    preds = probs.argmax(dim=-1).cpu().numpy()
    probs = torch.softmax(probs, dim=-1).cpu().numpy()[:, 1]
    metrics_dict = classification_report(labels, preds, probs)
    # Trainer expects simple scalars
    return {
        "accuracy": metrics_dict["accuracy"],
        "macro_f1": metrics_dict["macro_f1"],
    }


def train_and_evaluate(cfg: Optional[TrainConfig] = None):
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)

    X_train_all, y_train_all, X_test, y_test = load_train_test_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all,
        y_train_all,
        test_size=0.1,
        random_state=cfg.seed,
        stratify=y_train_all,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=2
    )
    model = _apply_lora(base_model, cfg)

    train_dataset = TitleDataset(X_train, y_train, tokenizer, cfg.max_length)
    val_dataset = TitleDataset(X_val, y_val, tokenizer, cfg.max_length)
    test_dataset = TitleDataset(X_test, y_test, tokenizer, cfg.max_length)

    output_dir = os.path.join("experiments", "models", "c3_distilbert_lora")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    train_start = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - train_start
    # ensure adapter/tokenizer saved
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("[INFO] Evaluating on test set...")
    test_start = time.perf_counter()
    preds_output = trainer.predict(test_dataset)
    infer_time = time.perf_counter() - test_start

    logits = preds_output.predictions
    prob_pos = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()[:, 1].tolist()
    pred_labels = np.argmax(logits, axis=-1).tolist()

    metrics_dict = classification_report(y_test, pred_labels, prob_pos)
    params_info = count_parameters(model)
    timing = {
        "train_time_sec": train_time,
        "inference_time_sec": infer_time,
        "titles_per_sec": len(X_test) / infer_time if infer_time > 0 else 0.0,
        **params_info,
    }

    results_dir = os.path.join("experiments", "results")
    os.makedirs(results_dir, exist_ok=True)
    save_metrics_json(metrics_dict, os.path.join(results_dir, "c3_distilbert_lora_metrics.json"))
    save_predictions_csv(
        X_test, y_test, pred_labels, prob_pos, os.path.join(results_dir, "c3_distilbert_lora_predictions.csv")
    )

    timing_path = os.path.join(results_dir, "c3_distilbert_lora_timing.json")
    from src.eval_utils import save_metrics_json as _save_json  # reuse to ensure dirs

    _save_json(timing, timing_path)

    plot_confusion_matrix(
        y_test, pred_labels, path=os.path.join("experiments", "plots", "c3_distilbert_lora_confusion.png")
    )
    plot_pr_curve(
        y_test, prob_pos, path=os.path.join("experiments", "plots", "c3_distilbert_lora_pr.png")
    )

    print("[INFO] Metrics saved to", os.path.join(results_dir, "c3_distilbert_lora_metrics.json"))
    print("[INFO] Predictions saved to", os.path.join(results_dir, "c3_distilbert_lora_predictions.csv"))
    print("[INFO] Timing saved to", timing_path)
    return metrics_dict, timing


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="C3-Base: DistilBERT + LoRA fine-tune")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
    )
    train_and_evaluate(cfg)
