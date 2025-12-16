import os
import time
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from src.data_loader import load_train_test_data
from src.eval_utils import (
    classification_report,
    plot_confusion_matrix,
    plot_pr_curve,
    save_metrics_json,
    save_predictions_csv,
)
from src.features_cheap import extract_features


BASE_MODEL_DIR = os.path.join("experiments", "models", "c3_distilbert_lora")
BASE_MODEL_NAME = "distilbert-base-uncased"


def stage1_rule(title: str) -> bool:
    """Return True if rule says it's wrong (label 0)."""
    feats = extract_features(title)
    len_chars_raw = len(str(title))
    digit_ratio = float(feats[2])
    punct_ratio = float(feats[3])
    contains_noise_kw = bool(feats[7] > 0)
    len_tokens = float(feats[1])
    return (
        (digit_ratio > 0.25 and len_chars_raw > 20)
        or (punct_ratio > 0.20)
        or contains_noise_kw
        or (len_tokens > 25)
    )


def load_lora_model(model_dir: str = BASE_MODEL_DIR, base_model: str = BASE_MODEL_NAME):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"LoRA model dir not found: {model_dir}. Please run src.models_distilbert_lora first."
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    backbone = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
    model = PeftModel.from_pretrained(backbone, model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_stage2(
    texts: List[str],
    tokenizer,
    model,
    device,
    max_length: int = 64,
    batch_size: int = 32,
) -> Tuple[List[int], List[float]]:
    preds: List[int] = []
    probs: List[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            prob = torch.softmax(logits, dim=-1)
            pred = prob.argmax(dim=-1)
        preds.extend(pred.cpu().tolist())
        probs.extend(prob[:, 1].cpu().tolist())
    return preds, probs


def two_stage_predict():
    X_train, y_train, X_test, y_test = load_train_test_data()
    tokenizer, model, device = load_lora_model()

    start_total = time.perf_counter()
    # Stage 1
    stage1_flags = [stage1_rule(t) for t in X_test]
    stage1_indices = [i for i, flag in enumerate(stage1_flags) if flag]
    stage2_indices = [i for i, flag in enumerate(stage1_flags) if not flag]

    stage1_preds = [0] * len(stage1_indices)  # rule marks as wrong
    stage1_labels = [y_test[i] for i in stage1_indices]
    stage1_titles = [X_test[i] for i in stage1_indices]

    # Stage 2
    stage2_titles = [X_test[i] for i in stage2_indices]
    stage2_labels = [y_test[i] for i in stage2_indices]

    infer_start = time.perf_counter()
    stage2_preds, stage2_probs = predict_stage2(stage2_titles, tokenizer, model, device)
    infer_time = time.perf_counter() - infer_start

    # Merge results
    all_preds = [0] * len(X_test)
    all_probs = [0.0] * len(X_test)

    for idx, pred in zip(stage1_indices, stage1_preds):
        all_preds[idx] = pred
        all_probs[idx] = 0.99  # confident rule

    for local_idx, (pred, prob) in enumerate(zip(stage2_preds, stage2_probs)):
        idx = stage2_indices[local_idx]
        all_preds[idx] = pred
        all_probs[idx] = prob

    total_time = time.perf_counter() - start_total

    # Metrics
    coverage = len(stage1_indices) / len(X_test) if X_test else 0.0
    stage1_metrics = (
        classification_report(stage1_labels, stage1_preds)
        if stage1_indices
        else {"accuracy": 0.0, "macro_f1": 0.0, "micro_f1": 0.0}
    )
    overall_metrics = classification_report(y_test, all_preds, all_probs)

    results_dir = os.path.join("experiments", "results")
    os.makedirs(results_dir, exist_ok=True)

    metrics_payload: Dict[str, float] = {
        "stage1_coverage": coverage,
        "stage1_accuracy": stage1_metrics.get("accuracy", 0.0),
        "stage1_macro_f1": stage1_metrics.get("macro_f1", 0.0),
        "stage1_micro_f1": stage1_metrics.get("micro_f1", 0.0),
        **{f"overall_{k}": v for k, v in overall_metrics.items()},
    }
    save_metrics_json(metrics_payload, os.path.join(results_dir, "c3_two_stage_metrics.json"))
    save_metrics_json(
        {
            "total_time_sec": total_time,
            "inference_time_stage2_sec": infer_time,
            "titles_per_sec_overall": len(X_test) / total_time if total_time > 0 else 0.0,
            "titles_per_sec_stage2": len(stage2_titles) / infer_time if infer_time > 0 else 0.0,
        },
        os.path.join(results_dir, "c3_two_stage_timing.json"),
    )

    save_predictions_csv(
        X_test,
        y_test,
        all_preds,
        all_probs,
        os.path.join(results_dir, "c3_two_stage_predictions.csv"),
    )

    plot_confusion_matrix(
        y_test, all_preds, path=os.path.join("experiments", "plots", "c3_two_stage_confusion.png")
    )
    plot_pr_curve(
        y_test, all_probs, path=os.path.join("experiments", "plots", "c3_two_stage_pr.png")
    )

    print(f"[INFO] Stage1 coverage: {coverage:.3f}")
    print("[INFO] Overall metrics saved to experiments/results/c3_two_stage_metrics.json")


if __name__ == "__main__":
    two_stage_predict()
