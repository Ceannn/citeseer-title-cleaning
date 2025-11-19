import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from src.data_loader import load_train_test_data


# ======================
# 1. Dataset 定义：按需编码版（简单稳定）
# ======================

class TitleDataset(Dataset):
    """
    按需调用 tokenizer 的 Dataset：
    - 保存原始文本和 label
    - 在 __getitem__ 里调用 tokenizer
    """
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 64):
        self.texts = [str(t) for t in texts]
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",      # 固定长度，简单稳定
            max_length=self.max_length,
            return_tensors="pt",
        )

        # enc 里每个值形状是 (1, L)，去掉 batch 维度
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# ======================
# 2. metrics 计算
# ======================

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ======================
# 3. 主流程：微调 + 测试集评估
# ======================

def train_and_evaluate_bert(
    model_name: str = "bert-base-uncased",
    max_length: int = 64,
    num_epochs: int = 1,
    train_sample_cap: Optional[int] = None,
):
    """
    微调 BERT 做标题二分类：
    - 训练数据来自 positive/negative 文本
    - 测试数据来自 testSet.xlsx
    - 可选用 train_sample_cap 限制训练样本（方便快速实验）
    """

    # 3.1 加载数据
    X_train_all, y_train_all, X_test, y_test = load_train_test_data()
    print(f"[BERT] Total train samples: {len(X_train_all)}, test samples: {len(X_test)}")

    # 可选：限制训练样本（调参/开发模式）
    if train_sample_cap is not None and len(X_train_all) > train_sample_cap:
        from sklearn.utils import shuffle
        X_train_all, y_train_all = shuffle(X_train_all, y_train_all, random_state=42)
        X_train_all = X_train_all[:train_sample_cap]
        y_train_all = y_train_all[:train_sample_cap]
        print(f"[BERT] Using subset of train data: {len(X_train_all)} samples")

    # train / val 切分
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all,
        y_train_all,
        test_size=0.1,
        random_state=42,
        stratify=y_train_all,
    )

    print(f"[BERT] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 3.2 tokenizer & 模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("[BERT] Using device:", device)
    print("[DEBUG] Model first param device:", next(model.parameters()).device)

    # 3.3 构造 Dataset
    train_dataset = TitleDataset(X_train, y_train, tokenizer, max_length=max_length)
    val_dataset = TitleDataset(X_val, y_val, tokenizer, max_length=max_length)
    test_dataset = TitleDataset(X_test, y_test, tokenizer, max_length=max_length)

    # 3.4 TrainingArguments
    output_dir = os.path.join("experiments", "bert_finetune")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",          # 你的 transformers 版本支持 eval_strategy
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],                   # 不用 wandb / tensorboard
        fp16=True if device.type == "cuda" else False,  # GPU 时开启混合精度
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
    )

    # 3.5 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 3.6 训练
    print("[BERT] Start training...")
    trainer.train()

    # 3.7 在测试集上评估
    print("[BERT] Evaluating on test set...")
    preds_output = trainer.predict(test_dataset)
    logits = preds_output.predictions
    y_pred = np.argmax(logits, axis=-1)

    print("\n=== [BERT Fine-tune] Classification Report (Test Set) ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("=== [BERT Fine-tune] Confusion Matrix (Test Set) ===")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    # 开发调试：可以先用子集，比如 40000
    # 最终实验：把 train_sample_cap 改成 None 跑全量
    train_and_evaluate_bert(
        model_name="bert-base-uncased",
        max_length=64,
        num_epochs=1,
        train_sample_cap=40000,   # 先小样本验证没问题，再全量
    )
