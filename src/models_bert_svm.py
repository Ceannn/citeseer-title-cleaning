"""
Train classic classifiers on frozen BERT embeddings for title classification.

No fine-tuning is performed: BERT is only used as a feature extractor.
Supports multiple pooling strategies and classifier families for easy comparison
with Naive Bayes / Word2Vec + SVM baselines.
"""
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import classification_report as sk_cls_report, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_train_test_data
from src.eval_utils import classification_report as eval_cls_report
from src.features_bert import BERTEmbedder, EmbedStrategy


ClassifierName = str


def _build_classifier(name: ClassifierName):
    name = name.lower()
    if name == "linear_svm":
        return LinearSVC()
    if name == "logreg":
        return LogisticRegression(max_iter=200, n_jobs=-1, solver="liblinear")
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
        )
    raise ValueError(f"Unsupported classifier: {name}")


def _print_metrics(y_true: List[int], y_pred: List[int], title: str) -> Dict[str, float]:
    print(f"\n=== {title} ===")
    print(sk_cls_report(y_true, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    return eval_cls_report(y_true, y_pred)


def train_and_evaluate(
    embed_strategy: EmbedStrategy = "cls",
    classifier: ClassifierName = "linear_svm",
    model_name: str = "bert-base-uncased",
    max_length: int = 64,
    batch_size: int = 32,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Encode titles with frozen BERT, then train a classic classifier and evaluate on test set.
    Returns (metrics_dict, shapes) for easy logging.
    """
    X_train, y_train, X_test, y_test = load_train_test_data()
    encoder = BERTEmbedder(model_name=model_name, max_length=max_length)

    print(f"[ENC] Using strategy={embed_strategy}, model={model_name}, device={encoder.device}")
    X_train_vec = encoder.encode(X_train, strategy=embed_strategy, batch_size=batch_size)
    X_test_vec = encoder.encode(X_test, strategy=embed_strategy, batch_size=batch_size)
    print(f"[ENC] Train vectors: {X_train_vec.shape}, Test vectors: {X_test_vec.shape}")

    clf = _build_classifier(classifier)
    print(f"[CLF] Training {classifier} ...")
    clf.fit(X_train_vec, y_train)

    print("[CLF] Predicting on test set ...")
    y_pred = clf.predict(X_test_vec)

    metrics = _print_metrics(y_test, y_pred, title=f"{classifier} on {embed_strategy} embeddings")
    shapes = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "embedding_dim": X_train_vec.shape[1] if X_train_vec.size else 0,
    }
    return metrics, shapes


def run_default_grid():
    """
    Quick loop over a few embedding/classifier combos for table-friendly outputs.
    Adjust lists as needed for your slides.
    """
    embed_strategies: List[EmbedStrategy] = ["cls", "mean", "last4_mean", "last4_cls_concat"]
    classifiers: List[ClassifierName] = ["linear_svm", "logreg", "random_forest"]

    results = []
    for es in embed_strategies:
        for clf_name in classifiers:
            print(f"\n[RUN] strategy={es} | classifier={clf_name}")
            metrics, shapes = train_and_evaluate(embed_strategy=es, classifier=clf_name)
            results.append(
                {
                    "embed_strategy": es,
                    "classifier": clf_name,
                    **metrics,
                    **shapes,
                }
            )

    # Nicely format a compact table
    header = ["embed", "clf", "acc", "macro_f1", "micro_f1"]
    print("\n=== Summary (copy to PPT/table) ===")
    print("\t".join(header))
    for row in results:
        print(
            f"{row['embed_strategy']}\t{row['classifier']}\t"
            f"{row.get('accuracy', 0):.4f}\t{row.get('macro_f1', 0):.4f}\t{row.get('micro_f1', 0):.4f}"
        )


if __name__ == "__main__":
    run_default_grid()
