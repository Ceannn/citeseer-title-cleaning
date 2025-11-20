import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# === 手动填写（或改成从文件读取）三种方法的最终指标 ===
# 这些是你之前跑出来的大致结果，你可以根据最后一次实验精确改一下

METRICS = {
    "Naive Bayes (TF-IDF)": {
        "accuracy": 0.7210,
        "macro_f1": 0.7100,
    },
    "Word2Vec + SVM": {
        "accuracy": 0.7280,
        "macro_f1": 0.7251,
    },
    "BERT Fine-tune": {
        "accuracy": 0.8900,   # 你有一版是 0.890 / 0.893，都可以按最终结果改
        "macro_f1": 0.8882,
    },
}

# 可选：BERT epoch sweep (40k 子集) 的结果，用来画收敛曲线
BERT_EPOCH_SWEEP = {
    1: {"accuracy": 0.8870, "macro_f1": 0.8846},
    2: {"accuracy": 0.8740, "macro_f1": 0.8722},
    3: {"accuracy": 0.8880, "macro_f1": 0.8853},
}


def ensure_plot_dir():
    os.makedirs("experiments/plots", exist_ok=True)


def plot_model_comparison():
    ensure_plot_dir()

    models = list(METRICS.keys())
    accs = [METRICS[m]["accuracy"] for m in models]
    f1s = [METRICS[m]["macro_f1"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.title("Model Comparison on Test Set (Accuracy & Macro-F1)")

    plt.bar(x - width / 2, accs, width, label="Accuracy")
    plt.bar(x + width / 2, f1s, width, label="Macro-F1")

    plt.xticks(x, models, rotation=15, ha="right")
    plt.ylim(0.6, 1.0)
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()

    out_path = "experiments/plots/model_comparison_acc_f1.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[SAVE] {out_path}")


def plot_bert_epoch_sweep():
    ensure_plot_dir()

    epochs = sorted(BERT_EPOCH_SWEEP.keys())
    accs = [BERT_EPOCH_SWEEP[e]["accuracy"] for e in epochs]
    f1s = [BERT_EPOCH_SWEEP[e]["macro_f1"] for e in epochs]

    plt.figure(figsize=(6, 4))
    plt.title("BERT Fine-tuning: Epoch vs Performance (40k subset)")
    plt.plot(epochs, accs, marker="o", label="Accuracy")
    plt.plot(epochs, f1s, marker="s", label="Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0.85, 0.90)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(epochs)
    plt.legend()
    plt.tight_layout()

    out_path = "experiments/plots/bert_epoch_sweep.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    sns.set_style("whitegrid")
    plot_model_comparison()
    plot_bert_epoch_sweep()
