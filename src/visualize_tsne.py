import os
import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

from gensim.models import Word2Vec

import torch
from transformers import AutoTokenizer, AutoModel

from src.data_loader import load_train_test_data


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def ensure_plot_dir():
    os.makedirs("experiments/plots", exist_ok=True)


def sample_subset(X, y, n_samples: int = 2000) -> Tuple[list, list]:
    """从全量 train+test 里抽一部分做可视化，避免太慢。"""
    texts = list(X)
    labels = list(y)
    n = len(texts)
    if n <= n_samples:
        return texts, labels

    idx = np.random.choice(n, size=n_samples, replace=False)
    texts_sub = [texts[i] for i in idx]
    labels_sub = [labels[i] for i in idx]
    return texts_sub, labels_sub


# ========= 1. TF-IDF =========

def build_tfidf_features(texts: list, max_features: int = 5000) -> np.ndarray:
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_tfidf = vec.fit_transform(texts)
    return X_tfidf.toarray()


# ========= 2. Word2Vec =========

def tokenize_simple(text: str) -> list:
    # 非常简单的英文 token 化
    return str(text).lower().split()


def build_w2v_features(texts: list, model_path: str) -> np.ndarray:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到 Word2Vec 模型: {model_path}")

    print(f"[W2V] Loading model from: {model_path}")
    w2v = Word2Vec.load(model_path)
    dim = w2v.vector_size

    def sent_vec(tokens):
        vs = [w2v.wv[w] for w in tokens if w in w2v.wv]
        if not vs:
            return np.zeros(dim, dtype=np.float32)
        return np.mean(vs, axis=0)

    all_vecs = []
    for t in texts:
        tokens = tokenize_simple(t)
        all_vecs.append(sent_vec(tokens))
    return np.vstack(all_vecs)


# ========= 3. BERT =========

def build_bert_features(texts: list,
                        model_name: str = "bert-base-uncased",
                        max_length: int = 64,
                        batch_size: int = 64) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[BERT-ENC] Using device:", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            cls = outputs.last_hidden_state[:, 0, :]  # [CLS] 向量
            all_embeddings.append(cls.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


# ========= t-SNE & 绘图 =========

def run_tsne(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    tsne = TSNE(
        n_components=n_components,
        perplexity=30,
        learning_rate=200,
        #n_iter=1000,
        init="pca",
        random_state=RANDOM_SEED,
    )
    X_2d = tsne.fit_transform(X)
    return X_2d


def plot_tsne(X_2d: np.ndarray, y: list, title: str, save_path: str):
    ensure_plot_dir()

    y = np.array(y)
    plt.figure(figsize=(6, 5))
    sns.set_style("whitegrid")

    # 0 = wrong title, 1 = correct title
    colors = ["red" if label == 0 else "green" for label in y]

    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=colors,
        alpha=0.6,
        s=10,
        edgecolors="none",
    )
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[SAVE] {save_path}")


def main():
    # 1. 取 train+test 一起做可视化（标题 + 标签）
    X_train_all, y_train_all, X_test, y_test = load_train_test_data()
    texts_all = list(X_train_all) + list(X_test)
    labels_all = list(y_train_all) + list(y_test)

    print(f"[DATA] Total samples for TSNE pool: {len(texts_all)}")

    # 为了加速，只采样一部分
    texts_sub, labels_sub = sample_subset(texts_all, labels_all, n_samples=2000)
    print(f"[DATA] Using subset for TSNE: {len(texts_sub)} samples")

    # = TF-IDF =
    print("[TSNE] Building TF-IDF features...")
    X_tfidf = build_tfidf_features(texts_sub, max_features=5000)
    print("[TSNE] Running t-SNE on TF-IDF...")
    X_tfidf_2d = run_tsne(X_tfidf)
    plot_tsne(
        X_tfidf_2d,
        labels_sub,
        title="t-SNE of Titles (TF-IDF representation)",
        save_path="experiments/plots/tsne_tfidf.png",
    )

    # = Word2Vec =
    print("[TSNE] Building Word2Vec features...")
    w2v_path = os.path.join("experiments", "models", "word2vec_dim100_window5_min2.model")
    X_w2v = build_w2v_features(texts_sub, w2v_path)
    print("[TSNE] Running t-SNE on Word2Vec...")
    X_w2v_2d = run_tsne(X_w2v)
    plot_tsne(
        X_w2v_2d,
        labels_sub,
        title="t-SNE of Titles (Word2Vec representation)",
        save_path="experiments/plots/tsne_word2vec.png",
    )

    # = BERT =
    print("[TSNE] Building BERT [CLS] features...")
    X_bert = build_bert_features(texts_sub, model_name="bert-base-uncased", max_length=64)
    print("[TSNE] Running t-SNE on BERT...")
    X_bert_2d = run_tsne(X_bert)
    plot_tsne(
        X_bert_2d,
        labels_sub,
        title="t-SNE of Titles (BERT [CLS] representation)",
        save_path="experiments/plots/tsne_bert_cls.png",
    )


if __name__ == "__main__":
    main()
