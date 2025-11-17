# src/features_w2v.py
import os
import re
import multiprocessing
from typing import List

import numpy as np
from gensim.models import Word2Vec

from src.data_loader import load_train_test_data

# === 路径设置 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "experiments", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# 你可以按需要改文件名（比如加 dim 数）
W2V_MODEL_PATH = os.path.join(MODEL_DIR, "word2vec_dim100_window5_min2.model")

# 简单英文 token 模式：按单词切，中英文混在一起问题也不大
TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize_title(title: str) -> List[str]:
    """把标题转成小写 token 列表。"""
    text = str(title).lower()
    tokens = TOKEN_PATTERN.findall(text)
    return tokens


def build_corpus() -> List[List[str]]:
    """
    用训练 + 测试标题一起构造 Word2Vec 语料。
    这样 test 里的词也有机会被训练到。
    """
    X_train, _, X_test, _ = load_train_test_data()

    corpus = []
    for t in X_train:
        toks = tokenize_title(t)
        if toks:
            corpus.append(toks)

    for t in X_test:
        toks = tokenize_title(t)
        if toks:
            corpus.append(toks)

    print(f"[W2V] Corpus size (sentences): {len(corpus)}")
    return corpus


def train_word2vec(
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    sg: int = 1,
    epochs: int = 10,
    save_path: str = W2V_MODEL_PATH,
):
    """
    训练 Word2Vec 模型并保存到磁盘。
    - sg=1: skip-gram（一般语义表达会更好）
    """
    corpus = build_corpus()

    workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"[W2V] Training Word2Vec: dim={vector_size}, window={window}, min_count={min_count}, "
          f"sg={sg}, epochs={epochs}, workers={workers}")

    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
    )

    model.save(save_path)
    print(f"[W2V] Saved model to: {save_path}")


def load_word2vec_model(path: str = W2V_MODEL_PATH) -> Word2Vec:
    """从磁盘加载已训练好的 Word2Vec 模型。"""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Word2Vec model not found at {path}. "
            f"Please run `python -m src.features_w2v` first to train it."
        )
    print(f"[W2V] Loading model from: {path}")
    return Word2Vec.load(path)


def titles_to_vectors(titles: List[str], model: Word2Vec) -> np.ndarray:
    """
    把一批标题转成句向量：
    - 对每个标题分词
    - 取在词表中的词的词向量
    - 对这些词向量求均值，作为标题向量
    - 没有任何词在词表中的，就用全 0 向量
    """
    dim = model.vector_size
    vectors = np.zeros((len(titles), dim), dtype=np.float32)

    for i, title in enumerate(titles):
        tokens = tokenize_title(title)
        word_vecs = []

        for tok in tokens:
            if tok in model.wv:
                word_vecs.append(model.wv[tok])

        if word_vecs:
            # 平均池化
            vectors[i] = np.mean(word_vecs, axis=0)
        # 否则保持全 0 向量

    return vectors


if __name__ == "__main__":
    # 直接运行：python -m src.features_w2v
    # 会训练一个 Word2Vec 模型并存到 experiments/models 下
    train_word2vec()