import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.data_loader import load_train_test_data
from src.features_w2v import load_word2vec_model, titles_to_vectors


def tsne_word2vec_sample(n_samples: int = 2000, random_state: int = 42):
    """
    随机抽一部分训练样本，用 Word2Vec 句向量 + t-SNE 做 2D 可视化。
    红色：错误标题(0)，绿色：正确标题(1)
    """
    X_train_text, y_train, _, _ = load_train_test_data()
    N = len(X_train_text)
    n_samples = min(n_samples, N)

    idx = np.random.RandomState(random_state).choice(N, size=n_samples, replace=False)
    texts = [X_train_text[i] for i in idx]
    labels = np.array([y_train[i] for i in idx])

    # 1) 加载训练好的 Word2Vec 模型
    w2v_model = load_word2vec_model()

    # 2) 文本 -> 句向量
    X_vec = titles_to_vectors(texts, w2v_model)

    # 3) t-SNE 降到 2D
    print("[TSNE] Running t-SNE on sample vectors...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="random",
        random_state=random_state,
    )
    X_2d = tsne.fit_transform(X_vec)

    # 4) 画图
    plt.figure(figsize=(8, 6))
    mask_0 = labels == 0
    mask_1 = labels == 1

    plt.scatter(
        X_2d[mask_0, 0],
        X_2d[mask_0, 1],
        s=5,
        alpha=0.6,
        label="wrong (0)",
    )
    plt.scatter(
        X_2d[mask_1, 0],
        X_2d[mask_1, 1],
        s=5,
        alpha=0.6,
        label="correct (1)",
    )

    plt.title("t-SNE of Word2Vec title embeddings")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tsne_word2vec_sample()