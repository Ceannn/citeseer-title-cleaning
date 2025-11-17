from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import load_train_test_data
from src.features_w2v import load_word2vec_model, titles_to_vectors


def train_and_evaluate_w2v_svm():
    # 1) 读取训练 + 测试数据（和 NB 用的是同一套）
    X_train_text, y_train, X_test_text, y_test = load_train_test_data()
    print(f"[SVM] Train samples: {len(X_train_text)}, Test samples: {len(X_test_text)}")

    # 2) 加载已训练好的 Word2Vec 模型
    w2v_model = load_word2vec_model()

    # 3) 文本 -> Word2Vec 句向量
    print("[SVM] Converting titles to vectors...")
    X_train_vec = titles_to_vectors(X_train_text, w2v_model)
    X_test_vec = titles_to_vectors(X_test_text, w2v_model)
    print(f"[SVM] Train vectors shape: {X_train_vec.shape}, Test vectors shape: {X_test_vec.shape}")

    # 4) 训练 Linear SVM
    # 232k 样本 + 100 维特征，用 LinearSVC 比核 SVM 高效很多
    clf = LinearSVC()
    print("[SVM] Training LinearSVC...")
    clf.fit(X_train_vec, y_train)

    # 5) 在测试集上预测
    y_pred = clf.predict(X_test_vec)

    # 6) 打印评估指标
    print("\n=== [Word2Vec + LinearSVC] Classification Report (Test Set) ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("=== [Word2Vec + LinearSVC] Confusion Matrix (Test Set) ===")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    # 直接运行：python -m src.models_svm
    train_and_evaluate_w2v_svm()