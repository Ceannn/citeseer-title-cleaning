# src/data_loader.py
import os
from typing import List, Tuple

import pandas as pd

# === 路径设置：以当前文件为基准，自动找到项目根目录 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

TEST_EXCEL_FILENAME = "testSet-1000.xlsx"
TEST_TITLE_COL = "title given by manchine"   
TEST_LABEL_COL = "Y/N"  


def read_title_lines(path: str, encoding: str = "gb18030") -> List[str]:
    """
    读取“每行一个标题”的纯文本文件。
    用 gb18030 是为了兼容 Windows/中文环境导出的文件。
    """
    titles = []
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            titles.append(line)
    return titles


def load_train_from_txt(data_dir: str = DATA_DIR) -> Tuple[list, list]:
    """
    只从 positive_trainingSet / negative_trainingSet 读取训练数据。
    返回: X(标题列表), y(标签列表)
    """
    pos_path = os.path.join(data_dir, "positive_trainingSet")
    neg_path = os.path.join(data_dir, "negative_trainingSet")

    if not os.path.exists(pos_path):
        raise FileNotFoundError(f"未找到正类文件: {pos_path}")
    if not os.path.exists(neg_path):
        raise FileNotFoundError(f"未找到负类文件: {neg_path}")

    X_pos = read_title_lines(pos_path, encoding="gb18030")
    X_neg = read_title_lines(neg_path, encoding="gb18030")

    X = X_pos + X_neg
    y = [1] * len(X_pos) + [0] * len(X_neg)  # 1 = 正类, 0 = 负类

    print(f"[INFO] 正类样本数: {len(X_pos)}, 负类样本数: {len(X_neg)}, 总数: {len(X)}")
    return X, y


def _map_labels(raw_series) -> list:
    """
    把 testSet 里的标签映射成 0/1。
    这里先尝试直接转 int，如果失败，就按常见字符串映射。
    你可以根据自己 Excel 的实际取值来改。
    """
    try:
        return raw_series.astype(int).tolist()
    except Exception:
        # 如果不是数字，比如 'pos'/'neg'、'correct'/'wrong' 等
        mapping_pos = {"pos", "positive", "clean", "correct", "1", 1, "y", "yes"}
        mapping_neg = {"neg", "negative", "noisy", "wrong", "0", 0, "n", "no"}

        labels = []
        for v in raw_series:
            if isinstance(v, str):
                v_norm = v.strip().lower()
            else:
                v_norm = str(v).strip().lower()

            if v_norm in mapping_pos:
                labels.append(1)
            elif v_norm in mapping_neg:
                labels.append(0)
            else:
                # 实在识别不了的先当负类，也可以 raise 出来调试
                # 你可以在这里 print(v) 看看是啥奇怪标签
                labels.append(0)
        return labels


def load_train_test_data(
    data_dir: str = DATA_DIR,
    test_title_col: str = TEST_TITLE_COL,
    test_label_col: str = TEST_LABEL_COL,
) -> Tuple[list, list, list, list]:
    """
    读取完整训练集 + 测试集：
    - 训练：来自 positive_trainingSet / negative_trainingSet
    - 测试：来自 testSet.xlsx（带标题和标签）

    返回: X_train, y_train, X_test, y_test
    """
    # 1) 训练数据
    X_train, y_train = load_train_from_txt(data_dir=data_dir)

    # 2) 测试数据（Excel）
    test_path = os.path.join(data_dir, TEST_EXCEL_FILENAME)
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"未找到测试集 Excel 文件: {test_path}")

    df = pd.read_excel(test_path)
    print("[INFO] testSet.xlsx 列名:", list(df.columns))

    if test_title_col not in df.columns or test_label_col not in df.columns:
        raise ValueError(
            f"找不到指定列: title_col={test_title_col}, label_col={test_label_col}，"
            f"实际列名为: {list(df.columns)}"
        )

    X_test = df[test_title_col].astype(str).tolist()
    y_test = _map_labels(df[test_label_col])

    print(f"[INFO] 测试集样本数: {len(X_test)}")
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # 用来检查数据是否能正确加载
    X_train, y_train, X_test, y_test = load_train_test_data()
    print(f"[CHECK] 训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")
    print("[CHECK] 训练样本示例:", X_train[0] if X_train else "N/A")
    print("[CHECK] 测试样本示例:", X_test[0] if X_test else "N/A", "标签:", y_test[0] if X_test else "N/A")
