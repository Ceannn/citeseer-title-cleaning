# src/data_loader.py
import os
from typing import List, Tuple

import pandas as pd


def read_title_lines(path: str, encoding: str = "utf-8") -> List[str]:
    """
    读取“每行一个标题”的纯文本文件。
    如果你的文件不是 utf-8，可以把 encoding 参数改成 'gbk' 等。
    """
    titles = []
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            titles.append(line)
    return titles


def load_train_test_data(
    data_dir: str = "data/raw",
    test_title_col: str = "title",
    test_label_col: str = "label",
) -> Tuple[list, list, list, list]:
    """
    读取训练/测试数据并返回:
        X_train, y_train, X_test, y_test

    约定：
    - 训练集：
      - positive_trainingSet：每行一个“正类标题”（比如 label=1）
      - negative_trainingSet：每行一个“负类标题”（比如 label=0）
    - 测试集 testSet.xlsx：
      - 里边有一个标题列，如 'title'
      - 有一个标签列，如 'label'，值为 0/1 或类似

    如果 testSet.xlsx 里的列名不是 title/label，
    你只需要在调用时改一下 test_title_col / test_label_col，
    或者在这里把默认值改掉即可。
    """

    # === 1) 构造文件路径 ===
    pos_path = os.path.join(data_dir, "positive_trainingSet")
    neg_path = os.path.join(data_dir, "negative_trainingSet")
    test_path = os.path.join(data_dir, "testSet.xlsx")

    # === 2) 读取训练集（每行一个标题） ===
    # 如果编码报错，可以把 encoding 改成 'gbk' 或你实际的编码
    X_pos = read_title_lines(pos_path, encoding="utf-8")
    X_neg = read_title_lines(neg_path, encoding="utf-8")

    X_train = X_pos + X_neg
    y_train = [1] * len(X_pos) + [0] * len(X_neg)  # 1 = 正类, 0 = 负类

    # === 3) 读取测试集（Excel 格式） ===
    # 先把整个表读出来
    df = pd.read_excel(test_path)

    # 打印一下列名，帮你确认（只在你直接运行本文件时会触发）
    print("Columns in testSet.xlsx:", list(df.columns))

    # 根据列名取出标题和标签
    # 默认假设有 'title' 和 'label' 两列
    # 如果你的实际列名不同，比如 'Title'/'Label' 或 'text'/'y'，
    # 就把 test_title_col / test_label_col 改成对应名字。
    if test_title_col not in df.columns or test_label_col not in df.columns:
        raise ValueError(
            f"找不到指定列: title_col={test_title_col}, label_col={test_label_col}，"
            f"实际列名为: {list(df.columns)}"
        )

    X_test = df[test_title_col].astype(str).tolist()

    # 标签如果不是 0/1，可以在这里做一个映射
    y_raw = df[test_label_col]
    # 尝试转成 int，如果失败你再改这里的逻辑
    try:
        y_test = y_raw.astype(int).tolist()
    except ValueError:
        # 例如标签是 'pos'/'neg' 之类，可以改成：
        # y_test = [1 if v == 'pos' else 0 for v in y_raw]
        raise ValueError(
            f"标签列无法直接转为 int，请检查 testSet.xlsx 中 '{test_label_col}' 的取值，"
            "并在 data_loader.py 中自定义映射逻辑。"
        )

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # 简单测试一下能不能正常读数据
    X_train, y_train, X_test, y_test = load_train_test_data()

    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print("Train sample:", X_train[0] if X_train else "N/A")
    print("Test sample:", X_test[0] if X_test else "N/A", "Label:", y_test[0] if y_test else "N/A")
