# CiteSeer Title Cleaning / Classification (IR Course Project)

本项目针对 CiteSeer 数据中“机器抽取标题”的质量问题，构建了一个二分类系统：  
给定一条标题文本，判断其为 **Wrong（错误/噪声标题）** 或 **Correct（正确标题）**。

我们完成了课程要求的 **Task A / Task B / Task C**，并在 Task C 中实现了三条不同的 Transformer 路线（精度上限、可解释对比、工程化部署），同时提供统一的数据入口、统一评估指标与统一产出格式，保证对比实验公平、可复现。

---

## 目录
- [项目亮点](#项目亮点)
- [数据与任务定义](#数据与任务定义)
- [方法总览](#方法总览)
  - [Task A: TF-IDF + Naive Bayes](#task-a-tf-idf--naive-bayes)
  - [Task B: Word2Vec + SVM](#task-b-word2vec--svm)
  - [Task C: Transformer 系列](#task-c-transformer-系列)
    - [C1: End-to-end Fine-tune BERT](#c1-end-to-end-fine-tune-bert)
    - [C2: Frozen BERT Embeddings + 传统分类器大乱斗](#c2-frozen-bert-embeddings--传统分类器大乱斗)
    - [C3: DistilBERT + LoRA（工程化方案）](#c3-distilbert--lora工程化方案)
- [统一评估与产出](#统一评估与产出)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [复现说明](#复现说明)
- [团队分工](#团队分工)

---

## 项目亮点
- **统一入口/统一评估/统一落盘**：所有模型共享同一数据读取与评估产出，确保可控对比。
- **三条 Transformer 路线互补**：
  - **精度上限**：端到端 BERT 微调（Fine-tune）
  - **可解释对比**：Frozen BERT embedding + pooling + classifier ablation
  - **工程化落地**：DistilBERT + LoRA +（廉价特征/两阶段过滤）权衡效果与效率
- **可视化解释**：通过 t-SNE / PCA 等方法展示不同表示空间的可分性。

---

## 数据与任务定义
### 任务
- 输入：标题文本（title）
- 输出：二分类标签（Wrong / Correct）

### 数据来源
- 训练：正/负样本来自文本文件（pos/neg）
- 测试：`testSet-1000.xlsx`（1000 条样本）
- 评估口径：统一输出 Accuracy、Macro-F1，并附带 Confusion Matrix、PR 曲线与 AUPR（用于阈值行为分析）


---

## 方法总览

### Task A: TF-IDF + Naive Bayes
- **思路**：用 TF-IDF 将标题表示为高维稀疏向量，使用 Multinomial Naive Bayes 分类。
- **特点**：训练/推理速度快、可解释性强，是稳健 baseline，但对语义与结构噪声能力有限。

### Task B: Word2Vec + SVM
- **思路**：训练 Word2Vec 得到词向量；标题句向量通过（平均/聚合）得到；下游用 Linear SVM 分类。
- **特点**：相比 TF-IDF 更“语义”，但句向量聚合容易丢失结构信息（顺序/关键 token），提升通常有限。

---

## Task C: Transformer 系列

### C1: End-to-end Fine-tune BERT
- **思路**：使用 `BertForSequenceClassification`（Encoder + 分类头），对本任务进行监督训练（Cross-Entropy loss）。
- **端到端含义**：不仅训练分类头，也更新 BERT 编码器参数，让表示空间针对“wrong/correct”重新组织。
- **工程优化（我们做过）**：
  - GPU / fp16 混合精度加速
  - 支持 train subset（快速迭代验证链路）
- **定位**：最终精度上限方案（效果最强，但训练成本更高）。

### C2: Frozen BERT Embeddings + 传统分类器大乱斗
- **整体实现思路**：数据读取 → 冻结 BERT 提取 embedding → 下游分类器训练与对比  
- **表示提取**：
  - 使用 HuggingFace `AutoModel`（无分类头），开启 `eval()` + `no_grad()`
  - 只输出中间表示，不做反向传播；同一文本多次编码结果一致，易复现
- **Pooling 策略对比**（控制变量）：
  - `CLS`（最后一层 [CLS]）
  - `mean`（token mean pooling，mask 掉 padding）
  - `last4_mean`（最后四层先融合，再做 mean pooling）
- **分类器对比**（同 embedding 上公平竞技）：
  - Logistic Regression
  - Linear SVM
  - Random Forest
- **代表性结果（我们本次实验）**：
  - **CLS + Linear SVM**：Acc=0.8410 / Macro-F1=0.8398（最佳）
  - CLS + LogReg：Acc=0.8330 / Macro-F1=0.8321
  - CLS + RF：Acc=0.8170 / Macro-F1=0.8167
  - mean pooling 整体略弱，last4_mean 与 CLS 接近
- **结论**：Frozen BERT 的稠密向量空间中，类别已接近线性可分，线性模型（SVM/LR）往往优于树模型。

### C3: DistilBERT + LoRA（工程化方案）
我们进一步在更轻量的 backbone 上实现参数高效微调与工业化增强：

#### C3-Base：DistilBERT + LoRA（SEQ_CLS）
- 模型：`distilbert-base-uncased` → `AutoModelForSequenceClassification`
- LoRA 注入：q/k/v/out 等线性层（r=8, alpha=16, dropout=0.1）
- 训练：以 Macro-F1 选 best model，支持 fp16（有 GPU 时）
- 输出：模型权重、metrics/preds/timing JSON/CSV、混淆矩阵与 PR 曲线 PNG

#### C3-FeatConcat：LoRA + 廉价特征拼接（推荐工程最优解）
- 额外提取 8 维“廉价结构特征”（如 length、token_count、digit_ratio、punct_ratio、upper_ratio、space_ratio、repeat_ngram、noise_kw）
- 结构：CLS(768) + MLP(8→64) 拼接成 832 维，再分类
- 直觉：把“格式异常/噪声信号”显式喂给模型，补齐纯语义模型的盲点

#### C3-2Stage：规则快速过滤 + LoRA 精判（吞吐优先）
- Stage1：用规则（数字比例、标点比例、关键词、过长等）直接判定明显 wrong
- Stage2：对剩余样本调用 LoRA 模型推理
- 定位：适合大规模清洗/吞吐优先，但可能带来额外误杀，AUPR/排序能力下降

---

## 统一评估与产出
- `src.eval_utils`：统一计算 classification report、混淆矩阵、PR 曲线与 AUPR，并保存：
  - `experiments/results/*.json`（metrics、timing）
  - `experiments/results/*.csv`（predictions）
  - `experiments/plots/*.png`（confusion matrix、PR curve、对比图）
- `src.visualize_metrics.py`：自动读取 `experiments/results` 汇总对比图（acc/f1 等）

---
