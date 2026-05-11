# NLocalPBO

基于图神经网络的伪布尔优化（ Pseudo-Boolean Optimization, PBO ）局部搜索求解器。

## 项目概述

本项目包含 6 个核心 Python 文件，实现了从问题转换、数据预处理、模型训练到推理求解的完整流程。

## 文件说明

| 文件 | 功能 |
|------|------|
| `convert_opb_to_tuple.py` | 将 `.opb`/`.mps`/`.lp` 问题文件转换为元组格式 |
| `preprocess_tuple_to_pt.py` | 将元组文件预处理为 PyTorch 张量格式 (`.pt`) |
| `network.py` | 定义 `PBInfer` 图神经网络模型 |
| `run_pbpred.py` | 训练模型 |
| `eval.py` | 评估训练好的模型在测试集上的表现 |
| `generate_sol.py` | 使用模型预测结果生成可行解 |

## 数据处理流程

```
.opb/.mps/.lp 文件
       │
       ▼
convert_opb_to_tuple.py
       │  (输出 .data 元组文件)
       ▼
preprocess_tuple_to_pt.py
       │  (输出 .pt 张量文件)
       ▼
network.py + run_pbpred.py
       │  (训练输出 .pth 模型文件)
       ▼
generate_sol.py / eval.py
       │  (生成解或评估模型)
       ▼
.sol 解文件
```

## 依赖

```bash
pip install pyscipopt torch tqdm torch-scatter numpy
```

## 使用方法

### 1. 转换问题格式

```bash
# 修改 convert_opb_to_tuple.py 中的配置
input_folder = "你的输入文件夹"
output_folder = "你的输出文件夹"
MAX_WORKERS = 36

python convert_opb_to_tuple.py
```

### 2. 预处理为张量

```bash
python preprocess_tuple_to_pt.py \
    -i 输入文件夹 \
    -o 输出文件夹 \
    -w 36 \
    -e .data \
    --max-byte 1000
```

### 3. 训练模型

```bash
python run_pbpred.py \
    -i pbo_pt_训练集 \
    -lr 1e-3 \
    -e 100 \
    -m 模型保存路径 \
    --validate-pt
```

### 4. 评估模型

```bash
python eval.py \
    -i pbo_pt_测试集 \
    -m 模型路径
```

### 5. 生成解

```bash
python generate_sol.py \
    -i pbo_pt_测试集 \
    --threshold 0.5
```

## 模型架构 ( PBInfer )

基于约束-变量二分图的神经网络：

- **变量节点**：每个变量 x 有正负两个文字状态（x 和 ¬x）
- **约束节点**：RHS + 约束类型（≤/=/≥）
- **消息传递**：多层 GRU + MLP 实现变量与约束的信息交互
- **双生文字交互**：通过状态交换模拟变量取反操作
- **输出**：每个变量被赋值为 1 的概率

## 数据格式

### 元组格式 (.data)
```python
(var_num, cons_num, clauses)
# clauses: [[(con_id, var_id, coeff), ..., op_code, rhs], ...]
```

### 张量格式 (.pt)
```python
{
    "var_num": int,
    "con_num": int,
    "edge_indices": Tensor [2, num_edges],
    "rhs": Tensor [con_num, 1],
    "con_type": Tensor [con_num, 1],
    "con_var_indices": List[Tensor],
    "con_coeffs": List[Tensor],
    "var_features": Tensor [var_num, 8],  # 8维变量特征
}
```

## 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `MAX_WORKERS` | 预处理并行进程数 | 36 |
| `--max-byte` | 最大处理文件大小 (MB) | 1000 |
| `--threshold` | 解生成二值化阈值 | 0.5 |
| `-lr` | 学习率 | 1e-3 |
| `-e` | 训练轮数 | 100 |

## 注意事项

- 训练前会自动验证 `.pt` 文件完整性，损坏文件会被隔离到 `quarantine` 目录
- 变量特征为 8 维，包含约束参与度、系数统计量等信息
- 目标函数在转换阶段会被归一化并缩放至 RHS=5.0
