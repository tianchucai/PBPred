# NLocalPBO - Neural Local Search for Pseudo-Boolean Optimization

基于神经网络的伪布尔优化（PBO）局部搜索求解器项目。

## 项目概述

本项目提供了一个完整的管道，用于处理伪布尔优化问题，包括问题转换、数据预处理、模型训练和推理预测。

### 主要功能

1. **问题转换**：将 `.opb`、`.mps`、`.lp` 格式的优化问题转换为统一的元组格式
2. **数据预处理**：将元组数据转换为 PyTorch 张量格式，用于模型训练和推理
3. **神经网络模型**：基于图神经网络的预测模型
4. **推理预测**：使用训练好的模型进行变量选择预测
5. **解生成**：根据预测结果生成可行解

## 文件结构

```
NLocalPBO/
├── convert_opb_to_tuple.py    # 将 .opb/.mps/.lp 文件转换为元组
├── preprocess_tuple_to_pt.py  # 将元组文件预处理为 PyTorch 张量
├── network.py                 # 神经网络模型定义
├── eval.py                    # 模型评估脚本
├── run_pbpred.py              # 运行预测脚本
├── generate_sol.py            # 解生成脚本
└── NLocalSAT-master/          # NLocalSAT 求解器（子模块）
    ├── solvers/               # SAT 求解器集合
    ├── scripts/               # 训练和运行脚本
    ├── solution_space_generator/ # 解空间生成器
    └── satutils/              # SAT 工具函数
```

## 安装依赖

### 基础依赖

```bash
pip install pyscipopt torch tqdm
```

### SCIP 求解器

本项目使用 PySCIPOpt 读取优化问题文件。确保已安装 SCIP 求解器：

- Linux: `sudo apt-get install scipopt`
- macOS: `brew install scip`
- Windows: 从 [SCIP 官网](https://www.scipopt.org/) 下载

## 使用方法

### 1. 转换问题文件

将 `.opb`、`.mps` 或 `.lp` 文件转换为元组格式：

```python
from convert_opb_to_tuple import convert_problem_to_feasibility_tuple

# 转换单个问题
data_tuple = convert_problem_to_feasibility_tuple("problem.opb")
print(f"变量数: {data_tuple[0]}, 约束数: {data_tuple[1]}")

# 批量转换（在脚本中设置 input_folder 和 output_folder）
python convert_opb_to_tuple.py
```

### 2. 预处理数据

将元组文件转换为 PyTorch 张量：

```python
from preprocess_tuple_to_pt import process_single_file

# 处理单个文件
result = process_single_file("problem.opb.jb")
if result is not None:
    print(f"约束特征: {result['con_feat'].shape}")
    print(f"变量特征: {result['var_feat'].shape}")
```

### 3. 运行预测

使用训练好的模型进行预测：

```bash
python run_pbpred.py --input <input_dir> --output <output_dir> --model <model_path>
```

### 4. 生成解

根据预测结果生成可行解：

```bash
python generate_sol.py --input <input_dir> --output <output_dir>
```

## 数据格式

### 输入格式

支持以下问题格式：
- `.opb`：伪布尔优化问题格式
- `.mps`：数学规划系统格式
- `.lp`：线性规划格式

### 中间格式（元组）

转换后的元组格式为：
```python
(var_num, cons_num, clauses)
```
- `var_num`: 变量数量
- `cons_num`: 约束数量
- `clauses`: 约束列表，每个约束包含系数元组和操作符

### 输出格式

预处理后的张量包含：
- `con_feat`: 约束特征矩阵
- `var_feat`: 变量特征矩阵
- `adj`: 邻接矩阵（约束-变量关联）

## 示例

```python
# 完整流程示例
from convert_opb_to_tuple import convert_problem_to_feasibility_tuple
from preprocess_tuple_to_pt import process_single_file
import torch

# 1. 转换问题
data_tuple = convert_problem_to_feasibility_tuple("test.opb")

# 2. 保存元组
with open("test.opb.jb", "w") as f:
    f.write(str(data_tuple))

# 3. 预处理
result = process_single_file("test.opb.jb")

# 4. 加载模型并预测
model = torch.load("model.pt")
model.eval()

with torch.no_grad():
    prediction = model(result['con_feat'], result['var_feat'], result['adj'])
```

## 注意事项

1. 确保输入文件夹中只包含支持的文件格式（`.opb`, `.mps`, `.lp`）
2. 预处理时会自动跳过无效或损坏的文件
3. 建议根据服务器核心数调整 `MAX_WORKERS` 参数
4. 目标函数会被归一化并缩放到 RHS=5.0，以防止数值爆炸

## 许可证

本项目基于 NLocalSAT 项目开发，具体许可证请参考子模块中的 LICENSE 文件。
