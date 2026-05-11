import torch
import torch.nn as nn
from torch_scatter import scatter_sum


class PBLinear(nn.Module):
    """保持稳定性：线性层 + LayerNorm"""

    def __init__(self, in_features, out_features):
        super(PBLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x):
        return self.layer_norm(self.linear(x))


class PBInfer(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=8, var_feat_dim=8):
        super(PBInfer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1. 初始投影
        self.var_init_embed = nn.Linear(var_feat_dim, hidden_dim)
        self.con_init_embed = nn.Linear(2, hidden_dim)

        # 2. 消息传递层 (参考 network016 的 MLP 结构)
        self.v2c_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 是为了合并系数特征
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.c2v_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 3. 状态更新层 (使用 network016 证明稳定的 GRU)
        # var_gru 的输入：[来自约束的消息 + 来自双生兄弟的消息]
        self.var_gru = nn.GRUCell(hidden_dim * 2, hidden_dim)
        self.con_gru = nn.GRUCell(hidden_dim, hidden_dim)

        # 4. 输出预测头 (合并正负文字的状态)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, var_num, con_num, var_features, edges, con_coeffs, rhs, con_type, device):
        # 初始化双生文字 (Literals)
        # 前 var_num 个是 x, 后 var_num 个是 not x
        h_pos = self.var_init_embed(var_features)
        h_neg = self.var_init_embed(var_features)  # 初始时 x 和 not x 具有相同先验
        h_var = torch.cat([h_pos, h_neg], dim=0)  # [2 * var_num, hidden_dim]

        # 初始化约束节点
        con_input = torch.cat([rhs, con_type], dim=1)
        h_con = self.con_init_embed(con_input)

        row, col = edges.indices()  # row: 约束索引, col: 变量索引

        for _ in range(self.num_layers):
            # --- 1. 文字 -> 约束 (Literals -> Constraints) ---
            # 只有正向文字直接参与 PBO 线性项
            v_msg_input = torch.cat([h_var[:var_num][col], con_coeffs.unsqueeze(-1)], dim=-1)
            v_msgs = torch.tanh(self.v2c_mlp(v_msg_input))

            agg_v2c = scatter_sum(v_msgs, row, dim=0, dim_size=con_num)
            h_con = self.con_gru(agg_v2c, h_con)

            # --- 2. 约束 -> 文字 (Constraints -> Literals) ---
            c_msg_input = torch.cat([h_con[row], con_coeffs.unsqueeze(-1)], dim=-1)
            c_msgs = torch.tanh(self.c2v_mlp(c_msg_input))

            agg_c2v = scatter_sum(c_msgs, col, dim=0, dim_size=var_num)

            # --- 3. 双生变量交互 (Flip & Update) ---
            # 构造输入文字更新的消息：正向文字有 agg_c2v，负向文字暂时设为 0
            # 它们会通过下面的 flip 操作相互影响
            msg_to_literals = torch.cat([agg_c2v, torch.zeros_like(agg_c2v)], dim=0)

            # 核心：交换正负文字的状态 (Flip)
            # flip_h 的前 var_num 是 not x 的状态，后 var_num 是 x 的状态
            flip_h = torch.cat([h_var[var_num:], h_var[:var_num]], dim=0)

            # 拼接 [来自约束的消息, 来自双生兄弟的消息]
            gru_input = torch.cat([msg_to_literals, flip_h], dim=-1)
            h_var = self.var_gru(gru_input, h_var)

        # 最终预测：结合 x 和 not x 的推理结果
        out_input = torch.cat([h_var[:var_num], h_var[var_num:]], dim=-1)
        var_pred = torch.softmax(self.prediction_head(out_input), dim=1)[:, 1]
        return var_pred


# ==============================================================================
# ===== 数值稳定的 trc_loss =====
# ==============================================================================
def trc_loss(var_values, batch_con_var_indices, batch_con_coeffs, batch_con_types, batch_con_rhs, device):
    con_types = batch_con_types.to(device).flatten() # 确保是 1D
    con_rhs = batch_con_rhs.to(device).flatten() # 确保是 1D
    con_num_i = con_types.shape[0]

    constraint_loss = torch.tensor(0.0, device=device)
    if con_num_i == 0 or not batch_con_var_indices:
        return constraint_loss

    # --- 1. 数据准备 (GPU-native) ---
    # 将 list of tensors 拼合成一个大 tensor
    # (注意：如果 .pt 文件中这些已经是 1D tensor，则不需要 .to(device))
    all_var_indices = torch.cat(batch_con_var_indices).to(device)
    all_coeffs = torch.cat(batch_con_coeffs).to(device)

    # 创建一个 "batch" 向量，告诉 scatter_sum 每个元素属于哪个约束
    # e.g., [0, 0, 0, 1, 1, 2, 2, 2, ...]
    lengths = torch.tensor([len(indices) for indices in batch_con_var_indices], device=device)
    con_batch_ptr = torch.repeat_interleave(torch.arange(con_num_i, device=device), repeats=lengths)

    # --- 2. 计算 Ax (GPU-native) ---
    # (1) 根据索引从 x 向量中收集所有需要的值
    all_values = var_values[all_var_indices] # GPU gather
    
    # (2) 计算 A_ij * x_j
    all_products = all_coeffs * all_values # GPU element-wise
    
    # (3) 对每个约束 i，求和 (sum(A_ij * x_j))
    weighted_sum = scatter_sum(all_products, con_batch_ptr, dim=0, dim_size=con_num_i) # GPU scatter_sum

    # --- 3. 计算违反量 (GPU-native, 与之前相同) ---
    loss_ge = torch.relu(con_rhs - weighted_sum) # >=
    loss_le = torch.relu(weighted_sum - con_rhs) # <=
    loss_eq = torch.abs(weighted_sum - con_rhs) # ==
    
    raw_violation = torch.where(
        con_types == 1, loss_ge,
        torch.where(con_types == -1, loss_le, loss_eq)
    ) 
    
    constraint_loss = torch.log1p(raw_violation).mean()

    return constraint_loss


def trc_test(
    var_probs,
    batch_con_var_indices,
    batch_con_coeffs,
    batch_con_types,
    batch_con_rhs,
    device,
    threshold: float = 0.5,
    tol: float = 1e-8,
):
    """
    GPU 加速版测试函数。
    严格对齐原版 trc_test 逻辑，同时确保所有中间变量在同一 device。
    """
    # --- 1. 基础数据准备与设备搬运 ---
    con_types = batch_con_types.to(device).flatten()
    con_rhs = batch_con_rhs.to(device).flatten()
    con_num = con_types.shape[0]

    if con_num == 0 or not batch_con_var_indices:
        return torch.tensor(0.0, device=device), 0

    # 变量概率转二值化变量 (确保在 GPU)
    var_probs = var_probs.to(device).flatten()
    var_values = (var_probs > threshold).float()
    #var_values = var_probs.float()  # 直接使用概率值进行计算

    # --- 2. 拼接索引与系数 (核心加速点) ---
    all_var_indices = torch.cat(batch_con_var_indices).to(device)
    all_coeffs = torch.cat(batch_con_coeffs).to(device)
    lengths = torch.tensor([len(ix) for ix in batch_con_var_indices], device=device)

    # 映射每个系数到对应的约束 ID
    con_batch_ptr = torch.repeat_interleave(torch.arange(con_num, device=device), repeats=lengths)

    # --- 3. 计算 Ax (全向量化) ---
    selected_values = var_values[all_var_indices]
    products = all_coeffs * selected_values
    weighted_sum = scatter_sum(products, con_batch_ptr, dim=0, dim_size=con_num)

    # --- 4. 计算违反量 (逻辑与 trc_loss 严格一致) ---
    loss_ge = torch.relu(con_rhs - weighted_sum)  # >=
    loss_le = torch.relu(weighted_sum - con_rhs)  # <=
    loss_eq = torch.abs(weighted_sum - con_rhs)  # ==

    raw_violation = torch.where(
        con_types == 1,
        loss_ge,
        torch.where(con_types == -1, loss_le, loss_eq),
    )

    # --- 5. 结果汇总 ---
    loss = torch.log1p(raw_violation).mean()
    satisfied_count = int((raw_violation <= tol).sum().item())

    return loss, satisfied_count