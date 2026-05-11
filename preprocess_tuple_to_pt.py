import argparse
import os
import torch
import ast
from tqdm import tqdm
import torch.nn.functional as F  # 若在 process_single_file 中使用到
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==============================================================================
# ===== 功能函数：处理单个 .jb/.data 文件并返回包含 Tensors 的字典 =====
# ==============================================================================
def process_single_file(jb_filepath):
    """
    读取单个文件 (只包含可行性)，执行预处理步骤，
    返回一个包含 Tensors 的字典 (不含解标签和目标)。
    """
    with open(jb_filepath, 'r') as f:
        try:
            data = ast.literal_eval(f.read())
        except Exception as e:
            print(f"\nError parsing file {jb_filepath}: {e}")
            return None  # 返回 None 表示处理失败
        
    var_num = data[0]
    con_num_declared = data[1]  # 文件声明的约束数
    clauses = data[2]

    # 基本检查
    if var_num <= 0:
        print(f"\nWarning: File {jb_filepath} has var_num <= 0 ({var_num}). Skipping.")
        return None
            
    # --- 2. 处理约束 ---
    con_var_indices, con_coeffs = [], []
    con_idx, var_idx = [], []
    rhs, con_type = [], []

    var_coeffs_map = {i: [] for i in range(var_num)}
    var_con_types_map = {i: [] for i in range(var_num)}
    
    actual_con_count = 0  # 实际处理的有效约束计数

    for constraint_index, sublist in enumerate(clauses):
        temp_vars_tuples = []
        temp_non_tuples = []
        for item in sublist:
            if isinstance(item, tuple):
                temp_vars_tuples.append(item)
            else:
                temp_non_tuples.append(item)

        if len(temp_non_tuples) < 2:
            print(f"\nWarning: File {jb_filepath} constraint {constraint_index} format error (non-tuples < 2). Skipping constraint: {sublist}")
            continue

        current_con_type = None
        current_rhs = None
        if temp_non_tuples[-2] in [-1, 0, 1]:
            current_con_type = temp_non_tuples[-2]
            current_rhs = temp_non_tuples[-1]
        elif len(temp_non_tuples) > 2 and temp_non_tuples[-3] in [-1, 0, 1]: 
            current_con_type = temp_non_tuples[-3]
            current_rhs = temp_non_tuples[-2] 
            print(f"\nWarning: File {jb_filepath} constraint {constraint_index} using potentially old format for type/rhs.")
        else:
            print(f"\nWarning: File {jb_filepath} constraint {constraint_index} cannot determine type/rhs. Skipping constraint: {sublist}")
            continue 

        if not isinstance(current_con_type, int) or current_con_type not in [-1, 0, 1]:
            print(f"\nWarning: File {jb_filepath} constraint {constraint_index} invalid type {current_con_type}. Skipping constraint.")
            continue
        if not isinstance(current_rhs, (int, float)):
            print(f"\nWarning: File {jb_filepath} constraint {constraint_index} invalid rhs {current_rhs}. Skipping constraint.")
            continue

        db_var_idx, coeff = [], []
        valid_vars_in_constraint = False
        for var_tuple in temp_vars_tuples:
            if len(var_tuple) != 3:
                print(f"\nWarning: File {jb_filepath} constraint {constraint_index} variable tuple format error. Skipping var: {var_tuple}")
                continue
            con_val, var_val, coeff_val = var_tuple 

            if not isinstance(var_val, int) or var_val <= 0:
                print(f"\nWarning: File {jb_filepath} constraint {constraint_index} invalid variable index {var_val}. Skipping var: {var_tuple}")
                continue
                 
            v_idx_0based = var_val - 1 

            if v_idx_0based >= var_num:
                print(f"\nWarning: File {jb_filepath} constraint {constraint_index} variable index {v_idx_0based} >= var_num {var_num}. Skipping var: {var_tuple}")
                continue
            if not isinstance(coeff_val, (int, float)):
                print(f"\nWarning: File {jb_filepath} constraint {constraint_index} invalid coefficient {coeff_val}. Skipping var: {var_tuple}")
                continue

            con_idx.append(actual_con_count) 
            var_idx.append(v_idx_0based)
            db_var_idx.append(v_idx_0based) 
            coeff.append(float(coeff_val))  
            var_coeffs_map[v_idx_0based].append(float(coeff_val))
            var_con_types_map[v_idx_0based].append(current_con_type)
            valid_vars_in_constraint = True

        if valid_vars_in_constraint:
            con_type.append(current_con_type)
            rhs.append(float(current_rhs))
            con_var_indices.append(torch.tensor(db_var_idx, dtype=torch.long))
            con_coeffs.append(torch.tensor(coeff, dtype=torch.float32))
            actual_con_count += 1 
        else:
            print(f"\nWarning: File {jb_filepath} constraint {constraint_index} has no valid variables after processing. Ignoring constraint: {sublist}")
             
    # --- 3. 构建变量特征 (8维) ---
    var_features_list = []
    feature_dim = 8
    
    for i in range(var_num):
        coeffs = var_coeffs_map[i]
        c_types = var_con_types_map[i]

        if not coeffs:
            constraint_features = torch.zeros(8, dtype=torch.float32)
        else:
            coeffs_t = torch.tensor(coeffs, dtype=torch.float32)
            c_types_t = torch.tensor(c_types, dtype=torch.float32)
            constraint_features = torch.tensor([
                float(len(coeffs)),                    # 约束参与度
                torch.mean(torch.abs(coeffs_t)),       # 平均绝对系数
                torch.std(coeffs_t).nan_to_num(0.0),   # 系数标准差
                torch.max(coeffs_t),                   # 最大系数
                torch.min(coeffs_t),                   # 最小系数
                torch.sum(c_types_t == 1).float(),     # >= 类型约束计数
                torch.sum(c_types_t == -1).float(),    # <= 类型约束计数
                torch.sum(c_types_t == 0).float()      # == 类型约束计数
            ], dtype=torch.float32)

        var_features_list.append(constraint_features)

    # --- 4. 组装最终 Tensors ---
    if not var_features_list:
        var_features = torch.empty((0, feature_dim), dtype=torch.float32)
    else:
        var_features = torch.stack(var_features_list)

    if not con_idx or not var_idx:
        edge_indices = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_indices = torch.tensor([con_idx, var_idx], dtype=torch.long)

    rhs_tensor = torch.tensor(rhs, dtype=torch.float32).unsqueeze(-1) if rhs else torch.empty((0,1), dtype=torch.float32)
    con_type_tensor = torch.tensor(con_type, dtype=torch.float32).unsqueeze(-1) if con_type else torch.empty((0,1), dtype=torch.float32)

    processed_data = {
        "var_num": var_num,
        "con_num": actual_con_count,           # 实际有效约束数量
        "edge_indices": edge_indices,
        "rhs": rhs_tensor,
        "con_type": con_type_tensor,
        "con_var_indices": con_var_indices,    # List[Tensor]
        "con_coeffs": con_coeffs,              # List[Tensor]
        "var_features": var_features,          # [var_num, 8]
    }
    
    if actual_con_count == 0:
        print(f"\nWarning: File {jb_filepath} resulted in 0 valid constraints after processing. Skipping save.")
        return None
        
    return processed_data

def make_output_filename(input_name, matched_ext, output_suffix=".pt"):
    """
    把输入文件名去掉匹配到的最后扩展名，追加 .pt
    例: a.opb.jb + '.jb' -> a.opb.pt；b.data + '.data' -> b.pt
    """
    base = input_name[:-len(matched_ext)] if matched_ext and input_name.endswith(matched_ext) else os.path.splitext(input_name)[0]
    return base + output_suffix

def process_and_save(jb_filepath, output_folder, exts, output_suffix, max_bytes_limit):
    """
    子进程任务：处理并直接保存，避免主进程传输大 Tensor。
    """
    try:
        filename = os.path.basename(jb_filepath)
        # --- 新增：文件大小检查 ---
        try:
            file_size_bytes = os.path.getsize(jb_filepath)
            if file_size_bytes > max_bytes_limit:
                gb_size = file_size_bytes / (1024**2)
                limit_gb = max_bytes_limit / (1024**2)
                msg = f"File size ({gb_size:.2f} MB) exceeds limit ({limit_gb:.2f} MB)"
                return ("skip", filename, msg)
        except OSError as e:
             return ("err", filename, f"Cannot get file size: {e}")
        # --- 检查结束 ---
        # 找到与配置匹配的扩展名（按长到短优先，确保 .opb.jb 时优先匹配 .jb）
        matched_ext = None
        for ext in sorted(exts, key=len, reverse=True):
            if filename.endswith(ext):
                matched_ext = ext
                break

        output_filename = make_output_filename(filename, matched_ext, output_suffix=output_suffix)
        output_filepath = os.path.join(output_folder, output_filename)

        processed_data_dict = process_single_file(jb_filepath)
        if processed_data_dict is not None:
            torch.save(processed_data_dict, output_filepath)
            return ("ok", filename, "")
        else:
            return ("skip", filename, "returned None")
    except Exception as e:
        return ("err", os.path.basename(jb_filepath), str(e))

# ==============================================================================
# ===== 主程序：遍历文件夹，处理文件并保存 =====
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess .data files into .pt files for faster loading.")
    parser.add_argument("-i", "--input_folder", default="", help="Folder containing the input files.")
    parser.add_argument("-o", "--output_folder", default="", help="Folder where the processed .pt files will be saved.")
    parser.add_argument("-w", "--workers", type=int, default=36, help="并行进程数，默认等于 CPU 核心数。")
    parser.add_argument("-e", "--ext", nargs="+", default=[".jb", ".data"], help="输入文件扩展名列表，默认同时支持 .jb 与 .data")
    parser.add_argument("--output-suffix", default=".pt", help="输出文件后缀，默认 .pt")
    parser.add_argument("--max-byte", type=float, default=1000, help="要处理的最大文件大小 (MB我自己写的)，超过则跳过。")
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    exts = args.ext
    output_suffix = args.output_suffix
    max_bytes_limit = args.max_byte * 1024 * 1024

    os.makedirs(output_folder, exist_ok=True)

    # 收集符合扩展名的文件
    all_files = os.listdir(input_folder)
    jb_files = [f for f in all_files if any(f.endswith(ext) for ext in exts)]
    if not jb_files:
        print(f"Error: No input files with {exts} found in {input_folder}")
        exit(1)

    print(f"开始预处理 {len(jb_files)} 个文件（扩展名 {exts}）从 {input_folder} 到 {output_folder}")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    tasks = [os.path.join(input_folder, f) for f in jb_files]
    with ProcessPoolExecutor(max_workers=args.workers,max_tasks_per_child=1) as ex:
        futures = [ex.submit(process_and_save, p, output_folder, exts, output_suffix, max_bytes_limit) for p in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            status, fname, msg = fut.result()
            if status == "ok":
                processed_count += 1
            elif status == "skip":
                skipped_count += 1
            else:
                error_count += 1
                print(f"\n处理文件 {fname} 时发生错误: {msg}")

    print(f"\n预处理完成！成功处理并保存 {processed_count} 个文件。跳过 {skipped_count} 个，错误 {error_count} 个。")