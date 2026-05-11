import os
import re
import sys
import concurrent.futures
import traceback
from pyscipopt import Model
from tqdm import tqdm


def convert_problem_to_feasibility_tuple(problem_path):
    """
    使用 PySCIPOpt 读取 .lp 或 .mps/.opb 文件，并将其转换为
    一个只关注可行性（约束结构）的元组。

    修改点：将优化目标函数提取出来，进行归一化并乘以 10 缩放，
    使得 RHS 稳定在 5.0，以提供强信号并防止数值爆炸。

    Use PySCIPOpt to read .lp or .mps/.opb files and convert them into
    a tuple focusing only on feasibility (constraint structure).

    Modification: Extract the optimization objective function, normalize it
    and scale by 10, so that RHS stabilizes at 5.0 to provide strong signal
    and prevent numerical explosion.
    """
    model = Model()
    
    # 增加日志级别，减少pyscipopt自身的打印输出
    # Increase log level to reduce pyscipopt's own print output
    model.setParam('display/verblevel', 0)

    model.readProblem(problem_path)

    # 1. 获取变量和约束
    # 1. Get variables and constraints
    variables = model.getVars()
    constraints = model.getConss()

    var_num = len(variables)

    # 2. 创建变量名到索引的映射
    # 2. Create mapping from variable names to indices
    var_map = {}
    has_named_indices = True
    temp_vars = [None] * var_num

    for var in variables:
        match = re.search(r"x(\d+)$", var.name)
        if match:
            idx_1based = int(match.group(1))
            if 1 <= idx_1based <= var_num:
                if temp_vars[idx_1based - 1] is None:
                    temp_vars[idx_1based - 1] = var.name
                else:
                    has_named_indices = False
                    break
            else:
                has_named_indices = False
                break
        else:
            has_named_indices = False
            break

    if has_named_indices and not all(v is None for v in temp_vars):
        var_map = {name: i for i, name in enumerate(temp_vars)}
    else:
        var_map = {var.name: i for i, var in enumerate(variables)}

    # 3. 提取约束数据
    # 3. Extract constraint data
    clauses = []
    valid_cons_count = 0
    scip_inf = model.infinity()

    # --- 处理常规约束 ---
    # --- Process regular constraints ---
    for cons in constraints:
        if not cons.isLinear():
            continue

        clause_data = []
        lhs = model.getLhs(cons)
        rhs = model.getRhs(cons)
        op_code = -99
        rhs_value = 0.0

        if abs(lhs - rhs) < 1e-9:  # ==
            op_code = 0
            rhs_value = rhs
        elif lhs <= -scip_inf:  # <=
            op_code = -1
            rhs_value = rhs
        elif rhs >= scip_inf:  # >=
            op_code = 1
            rhs_value = lhs
        else:
            continue  # 跳过范围约束 / Skip range constraints

        try:
            cons_vars_dict = model.getValsLinear(cons)
        except Exception:
            continue

        if not cons_vars_dict:
            continue

        variable_found_in_cons = False
        for var_name_str, coeff in cons_vars_dict.items():
            if var_name_str not in var_map:
                continue

            var_idx_0based = var_map[var_name_str]
            var_idx_1based = var_idx_0based + 1

            clause_data.append((valid_cons_count, var_idx_1based, float(coeff)))
            variable_found_in_cons = True

        if variable_found_in_cons:
            clause_data.append(op_code)
            clause_data.append(float(rhs_value))
            clauses.append(clause_data)
            valid_cons_count += 1

    # --- 【工程优化版】处理目标函数作为约束 ---
    # --- [Engineering Optimization] Process objective function as constraint ---
    # 逻辑：变量系数除以系数总和并乘 10，RHS 固定为 5.0
    # Logic: divide variable coefficients by sum and multiply by 10, RHS fixed at 5.0
    obj_clause_data = []
    obj_has_vars = False

    # 提取原始目标系数
    # Extract raw objective coefficients
    raw_obj_coeffs = []
    active_vars_indices = []

    for var in variables:
        obj_coeff = var.getObj()
        if abs(obj_coeff) > 1e-9:
            if var.name in var_map:
                raw_obj_coeffs.append(obj_coeff)
                active_vars_indices.append(var_map[var.name] + 1)
                obj_has_vars = True

    if obj_has_vars:
        # 计算系数总和（作为分母进行归一化）
        # Calculate sum of coefficients (as denominator for normalization)
        total_obj_sum = sum(raw_obj_coeffs)
        divisor = total_obj_sum if abs(total_obj_sum) > 1e-9 else 1.0

        for i in range(len(raw_obj_coeffs)):
            # 缩放逻辑：(c_i / sum) * 10
            # Scaling logic: (c_i / sum) * 10
            scaled_coeff = (raw_obj_coeffs[i] / divisor) * 10
            obj_clause_data.append((valid_cons_count, active_vars_indices[i], float(scaled_coeff)))

        # 约束类型为 <= (-1)，RHS 固定为 5.0
        # Constraint type is <= (-1), RHS fixed at 5.0
        obj_clause_data.append(-1)
        obj_clause_data.append(5.0)

        clauses.append(obj_clause_data)
        valid_cons_count += 1

    # 释放 SCIP 对象
    # Free SCIP object
    model.freeProb()

    # 4. 组合并返回最终的 tuple
    # 4. Combine and return final tuple
    output_tuple = (var_num, valid_cons_count, clauses)
    return output_tuple


def process_file_wrapper(problem_path, out_path):
    """
    处理单个文件的包装函数，用于多进程。
    Wrapper function for processing a single file, used for multiprocessing.
    """
    filename = os.path.basename(problem_path)
    try:
        data_tuple = convert_problem_to_feasibility_tuple(problem_path)

        if data_tuple[1] <= 0 or not data_tuple[2]:
            return ("WARNING", filename, "处理后没有有效的约束。跳过保存。/ No valid constraints after processing. Skip saving.")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(str(data_tuple))

        return ("SUCCESS", filename, "")

    except Exception as e:
        tb_str = traceback.format_exc()
        return ("ERROR", filename, f"发生严重错误: {e}\n{tb_str}")


if __name__ == "__main__":
    # --- 配置 ---
    # --- Configuration ---
    input_folder = ""
    output_folder = ""
    MAX_WORKERS = 36 # 根据服务器核心数调整 / Adjust based on server core count

    os.makedirs(output_folder, exist_ok=True)
    print(f"开始转换 {input_folder} (RHS=5.0 缩放版)，保存到 {output_folder}")
    print(f"Start converting {input_folder} (RHS=5.0 scaled version), saving to {output_folder}")

    if not os.path.exists(input_folder):
        print(f"Error: 文件夹 {input_folder} 不存在。/ Folder {input_folder} does not exist.")
        sys.exit()

    file_list = [f for f in os.listdir(input_folder) if f.endswith(".opb") or f.endswith(".mps") or f.endswith(".lp")]

    if not file_list:
        print(f"Error: 未找到支持的文件。/ No supported files found.")
        sys.exit()

    print(f"找到 {len(file_list)} 个文件。正在准备任务...")
    print(f"Found {len(file_list)} files. Preparing tasks...")

    tasks = []
    for filename in file_list:
        problem_path = os.path.join(input_folder, filename)
        out_path = os.path.join(output_folder, filename + ".data")
        tasks.append((problem_path, out_path))

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_file_wrapper, prob_path, out_path): prob_path
            for (prob_path, out_path) in tasks
        }

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="处理进度 / Processing"):
            status, filename, message = future.result()
            if status == "ERROR":
                print(f"\n[!] 失败: {filename}\n{message}")
            elif status == "WARNING":
                print(f"\n[!] 警告: {filename} - {message}")

    print("全部转换完成！/ All conversions completed!")