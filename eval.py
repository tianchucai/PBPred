import argparse
import time
from torch.utils.data import Dataset, DataLoader
import ast
import os
import re
import torch
from tqdm import tqdm
from network import PBInfer, trc_test 
import shutil  # 新增

# ====================== 修改后的自定义数据集类 (加载 .pt) ======================
class PBDataset(Dataset):
    def __init__(self, folder_path, validate=False, quarantine_dir='pbs_bad_pt_test'):
        # 查找 .pt 文件
        all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')]
        if not all_files:
            raise FileNotFoundError(f"在 {folder_path} 中没有找到任何 .pt 文件。请先运行预处理脚本。")

        if validate:
            ok, bad = [], []
            for p in tqdm(all_files, desc="Validating .pt", unit="file"):
                try:
                    # 仅在 CPU 上尝试加载，验证可读性（不保留引用，立即释放）
                    _ = torch.load(p, map_location="cpu")
                    ok.append(p)
                except Exception as e:
                    bad.append((p, str(e)))
            if bad:
                print(f"[警告] 检测到损坏/不可读取的 .pt 文件 {len(bad)} 个，将从训练集中移除。示例：")
                for i, (p, e) in enumerate(bad[:5]):
                    print(f"  - {os.path.basename(p)} | {e}")
                if quarantine_dir:
                    os.makedirs(quarantine_dir, exist_ok=True)
                    for p, _ in bad:
                        try:
                            shutil.move(p, os.path.join(quarantine_dir, os.path.basename(p)))
                        except Exception:
                            pass
                    print(f"[提示] 已将坏文件移动到: {quarantine_dir}")
            self.files = ok
        else:
            # 至少过滤掉 0 字节文件
            self.files = [p for p in all_files if os.path.getsize(p) > 0]

        if not self.files:
            raise FileNotFoundError("有效的 .pt 文件为空，请检查预处理或阈值/过滤设置。")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        # 直接加载预处理好的字典（放在 CPU，再按需搬到 GPU）
        data = torch.load(filename, map_location="cpu")
        return data  # 直接返回，无需任何计算
  
# ====================== 数据预处理函数 ======================
def collate_fn(batch):
    """由于每个样本尺寸不同，返回原始字典列表"""
    return batch  # 保持样本独立性


def eval_models(test_folder, model_folder, device):
    """
    评估所有保存模型在测试集上的表现
    :param test_folder: 测试集数据路径
    :param model_folder: 模型保存路径
    :param device: 计算设备
    """
    # ====================== 初始化测试集加载器 ======================
    test_dataset = PBDataset(test_folder)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    # ====================== 获取模型文件列表 ======================
    model_files = [f for f in os.listdir(model_folder) if f.endswith('.pth')]
    model_files = sorted(model_files, key=lambda x: int(re.search(r'epoch(\d+)', x).group(1)))

    results = {}
    model = PBInfer().to(device)
    
    # ====================== 遍历所有模型进行评估 ======================
    for model_file in tqdm(model_files, desc="Evaluating Models"):
        best = 10000
        satnum = 0
        allnum = 0
        kexingjie = 0
        # 加载模型权重
        model_path = os.path.join(model_folder, model_file)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        total_loss = 0.0
        sample_count = 0

        # ====================== 在测试集上评估 ======================
        with torch.no_grad():
            for samples in test_loader:
                data = samples[0]
                
                # (增加检查，防止0约束或0变量导致错误)
                if data['con_num'] == 0 or data['var_num'] == 0:
                    print(f"Skipping sample with 0 constraints or variables.")
                    continue
                
                edges = torch.sparse_coo_tensor(
                    data["edge_indices"],
                    torch.cat(data["con_coeffs"]),
                    size=(data["con_num"], data["var_num"])
                ).coalesce().to(device)

                # ===== 修改点 3：模型调用 (必须传入 var_features) =====
                var_pred = model(
                    data["var_num"],
                    data["con_num"],
                    data["var_features"].to(device), # <-- 传入特征
                    edges,
                    torch.cat(data["con_coeffs"]).to(device),
                    data["rhs"].to(device),
                    data["con_type"].to(device),
                    device
                )

                
                #print(f'预测向量是{var_pred}',flush=True)
                # ===== 修改点 4：var_pred 已经是 0/1 向量了 =====
                #comparison = var_pred 
                #print(comparison)
                #print(data["var_labels"])

                # 调用 test_trc_loss
                loss,sat = trc_test(
                    var_pred, # <--- 直接传入 0/1 决策
                    data['con_var_indices'],
                    data['con_coeffs'],
                    data["con_type"].flatten(),
                    data["rhs"].flatten(),
                    device
                )   

                total_loss += loss.item()
                sample_count += 1
                allnum += data['con_num']
                satnum += sat
                if(data['con_num']-sat<best):
                    best = data['con_num']-sat
                if(data['con_num']==sat):
                    kexingjie+=1
                    
        # 计算平均损失
        avg_loss = total_loss / sample_count
        
        # ===== 修改点 5：打印 var_pred (0/1 向量)，而不是 softmax =====
        print(f"模型 {model_file}: 测试损失 = {avg_loss:.10f} 最好表现 = {best} 解出来了 = {kexingjie} 满足比 = {satnum}/{allnum} 满足百分比 = {100*satnum/allnum:.4f}%",flush=True)
        #print(f'预测向量是{var_pred}',flush=True)
    return results


# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folders",default='', type=str)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    test_data_path = "pbo_pt_测试集"
    model_save_path = args.input_folders

    eval_results = eval_models(
        test_folder=test_data_path,
        model_folder=model_save_path,
        device=device
    )