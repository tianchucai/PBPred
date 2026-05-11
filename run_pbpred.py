import argparse
import time
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import ast
from network import PBInfer, trc_loss 
import os
from tqdm import tqdm
import shutil  # 新增

# ====================== 修改后的自定义数据集类 (加载 .pt) ======================
class PBDataset(Dataset):
    def __init__(self, folder_path, validate=True, quarantine_dir='pbs_pt_bad'):
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

def collate_fn(batch):
    return batch

def train(input_folder, device, accum_steps, learning_rate, epochs, model_folder, validate_pt=True, quarantine_dir='pbs_pt_bad'):
    model = PBInfer().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    os.makedirs(model_folder, exist_ok=True)
    
    dataset = PBDataset(input_folder, validate=validate_pt, quarantine_dir=quarantine_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        sample_count = 0
        optimizer.zero_grad()
        accum_count = 0
        total_loss = 0.0
        
        for batch_idx, samples in enumerate(dataloader):
            data = samples[0]
            if data['con_num'] == 0 or data['var_num'] == 0:
                print(f"Skipping sample with 0 constraints or variables.")
                continue

            edges = torch.sparse_coo_tensor(
                data["edge_indices"],
                torch.cat(data["con_coeffs"]),
                size=(data["con_num"], data["var_num"])
            ).coalesce().to(device)

            var_pred = model(
                data["var_num"],
                data["con_num"],
                data["var_features"].to(device),
                edges,
                torch.cat(data["con_coeffs"]).to(device),
                data["rhs"].to(device),
                data["con_type"].to(device),
                device
            )

            loss = trc_loss(
                var_pred,
                data['con_var_indices'],
                data['con_coeffs'],
                data["con_type"].flatten(),
                data["rhs"].flatten(),
                device,
            )
            
            # --- 改进后的跳过逻辑 ---
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: 检测到异常 Loss ({loss.item()})，正在跳过该样本。")
                
                # 安全检查：看看是不是模型权重已经坏了
                with torch.no_grad():
                    # 这里尝试计算概率，看 model 此时的输出是否已经是 NaN
                    prob_check = var_pred
                    if torch.isnan(prob_check).any():
                        print("错误：模型权重已损坏 (Weights contain NaN)! 建议降低学习率或检查模型结构。")
                        # 如果模型坏了，通常只能停止训练并回滚到上一个 Checkpoint
                
                # 必须手动清空当前可能残留的梯度
                optimizer.zero_grad() 
                continue 
            # ----------------------

            is_last_batch = batch_idx == len(dataloader) - 1
            actual_accum = min(accum_steps, accum_count + 1) if is_last_batch else accum_steps
            
            (loss / actual_accum).backward()
            total_loss += loss.item()
            epoch_loss += loss.item()
            sample_count += 1
            accum_count += 1

            if accum_count % accum_steps == 0 or is_last_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                avg_loss = total_loss / accum_count
                print(f"Epoch {epoch} | Batch {(batch_idx+1)/accum_steps:.0f} | Samples {accum_count} | Loss: {avg_loss:.10f}", flush=True)
                total_loss = 0.0
                accum_count = 0
        
        save_path = os.path.join(model_folder, f"model_epoch{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        
        if sample_count > 0:
            print(f'第{epoch}轮训练模型已保存，本轮平均训练损失{epoch_loss / sample_count:.10f}', flush=True)
        else:
            print(f'第{epoch}轮无有效样本，未进行训练。')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"用的设备是: {device}")
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folders", default='pbo_pt_训练集', type=str)
    parser.add_argument("-b", "--batch_size", default=64)
    parser.add_argument("-lr", "--learning_rate", default=1e-3)
    parser.add_argument("-e", "--epoch", default=100)
    parser.add_argument("-m", "--model_folder", default='', type=str)
    # 新增：启动前验证与隔离坏 .pt 文件
    parser.add_argument("--validate-pt", action="store_true", help="启动前逐个加载验证 .pt，跳过损坏文件")
    parser.add_argument("--quarantine", type=str, help="将损坏的 .pt 文件移动到该目录（与 --validate-pt 配合）")
    args = parser.parse_args()
    train(
        args.input_folders,
        device,
        accum_steps=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        epochs=int(args.epoch),
        model_folder=args.model_folder,
        validate_pt=args.validate_pt,
        quarantine_dir=args.quarantine
    )