import torch
import numpy as np
import os
from models import registry
from foundations import hparams

# ============================================================
#   (1) 定义对比输入
# ============================================================
torch.manual_seed(0)
compare_input = torch.randn(5, 2) 
print(f"--- 对比输入 (Shape: {compare_input.shape}) ---\n{compare_input.numpy()}\n")

# ============================================================
#   (2) 加载模型 B (原始教师 - 标准方式) + [新增] 打印权重
# ============================================================
print("=" * 60)
print("加载模型 B (原始教师 - 标准加载)")
print("=" * 60)
model_B_path = '/home/alvin/expand-and-cluster/data/sims/train_custom_teacher/seed_0/main/model_ep0_it0.pth'
teacher_B_out = None

try:
    teacher_hparams = hparams.ModelHparams(
        model_name='custom_teacher', model_init='kaiming_normal',
        batchnorm_init='uniform', act_fun='relu'
    )
    teacher_model_B = registry.get(teacher_hparams, outputs=1, d_in=2)
    teacher_ckpt = torch.load(model_B_path)
    teacher_model_B.load_state_dict(teacher_ckpt)
    teacher_model_B.eval()
    print(f"成功加载模型 B: {model_B_path}\n")

    # ----- [新增部分] 提取并打印模型 B 的权重结构 -----
    # 假设模型是简单的 MLP，我们可以按顺序获取参数
    # param_list[0]: Layer 1 Weights [Hidden, Input]
    # param_list[1]: Layer 1 Bias    [Hidden]
    # param_list[2]: Layer 2 Weights [Output, Hidden]
    # param_list[3]: Layer 2 Bias    [Output]
    params_B = list(teacher_model_B.parameters())
    
    W1_B = params_B[0].detach().cpu().numpy() # Shape: [Hidden, Input]
    b1_B = params_B[1].detach().cpu().numpy() # Shape: [Hidden]
    W2_B = params_B[2].detach().cpu().numpy() # Shape: [1, Hidden]
    b2_B = params_B[3].detach().cpu().numpy() # Shape: [1]

    print("=" * 60)
    print("模型 B (原始教师) 的权重结构")
    print("=" * 60)

    # 第一层权重
    print(f"\n--- 第一层 (Hidden) 的 {W1_B.shape[0]} 个神经元 ---\n")
    print("格式: 神经元: [Input 1 权重] [Input 2 权重]   [偏置]")
    print("-" * 59)
    for i in range(W1_B.shape[0]):
        # W1_B 的每一行对应一个神经元的输入权重
        weights_str = "   ".join([f"{w:11.7f}" for w in W1_B[i, :]])
        bias_str = f"{b1_B[i]:11.7f}"
        print(f"神经元 {i+1:02d}:  {weights_str}  {bias_str}")
    print("-" * 59)

    # 第二层权重
    print(f"\n--- 第二层 (Output) 输出层 ---")
    print(f"权重 (来自 L1 的 {W1_B.shape[0]} 个神经元，连接到唯一的输出):")
    # 展平以便打印，匹配模型 A 的格式
    print(W2_B.flatten()) 
    print(f"\n偏置 (L2 的 {len(b2_B)} 个输出):")
    for i, bias in enumerate(b2_B):
        print(f"  {bias:.7f}")
    print("-" * 59 + "\n")
    # -----------------------------------------------------

    with torch.no_grad():
        teacher_B_out = teacher_model_B(compare_input).detach().cpu().numpy().flatten()

except Exception as e:
    print(f"⚠️ 无法加载模型 B (原始教师): {e}\n")
    import traceback
    traceback.print_exc()


# ============================================================  
#   (3) 加载模型 A (重构教师 - 手动方式)  
# ============================================================  
print("=" * 60)  
print("加载模型 A (重构教师 - 手动加载)")  
print("=" * 60)  
  
model_A_path = "/home/alvin/expand-and-cluster/data/sims/ec_5e03884262/seed_-1/main/clustering_995dc42cbd/reconstructed_model/model_ep5000_it0.pth"  
affine_path = "/home/alvin/expand-and-cluster/data/sims/ec_5e03884262/seed_-1/main/clustering_995dc42cbd/reconstructed_model/affine.pth"  
teacher_A_out = None  
  
try:  
    # 1. 加载模型权重  
    state_dict_A = torch.load(model_A_path, map_location='cpu')  
    print(f"成功加载模型 A 的 state_dict: {model_A_path}\n")  
  
    # 2. 加载线性分量  
    thetas = torch.load(affine_path, map_location='cpu')  
    print(f"成功加载线性分量: {affine_path}")  
    print(f"线性分量形状: {thetas.shape}")  
      
    # 修正线性分量形状  
    if thetas.shape[1] != 1:  
        print(f"警告: thetas形状为 {thetas.shape},预期为 [3, 1]")  
        print(f"使用第一列作为线性分量\n")  
        thetas = thetas[:, 0:1]  # 只取第一列  
  
    # 3. 提取权重和偏置张量 (这部分必须在使用 W1_A 之前!)  
    fc0_fc_tensor = state_dict_A['fc_layers.0.fc']  
    fc0_b_tensor = state_dict_A['fc_layers.0.b']  
    fc1_fc_tensor = state_dict_A['fc_layers.1.fc']  
    fc1_b_tensor = state_dict_A['fc_layers.1.b']  
  
    # 使用 .clone().detach() 来创建新的无梯度副本  
    # L1 权重: [2, 4, N] -> [2, 4] -> .T -> [4, 2]  
    W1_A = fc0_fc_tensor[:, :, 0].T.clone().detach()  
    # L1 偏置: [4, N] -> [4]  
    b1_A = fc0_b_tensor[:, 0].clone().detach()  
      
    # L2 权重: [4, 1, N] -> [4, 1] -> .T -> [1, 4]  
    W2_A = fc1_fc_tensor[:, :, 0].T.clone().detach()  
    # L2 偏置: [1, N] -> [1]  
    b2_A = fc1_b_tensor[:, 0].clone().detach()  
    
    # ===== 添加权重输出部分 =====  
    print("\n" + "=" * 60)  
    print("重建模型 A 的权重")  
    print("=" * 60)  
    
    # 第一层权重  
    print(f"\n--- 第一层 (fc_layers[0]) 的 {W1_A.shape[0]} 个神经元 ---\n")  
    print("格式: 神经元: [Input 1 权重] [Input 2 权重]   [偏置]")  
    print("-" * 59)  
    for i in range(W1_A.shape[0]):  
        weights_str = "   ".join([f"{w:11.7f}" for w in W1_A[i, :]])  
        bias_str = f"{b1_A[i]:11.7f}"  
        print(f"神经元 {i+1:02d}:  {weights_str}  {bias_str}")  
    print("-" * 59)  
    
    # 第二层权重  
    print(f"\n--- 第二层 (fc) 输出层 ---")  
    print(f"权重 (来自 L1 的 {W1_A.shape[0]} 个神经元，连接到唯一的输出):")  
    print(W2_A.squeeze().numpy())  
    print(f"\n偏置 (L2 的 {len(b2_A)} 个输出):")  
    for i, bias in enumerate(b2_A):  
        print(f"  {bias:.7f}")  
    print("-" * 59)  
    
    # 线性分量  
    print(f"\n--- 线性分量 (thetas) ---")  
    print(f"形状: {thetas.shape}")  
    print("参数 (包含输入权重和偏置项):")  
    print(thetas.squeeze().detach().numpy())  # 添加 .detach()  
    print("-" * 59 + "\n")


    # 4. 准备输入 (添加偏置项用于线性分量)  
    x = torch.cat([compare_input, torch.ones(compare_input.shape[0], 1)], dim=1)  
  
    # 5. 完整的前向计算 (神经网络 + 线性分量)  
    with torch.no_grad():  
        # 神经网络部分  
        h_A = torch.relu(compare_input @ W1_A.T + b1_A)  
        nn_out = (h_A @ W2_A.T + b2_A)  
          
        # 线性分量部分  
        linear_out = (x @ thetas).squeeze()  
          
        # 最终输出 = 神经网络输出 + 线性修正  
        out_A = nn_out.squeeze() + linear_out  
      
    teacher_A_out = out_A.cpu().numpy().flatten()  
  
except Exception as e:  
    print(f"⚠️ 无法加载模型 A (重构教师): {e}")  
    print("请确保路径和字典键正确。\n")  
    import traceback  
    traceback.print_exc()

# ============================================================
#   (4) 和 (5) 对比输出 (与之前完全相同)
# ============================================================
# (这部分代码与之前完全相同)
print("\n\n==================== 输出对比 (5 个样本) ====================\n")
if teacher_B_out is not None:
    print("模型 B (原始教师) 输出:")
    print(teacher_B_out)
    print("-----------------------------------------------------------")
else:
    print("模型 B (原始教师) 未能计算输出。\n")

if teacher_A_out is not None:
    print("模型 A (重构教师) 输出:")
    print(teacher_A_out)
    print("-----------------------------------------------------------")
else:
    print("模型 A (重构教师) 未能计算输出。\n")

if teacher_A_out is not None and teacher_B_out is not None:
    print("\n==================== 输出矩阵对比 ====================\n")
    all_out = np.vstack([teacher_B_out, teacher_A_out])
    print("行 0 = 模型 B (原始教师)")
    print("行 1 = 模型 A (重构教师)")
    np.set_printoptions(precision=7, suppress=True)
    print(all_out)
    diff = np.abs(teacher_B_out - teacher_A_out)
    print("\n绝对差异 (B - A):")
    print(diff)
    print(f"\n平均绝对差异: {np.mean(diff):.7f}")
else:
    print("\n无法生成对比矩阵，因为至少有一个模型加载失败。")
    