import torch 
from models import registry 
from foundations import hparams 
import numpy as np # 导入 numpy 以便更好地格式化打印

# --- 1. 加载模型和权重 ---

# 加载教师模型 
model_hparams = hparams.ModelHparams( 
    model_name='students_custom(20)_2_12', 
    model_init='kaiming_normal', 
    batchnorm_init='uniform', 
    act_fun='relu' 
) 
teacher = registry.get(model_hparams, outputs=1, d_in=2) 

# 加载保存的权重 
checkpoint_path = '/home/alvin/expand-and-cluster/data/sims/ec_d592e06c8b/seed_-1/main/model_ep10000_it0.pth'
try:
    checkpoint = torch.load(checkpoint_path) 
    teacher.load_state_dict(checkpoint) 
    print(f"成功加载 checkpoint: {checkpoint_path}\n")
except Exception as e:
    print(f"加载 checkpoint 失败: {e}")
    # 即使加载失败，也继续执行以检查初始化（如果文件不存在）
    pass

# --- 2. 测试前向传播 ---

test_input = torch.randn(10, 2) 
output = teacher(test_input) 
print("--- 前向传播测试 ---")
print(f"Output shape: {output.shape}") 
# 限制输出，只打印前 2 个样本，防止刷屏
print(f"Output values (前2个样本):\n{output.detach().cpu().numpy()[:2]}") 
print(f"Output mean: {output.mean().item()}, std: {output.std().item()}")


# --- 3. (已修复) 检查权重 ---
print("\n--- 权重检查 ---")

# 检查 fc_layers 是否存在于模型上
if hasattr(teacher, 'fc_layers') and len(teacher.fc_layers) > 0:
    
    # 1. 获取第一层 (L1) 模块
    first_layer_module = teacher.fc_layers[0] 
    print(f"第一层 (fc_layers.0) 的模块类型: {type(first_layer_module)}")
    # L1 形状: [Inputs=2, Neurons=12, Students=20]
    print("第一层权重 .fc 的形状:", first_layer_module.fc.shape)
    # L1 形状: [Neurons=12, Students=20]
    print("第一层偏置 .b 的形状:", first_layer_module.b.shape)

    # 2. 获取第二层 (L2) 模块 (如果存在)
    second_layer_module = None
    all_weights_l2 = None
    all_bias_l2 = None
    if len(teacher.fc_layers) > 1:
        second_layer_module = teacher.fc_layers[1]
        print(f"\n第二层 (fc_layers.1) 的模块类型: {type(second_layer_module)}")
        # L2 形状: [L1_Neurons=12, L2_Outputs=1, Students=20]
        print("第二层权重 .fc 的形状:", second_layer_module.fc.shape)
        # L2 形状: [L2_Outputs=1, Students=20]
        print("第二层偏置 .b 的形状:", second_layer_module.b.shape)
        
        # 提取 L2 数据到 numpy
        all_weights_l2 = second_layer_module.fc.detach().cpu().numpy() # 形状 [12, 1, 20]
        all_bias_l2 = second_layer_module.b.detach().cpu().numpy()   # 形状 [1, 20]
    else:
        print("\n模型只有一层。")

    
    # --- 4. 【新功能】打印所有学生的 L1 和 L2 详细权重 ---
    print("\n\n--- 详细权重 (所有学生, L1 和 L2) ---")
    
    # 将 L1 完整的权重和偏置张量提取到 numpy
    # .fc 形状: [2, 12, 20]
    # .b 形状: [12, 20]
    all_weights_l1 = first_layer_module.fc.detach().cpu().numpy()
    all_bias_l1 = first_layer_module.b.detach().cpu().numpy()

    # 从权重形状确定学生数量 (第 2 轴，即第三个维度)
    num_students = all_weights_l1.shape[2]
    num_neurons_l1 = all_weights_l1.shape[1] # 应该是 12
    
    # 设置 numpy 打印选项，使其更易读
    np.set_printoptions(precision=6, suppress=True)

    # --------------------------------------------------
    # *** 修改：在同一个学生循环中打印 L1 和 L2 ***
    # --------------------------------------------------

    # 遍历所有学生
    for s_idx in range(num_students):
        print(f"\n==================== 学生 #{s_idx} ====================")
        
        # --- (A) 打印第一层 (L1) ---
        
        # 提取当前学生的 L1 权重和偏置
        student_weights_l1 = all_weights_l1[:, :, s_idx] # 形状 [2, 12]
        student_bias_l1 = all_bias_l1[:, s_idx]     # 形状 [12]

        print(f"--- 学生 #{s_idx}, 第一层 (fc_layers.0) 的 {num_neurons_l1} 个神经元 ---\n")
        print(f"格式: 神经元: [Input 1 权重] [Input 2 权重]   [偏置]")
        print("-----------------------------------------------------------")
        
        # 遍历 L1 的 12 个神经元
        for i in range(num_neurons_l1):
            w1 = student_weights_l1[0, i] # 来自 Input 1 的权重
            w2 = student_weights_l1[1, i] # 来自 Input 2 的权重
            b = student_bias_l1[i]        # L1 偏置
            
            print(f"神经元 {i+1:02d}: {w1:12.6f} {w2:12.6f} {b:12.6f}")
        print("-----------------------------------------------------------")


        # --- (B) 打印第二层 (L2) (如果存在) ---
        
        if second_layer_module is not None:
            # 提取当前学生的 L2 权重和偏置
            # .fc 形状 [12, 1, 20] -> [12, 1]
            student_weights_l2 = all_weights_l2[:, :, s_idx] 
            # .b 形状 [1, 20] -> [1]
            student_bias_l2 = all_bias_l2[:, s_idx]     
            
            print(f"\n--- 学生 #{s_idx}, 第二层 (fc_layers.1) 输出层 ---")
            
            # L2 权重 (从 L1 的 12 个神经元到 1 个输出神经元)
            # 形状 [12, 1]，我们将其 .flatten() 变为 [12] 以便打印
            print(f"权重 (来自 L1 的 12 个神经元，连接到唯一的输出):")
            print(student_weights_l2.flatten()) 
            
            # L2 偏置 (1 个输出神经元的偏置)
            # 形状 [1]，我们打印第 0 个元素
            print(f"\n偏置 (L2 的 1 个输出):")
            print(f"{student_bias_l2[0]:12.6f}")
            print("-----------------------------------------------------------")

    print(f"\n--- 所有 {num_students} 名学生的权重打印完毕 ---")
    # --------------------------------------------------
    # *** 修改结束 ***
    # --------------------------------------------------

    # (原始的第 5 部分已被合并到上面的循环中，故删除)

else:
    print("错误：在 'teacher' 模型上未找到 'fc_layers' 属性。")


# ============================================================
#  (4) 教师 vs 20 个学生 —— 相同输入下的输出对比
# ============================================================

print("\n\n==================== 教师 vs 学生 输出对比 ====================\n")

torch.manual_seed(0)
compare_input = torch.randn(5, 2)   # 5 个样本，用于对比

# --- 教师输出 ---
try:
    # 加载教师模型（路径改成你的教师checkpoint路径）
    teacher_hparams = hparams.ModelHparams(
        model_name='custom_teacher',
        model_init='kaiming_normal',
        batchnorm_init='uniform',
        act_fun='relu'
    )
    teacher_model = registry.get(teacher_hparams, outputs=1, d_in=2)
    teacher_ckpt = torch.load('/home/alvin/expand-and-cluster/data/sims/train_custom_teacher/seed_0/main/model_ep0_it0.pth')
    teacher_model.load_state_dict(teacher_ckpt)
    teacher_model.eval()
except:
    print("⚠️ 无法重新加载教师模型，请检查路径！")
    raise

with torch.no_grad():
    teacher_out = teacher_model(compare_input).detach().cpu().numpy().flatten()

print("教师输出:")
print(teacher_out)
print("-----------------------------------------------------------\n")


# --- 学生输出 ---
student_outputs = []

with torch.no_grad():
    for s_idx in range(num_students):
        # 取第 s_idx 个学生的 L1 权重 / 偏置
        W1 = torch.tensor(all_weights_l1[:, :, s_idx].T)  # 形状 [12, 2]
        b1 = torch.tensor(all_bias_l1[:, s_idx])          # 形状 [12]

        # L2
        W2 = torch.tensor(all_weights_l2[:, :, s_idx].T)  # [1, 12]
        b2 = torch.tensor(all_bias_l2[:, s_idx])          # [1]

        # 手动前向计算：relu(W1 x + b1) → W2 h + b2
        h = torch.relu(compare_input @ W1.T + b1)
        out = (h @ W2.T + b2).detach().cpu().numpy().flatten()

        student_outputs.append(out)

        print(f"学生 #{s_idx} 输出:")
        print(out)
        print("-----------------------------------------------------------")

# =============================================================
#  输出整合成对比表格
# =============================================================
print("\n==================== 输出矩阵（教师 + 20 学生） ====================\n")

import numpy as np

all_out = np.vstack([teacher_out] + student_outputs)  # 21 x 5
print(all_out)

print("\n行 0 = 教师，行 1-20 = 各学生\n")
