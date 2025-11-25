"""
加载并检查 'custom_teacher' 权重的脚本
(格式模仿学生网络)
"""
import torch
from models import registry
from foundations import hparams
import numpy as np
import os

# --- 1. 加载模型 (必须与保存脚本中的设置完全一致) ---

# 1a. 定义 Hparams (与保存脚本中相同)
model_hparams = hparams.ModelHparams(
    model_name='custom_teacher',
    model_init='kaiming_normal',
    batchnorm_init='uniform',
    act_fun='relu'
)

# 1b. 创建模型实例
# 关键：根据您在保存脚本中定义的 w1 形状 [4, 2] (4个神经元, 2个输入)
# 我们必须在这里明确指定 d_in=2，以便加载 state_dict
try:
    model = registry.get(model_hparams, outputs=1, d_in=2)
except Exception as e:
    print(f"创建模型 'custom_teacher' 失败。")
    print("请确保 'custom_teacher' 已在 models/registry.py 中注册，")
    print(f"并且它是一个能接受 d_in 和 outputs 参数的标准模型。错误: {e}")
    exit()


# 1c. 定义 checkpoint 路径 (与保存脚本中相同)
checkpoint_path = '/home/alvin/expand-and-cluster/data/sims/train_custom_teacher/seed_0/main/model_ep0_it0.pth'

# 1d. 加载保存的 state_dict
if not os.path.exists(checkpoint_path):
    print(f"错误：找不到 Checkpoint 文件: {checkpoint_path}")
    print("请先运行您的 '保存脚本' 来生成此文件。")
    exit()

try:
    checkpoint = torch.load(checkpoint_path) 
    model.load_state_dict(checkpoint) 
    print(f"成功加载自定义教师 checkpoint: {checkpoint_path}\n")
    model.eval() # 设置为评估模式
except Exception as e:
    print(f"加载 checkpoint 失败: {e}")
    print("这通常发生在模型架构 (d_in=2) 与保存的 state_dict 键或形状不匹配时。")
    exit()


# --- 2. 打印权重 (仿照学生格式) ---

print("\n\n--- 详细权重 (自定义教师模型) ---")

# 设置 numpy 打印选项 (使用7位小数，匹配您的保存脚本)
np.set_printoptions(precision=7, suppress=True)

# 教师模型只有一个 "学生" 实例
print(f"\n==================== 教师模型 (Teacher Model) ====================")

with torch.no_grad():
    # --- (A) 打印第一层 (L1) ---
    
    # 注意：标准模型的权重访问器是 .weight 和 .bias
    # 形状: [num_neurons, input_features] -> [4, 2]
    l1_weights = model.fc_layers[0].weight.detach().cpu().numpy() 
    # 形状: [num_neurons] -> [4]
    l1_bias = model.fc_layers[0].bias.detach().cpu().numpy()     
    num_neurons_l1 = l1_weights.shape[0] # 应该是 4

    print(f"--- 教师, 第一层 (fc_layers[0]) 的 {num_neurons_l1} 个神经元 ---\n")
    print(f"格式: 神经元: [Input 1 权重] [Input 2 权重]   [偏置]")
    print("-----------------------------------------------------------")

    # 遍历 L1 的 4 个神经元
    for i in range(num_neurons_l1):
        # l1_weights[i, 0] = 来自 Input 1 的权重
        # l1_weights[i, 1] = 来自 Input 2 的权重
        w1 = l1_weights[i, 0] 
        w2 = l1_weights[i, 1] 
        b = l1_bias[i]        # L1 偏置
        
        # 使用 7 位小数进行格式化
        print(f"神经元 {i+1:02d}: {w1:12.7f} {w2:12.7f} {b:12.7f}")
    print("-----------------------------------------------------------")


    # --- (B) 打印第二层 (L2) (输出层) ---
    
    # 注意：第二层 (输出层) 在 'custom_teacher' 中可能被命名为 'model.fc'
    if hasattr(model, 'fc'):
        # 形状: [output_features, num_neurons_l1] -> [1, 4]
        l2_weights = model.fc.weight.detach().cpu().numpy() 
        # 形状: [output_features] -> [1]
        l2_bias = model.fc.bias.detach().cpu().numpy()     
        
        print(f"\n--- 教师, 第二层 (fc) 输出层 ---")
        
        # l2_weights 形状是 [1, 4]。我们将其 .flatten() 
        # 以打印来自 L1 的 4 个神经元的权重。
        print(f"权重 (来自 L1 的 4 个神经元，连接到唯一的输出):")
        print(l2_weights.flatten()) 
        
        # 偏置 (1 个输出神经元的偏置)
        print(f"\n偏置 (L2 的 1 个输出):")
        print(f"{l2_bias[0]:12.7f}")
        print("-----------------------------------------------------------")
    else:
        print("\n未在模型上找到 'fc' (输出层)。")

print(f"\n--- 教师模型权重打印完毕 ---")