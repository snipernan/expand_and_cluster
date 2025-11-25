"""
加载并打印 'custom_teacher_deep' 权重 (支持多隐藏层)
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

# ---- 3. 加载 checkpoint ----
checkpoint_path = "/home/alvin/expand-and-cluster/data/sims/train_custom_teacher_deep/seed_0/main/model_ep0_it0.pth"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"找不到模型文件: {checkpoint_path}")

model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

# ---- 4. 打印权重 ----
np.set_printoptions(precision=7, suppress=True)
print("\n--- 详细权重 (自定义教师模型) ---\n")
print("==================== 教师模型 (Teacher Model) ====================\n")

with torch.no_grad():
    # 遍历每一层隐藏层
    for layer_idx, layer in enumerate(model.fc_layers):
        num_neurons = layer.weight.shape[0]
        input_size = layer.weight.shape[1]
        print(f"--- 教师, 第 {layer_idx+1} 层 (fc_layers[{layer_idx}]) 的 {num_neurons} 个神经元 ---\n")
        print(f"格式: 神经元: " + " ".join([f"[Input {i+1}]" for i in range(input_size)]) + "   [偏置]")
        print("-----------------------------------------------------------")
        for i in range(num_neurons):
            weights = layer.weight[i].cpu().numpy()
            bias = layer.bias[i].item()
            weights_str = "    ".join(f"{w: .7f}" for w in weights)
            print(f"神经元 {i+1:02d}:   {weights_str}   {bias: .7f}")
        print("-----------------------------------------------------------\n")

    # 输出层
    if hasattr(model, 'fc'):
        out_weights = model.fc.weight.detach().cpu().numpy().flatten()
        out_bias = model.fc.bias.item()
        print("--- 教师, 输出层 (fc) ---")
        print(f"权重 (来自最后一隐藏层的 {len(out_weights)} 个神经元，连接到唯一输出):")
        print("[" + " ".join(f"{w: .7f}" for w in out_weights) + "]\n")
        print("偏置 (输出层):")
        print(f"  {out_bias: .7f}")
        print("-----------------------------------------------------------")

print("\n--- 教师模型权重打印完毕 ---")
