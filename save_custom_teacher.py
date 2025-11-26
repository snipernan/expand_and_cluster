"""保存自定义教师模型的脚本"""
import os
import torch
import numpy as np
from platforms.platform import get_platform
from foundations import hparams
from models import registry

# --- 建议的、无冗余的参数 ---

# 1. 隐藏层 (4 个神经元, 每个 [2个权重 + 1个偏置])
w_and_b1 = np.array([
    [0.3745401, -0.9507143, 0.7319939],
    [0.5986585, -0.1560186, -0.1020448],
    [0.4401525, 0.1220382, 0.4900855],
    [-0.9772779, -0.3169584, 0.0564694]
])

# 真正的 w1 (权重) 和 b1 (偏置)
w1 = w_and_b1[:, :2]  # 形状: [4, 2]
b1 = w_and_b1[:, 2]   # 形状: [4,]

w2 = np.array([
    [-0.518608],
    [ 0.294154],
    [-0.217739],
    [ 0.760492]
])  # 形状: [4, 1]

b2 = np.array([-1.4785862]) # 形状: [1,]


# 创建模型超参数
model_hparams = hparams.ModelHparams(
    model_name='custom_teacher',
    model_init='kaiming_normal',
    batchnorm_init='uniform',
    act_fun='sigmoid'
)

# 创建模型时明确指定outputs  


# 警告：这里的 'mnist' (784个输入) 与您的 2 输入权重 (w1) 不匹配。
# 您必须确保 'custom_teacher' 模型在创建时
# 知道第一层的输入特征是 2。
dataset_hparams = hparams.DatasetHparams(
    dataset_name='mnist', # <--- 警告：请确保这与您的模型结构 (2-in) 兼容
    batch_size=128
)

training_hparams = hparams.TrainingHparams(
    optimizer_name='sgd',
    lr=0.1,
    training_steps='1ep',
)

# 创建模型 (假设 'custom_teacher' 接受 2 个输入)
# 您可能需要像这样传递输入维度：
# model = registry.get(model_hparams, input_shape=(2,), outputs=1)
# 或者确保 'custom_teacher' 默认是 2-in
model = registry.get(model_hparams, outputs=1)  
  
with torch.no_grad():  
    model.fc_layers[0].weight.data = torch.tensor(w1, dtype=torch.float32)  
    model.fc_layers[0].bias.data = torch.tensor(b1, dtype=torch.float32)  
    model.fc.weight.data = torch.tensor(w2.T, dtype=torch.float32)  # [1, 4]  
    model.fc.bias.data = torch.tensor(b2, dtype=torch.float32)  # [1] 

# 创建保存目录
# --- 修复 ---
# 手动设置根目录，因为 get_platform() 无法识别本地环境
# 你的路径就是 /home/expand-and-cluster
root_dir = '/home/alvin/expand-and-cluster/data/sims' 
save_dir = os.path.join(root_dir, "train_custom_teacher_s", "seed_0", "main")
# --- 结束修复 ---
os.makedirs(save_dir, exist_ok=True)

# 保存模型
torch.save(model.state_dict(), os.path.join(save_dir, "model_ep0_it0.pth"))

# 保存超参数
hparams_dict = {
    'model_hparams': model_hparams,
    'dataset_hparams': dataset_hparams,
    'training_hparams': training_hparams
}
torch.save(hparams_dict, os.path.join(save_dir, "hparams_dict"))

print(f"模型 (2->4->1) 已保存到: {save_dir}")
print("警告：请确保 'custom_teacher' 模型注册表和 'dataset_hparams' 匹配 2 维输入。")