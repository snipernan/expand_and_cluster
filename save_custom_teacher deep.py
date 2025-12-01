"""保存自定义教师模型的脚本"""
import os
import torch
import numpy as np
from platforms.platform import get_platform
from foundations import hparams
from models import registry


# =======================
#  1. 第一层隐藏层 (2→4)
# =======================
w_and_b1 = np.array([
    [0.3745401, -0.9507143, 0.7319939],
    [0.5986585, -0.1560186, -0.1020448],
    [0.4401525, 0.1220382, 0.4900855],
    [-0.9772779, -0.3169584, 0.0564694]
])
w1 = w_and_b1[:, :2]   # [4,2]
b1 = w_and_b1[:, 2]    # [4]


# =======================
#  2. 第二层隐藏层 (4→4) -- Kaiming Normal 初始化
# =======================

rng = np.random.default_rng(seed=2025)

fan_in = 4
std = np.sqrt(2.0 / fan_in)   # = sqrt(0.5)

# Kaiming normal for weights
w2_hid = rng.normal(loc=0.0, scale=std, size=(4, 4))

# Kaiming normal for biases (通常偏置可以设为 0，但你要求随机也按相同规则）
b2_hid = rng.normal(loc=0.0, scale=std, size=(4,))


# =======================
#  3. 输出层 (4→1)
# =======================
w3 = np.array([
    [-0.518608],
    [ 0.294154],
    [-0.217739],
    [ 0.760492]
])  # [4,1]
b3 = np.array([-1.4785862])  # [1]


# =======================
#  模型超参数
# =======================
model_hparams = hparams.ModelHparams(
    model_name='custom_teacher_deep',
    model_init='kaiming_normal',
    batchnorm_init='uniform',
    act_fun='relu'
)

dataset_hparams = hparams.DatasetHparams(
    dataset_name='mnist',   # ⚠️你必须确保 custom_teacher_deep 输入维度是 2
    batch_size=128
)

training_hparams = hparams.TrainingHparams(
    optimizer_name='sgd',
    lr=0.1,
    training_steps='1ep',
)

# =======================
#  创建 2→4→4→1 模型
# =======================
model = registry.get(model_hparams, outputs=1)


# =======================
#  覆盖权重
# =======================
with torch.no_grad():
    # 第一层 2→4
    model.fc_layers[0].weight.data = torch.tensor(w1, dtype=torch.float32)
    model.fc_layers[0].bias.data = torch.tensor(b1, dtype=torch.float32)

    # 第二层 4→4
    model.fc_layers[1].weight.data = torch.tensor(w2_hid, dtype=torch.float32)
    model.fc_layers[1].bias.data = torch.tensor(b2_hid, dtype=torch.float32)

    # 最终输出层 4→1
    model.fc.weight.data = torch.tensor(w3.T, dtype=torch.float32)   # [1,4]
    model.fc.bias.data = torch.tensor(b3, dtype=torch.float32)


# =======================
#  保存模型文件
# =======================
root_dir = '/home/alvin/expand-and-cluster/data/sims'
save_dir = os.path.join(root_dir, "train_custom_teacher_deep", "seed_0", "main")
os.makedirs(save_dir, exist_ok=True)

torch.save(model.state_dict(), os.path.join(save_dir, "model_ep0_it0.pth"))

hparams_dict = {
    'model_hparams': model_hparams,
    'dataset_hparams': dataset_hparams,
    'training_hparams': training_hparams,
}
torch.save(hparams_dict, os.path.join(save_dir, "hparams_dict"))

print(f"模型 (2→4→4→1) 已保存到: {save_dir}")
print("注意：你的 custom_teacher_deep 必须定义两层隐藏层。")
