import torch
import matplotlib.pyplot as plt
import numpy as np

# (假设模型文件位于您指定的路径)
file_name = '/home/alvin/expand-and-cluster/data/sims/ec_fc608b8f2e/seed_-1/main/clustering_31924a37bd/finetune_checkpoints/L1/model_ep5000_it0.pth'

# 1. 加载模型权重
#    map_location='cpu' 确保即使模型是在GPU上训练的，也能在CPU上加载
state_dict = torch.load(file_name, map_location=torch.device('cpu'))

# 2. 提取第一个隐藏层的权重
weights = state_dict['fc_layers.0.weight']
# 此时 'weights' 的形状是 torch.Size([20, 784])

# 3. 获取神经元数量
num_neurons = weights.shape[0] # 结果是 20
plot_count = num_neurons

# 4. 决定网格大小 (例如 5x4 或 5x5)
grid_size = int(np.ceil(np.sqrt(plot_count))) # 结果是 5

# 5. 创建子图画布
#    figsize=(10, 10) 使图像足够大，看得清楚
fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))

# 6. 将 'axes' 展平，以便使用单个索引 (i) 循环
axes = axes.flatten()

# 7. 循环遍历每个神经元并绘图
for i in range(plot_count):
    
    # 8. 关键步骤：重塑
    #    .detach().numpy() 是为了将torch张量转为numpy数组
    #    .reshape(28, 28) 将 1x784 向量转为 28x28 矩阵
    neuron_weights = weights[i].detach().numpy().reshape(28, 28)
    
    # 9. 绘图
    ax = axes[i]
    # 使用 imshow 将矩阵显示为图像，cmap='gray' 使用灰度图
    im = ax.imshow(neuron_weights, cmap='gray')
    ax.set_title(f"Neuron {i}")
    ax.axis('off') # 关闭坐标轴

# 10. 关闭多余的空子图 (如果创建了 5x5=25 个，但只有 20 个图)
for j in range(plot_count, len(axes)):
    axes[j].axis('off')

plt.suptitle("Visualization of First Hidden Layer Weights (fc_layers.0)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# 11. 保存图像
output_image_path = 'first_layer_weights_visualization.png'
plt.savefig(output_image_path)