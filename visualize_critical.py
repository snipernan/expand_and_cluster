import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 硬编码的教师权重 (与 custom_critical 中一致)
w1 = np.array([
    [0.3745401, -0.9507143],
    [0.5986585, -0.1560186],
    [0.4401525, 0.1220382],
    [-0.9772779, -0.3169584]
])
b1 = np.array([0.7319939, -0.1020448, 0.4900855, 0.0564694])

def custom_critical_demo(samples=2000):
    """
    这是您 datagen 中逻辑的副本，用于生成演示数据
    """
    num_neurons = 4
    samples_per_neuron = samples // num_neurons
    X_list = []
    
    # 记录每个点属于哪条线，方便画图上色
    labels = [] 

    range_min, range_max = -4.0, 4.0

    for i in range(num_neurons):
        w_x = w1[i, 0]
        w_y = w1[i, 1]
        b = b1[i]
        
        if abs(w_y) > abs(w_x):
            x1 = np.random.uniform(range_min, range_max, samples_per_neuron)
            x2 = -(w_x * x1 + b) / w_y
        else:
            x2 = np.random.uniform(range_min, range_max, samples_per_neuron)
            x1 = -(w_y * x2 + b) / w_x

        points = np.stack([x1, x2], axis=1)
        X_list.append(points)
        labels.append(np.full(samples_per_neuron, i)) # 标记属于哪个神经元

    X = np.concatenate(X_list, axis=0)
    L = np.concatenate(labels, axis=0)
    return X, L

# --- 开始可视化 ---

# 1. 生成数据
print("正在生成临界点...")
X, labels = custom_critical_demo(samples=1000)

plt.figure(figsize=(10, 8))
colors = ['red', 'green', 'blue', 'orange']

# 2. 画出理论上的 4 条直线 (Wx + b = 0)
x_range = np.linspace(-5, 5, 100)
print("正在绘制理论直线...")
for i in range(4):
    w_x = w1[i, 0]
    w_y = w1[i, 1]
    b = b1[i]
    
    # 直线方程: w_x * x + w_y * y + b = 0
    # y = -(w_x * x + b) / w_y
    if abs(w_y) > 1e-5:
        y_line = -(w_x * x_range + b) / w_y
        plt.plot(x_range, y_line, linestyle='--', color=colors[i], alpha=0.5, label=f'Neuron {i} Line')
    else:
        # 垂直线情况
        x_line = -b / w_x
        plt.vlines(x_line, -5, 5, linestyle='--', color=colors[i], alpha=0.5, label=f'Neuron {i} Line')

# 3. 画出生成的随机点
print("正在绘制样本点...")
for i in range(4):
    # 筛选出属于第 i 个神经元的点
    mask = (labels == i)
    plt.scatter(X[mask, 0], X[mask, 1], s=10, color=colors[i], alpha=0.8, label=f'Samples {i}')

# 4. 设置图表格式
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.title("Visualization of Generated Critical Points\n(Points where Neuron Activation is exactly 0)")
plt.xlabel("Input Dimension 1 (x1)")
plt.ylabel("Input Dimension 2 (x2)")
plt.grid(True, alpha=0.3)
plt.legend()

# 5. 保存图片
save_path = 'critical_points_viz.png'
plt.savefig(save_path)
print(f"\n图片已保存至: {save_path}")
print("您可以下载该图片查看。")

# --- 数值验证 ---
print("\n--- 数值验证 (检查 Wx+b 是否接近 0) ---")
X_tensor = torch.tensor(X, dtype=torch.float32)
W_tensor = torch.tensor(w1, dtype=torch.float32) # [4, 2]
B_tensor = torch.tensor(b1, dtype=torch.float32) # [4]

# 计算 Linear 输出: X @ W.T + B
# 形状: [N, 4]
linear_out = X_tensor @ W_tensor.T + B_tensor

# 检查每一组点对应的神经元输出是否为 0
for i in range(4):
    mask = (labels == i)
    # 取出属于神经元 i 的点，计算它们在神经元 i 上的输出
    out_vals = linear_out[mask, i] 
    mean_abs_val = out_vals.abs().mean().item()
    print(f"神经元 {i} 的样本在该神经元上的激活值 (Wx+b) 平均绝对误差: {mean_abs_val:.6f}")