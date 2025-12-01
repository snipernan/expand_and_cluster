"""
 # Created on 13.09.23
 #
 # Author: Flavio Martinelli, EPFL
 #
 # Adapted from: https://github.com/facebookresearch/open_lth
 #
 # Description: Contains all data generator policies functions
 #
"""
import os
import torch
import torchvision

from PIL import Image
from datasets.cifar10 import CIFAR10
from platforms.platform import get_platform

# TODO: might be turned into a basic class to enforce function signatures
# TODO: implement augmentation for the CIFAR10 case


def mnist(augment=None, d_in=None):
    train_set = torchvision.datasets.MNIST(
        train=True, root=os.path.join(get_platform().dataset_root, 'mnist'), download=True)
    transforms = [torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize(mean=[0.1307], std=[0.3081])]
    X = []
    for im in train_set.data.numpy():
        im = Image.fromarray(im, mode='L')
        for t in transforms:
            im = t(im)
        X.append(im)
    X = torch.concat(X)
    return X


def mnist_conv(augment=None, d_in=None):
    X = mnist(augment, d_in)
    return X.unsqueeze(1)


def cifar10(augment=None, d_in=None):
    # augment = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomCrop(32, 4)]
    train_set = CIFAR10(train=True, root=os.path.join(get_platform().dataset_root, 'cifar10'), download=True)
    transforms = [torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    X = []
    for im in train_set.data:
        for t in transforms:
            im = t(im)
        X.append(im)
    X = torch.stack(X, dim=0)
    return X


def cifar10_conv(augment=None, d_in=None):
    X = cifar10(augment, d_in)
    return X


def fashion_mnist(augment=None, d_in=None):
    train_set = torchvision.datasets.FashionMNIST(
        train=True, root=os.path.join(get_platform().dataset_root, 'fashion_mnist'), download=True)
    transforms = [torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize(mean=[0.5], std=[0.5])]
    X = []
    for im in train_set.data.numpy():
        im = Image.fromarray(im, mode='L')
        for t in transforms:
            im = t(im)
        X.append(im)
    X = torch.concat(X)
    return X

def custom_2d(augment=None, d_in=None, samples=8000): # 1. 添加 "samples" 参数
    X = torch.randn(samples, 2) 
    return X

def custom_critical(augment=None, d_in=None, samples=10000, epsilon=0.5):
    """
    生成位于教师网络第一层神经元 '临界带' (Wx+b ≈ 0) 附近的数据点。
    epsilon: 控制临界带的宽度。
             epsilon=0.0 -> 精确的临界线 (可能导致优化困难)
             epsilon=0.1 -> 临界带 (推荐，包含边界附近的激活和非激活点)
    """
    import torch
    import numpy as np

    # 1. 硬编码教师网络的权重和偏置
    # 形状: [4, 2]
    w1 = np.array([
        [0.3745401, -0.9507143],
        [0.5986585, -0.1560186],
        [0.4401525, 0.1220382],
        [-0.9772779, -0.3169584]
    ])
    # 形状: [4,]
    b1 = np.array([0.7319939, -0.1020448, 0.4900855, 0.0564694])

    # 2. 准备生成
    num_neurons = 4
    samples_per_neuron = samples // num_neurons
    remainder = samples % num_neurons
    
    X_list = []

    # 设定生成数据的范围
    range_min, range_max = -4.0, 4.0

    for i in range(num_neurons):
        w_x = w1[i, 0]
        w_y = w1[i, 1]
        b = b1[i]
        
        current_samples = samples_per_neuron + (1 if i < remainder else 0)
        
        if current_samples == 0:
            continue

        # --- [关键修改] 生成 epsilon 扰动 ---
        # 我们希望 Wx + b = -noise
        # 这样，激活值 (Wx+b) 将在 [-epsilon, epsilon] 之间波动
        noise = np.random.uniform(-epsilon, epsilon, current_samples)

        # 生成策略：
        if abs(w_y) > abs(w_x):
            # w_y 足够大，采样 x1，算 x2
            x1 = np.random.uniform(range_min, range_max, current_samples)
            # 公式修改：加入 noise 扰动
            # 原理推导: w_x*x1 + w_y*x2 + b = -noise  =>  w_y*x2 = -(w_x*x1 + b + noise)
            x2 = -(w_x * x1 + b + noise) / w_y
        else:
            # w_x 比较大，采样 x2，算 x1
            x2 = np.random.uniform(range_min, range_max, current_samples)
            # 公式修改：加入 noise 扰动
            x1 = -(w_y * x2 + b + noise) / w_x

        points = np.stack([x1, x2], axis=1)
        X_list.append(torch.tensor(points, dtype=torch.float32))

    # 3. 合并
    X = torch.cat(X_list, dim=0)

    # 4. 打乱顺序
    perm = torch.randperm(X.size(0))
    X = X[perm]

    return X