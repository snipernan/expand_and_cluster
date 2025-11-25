"""
# Created on 10.09.23
#
# Author: Flavio Martinelli, EPFL.
#
# Adapted from: https://github.com/facebookresearch/open_lth
#
# Description: Contains possible initializers
#
"""

import torch
import models.students_mnist_lenet, models.students_cifar_lenet
from models import students_custom  # <--- 1. 导入您的自定义模型


def binary(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(w.weight)
        sigma = w.weight.data.std()
        w.weight.data = torch.sign(w.weight.data) * sigma

    # --- 2. 添加了对 students_custom 的检查 (并修复了 w.weight bug) ---
    if isinstance(w, models.students_mnist_lenet.Model.ParallelFCModule) or \
       isinstance(w, models.students_mnist_lenet.Model.InitialParallelFCModule) or \
       isinstance(w, students_custom.Model.ParallelFCModule) or \
       isinstance(w, students_custom.Model.InitialParallelFCModule):
        torch.nn.init.kaiming_normal_(w.fc)
        sigma = w.fc.data.std()
        w.fc.data = torch.sign(w.fc.data) * sigma  # <--- (修复了 .weight -> .fc)


def kaiming_normal(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(w.weight)

    # --- 3. 添加了对 students_custom 的检查 ---
    if isinstance(w, models.students_mnist_lenet.Model.ParallelFCModule) or \
       isinstance(w, models.students_mnist_lenet.Model.InitialParallelFCModule) or \
       isinstance(w, models.students_cifar_lenet.Model.ParallelFCModule) or \
       isinstance(w, models.students_cifar_lenet.Model.InitialParallelFCModule) or \
       isinstance(w, students_custom.Model.ParallelFCModule) or \
       isinstance(w, students_custom.Model.InitialParallelFCModule):
        torch.nn.init.kaiming_normal_(w.fc)


def kaiming_uniform(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(w.weight)

    # --- 4. 添加了对 students_custom 的检查 ---
    if isinstance(w, models.students_mnist_lenet.Model.ParallelFCModule) or \
       isinstance(w, models.students_mnist_lenet.Model.InitialParallelFCModule) or \
       isinstance(w, students_custom.Model.ParallelFCModule) or \
       isinstance(w, students_custom.Model.InitialParallelFCModule):
        torch.nn.init.kaiming_uniform_(w.fc)


def orthogonal(w):
    if isinstance(w, torch.nn.Linear) or isinstance(w, torch.nn.Conv2d):
        torch.nn.init.orthogonal_(w.weight)

    # --- 5. 添加了对 students_custom 的检查 ---
    if isinstance(w, models.students_mnist_lenet.Model.ParallelFCModule) or \
       isinstance(w, models.students_mnist_lenet.Model.InitialParallelFCModule) or \
       isinstance(w, students_custom.Model.ParallelFCModule) or \
       isinstance(w, students_custom.Model.InitialParallelFCModule):
        torch.nn.init.orthogonal_(w.fc)
