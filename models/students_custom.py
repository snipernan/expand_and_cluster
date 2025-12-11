"""自定义输入维度的学生网络模型"""  
import typing  
import numpy as np  
import torch  
import torch.nn as nn  
  
from foundations import hparams  
from lottery.desc import LotteryDesc  
from models import base  
from pruning import sparse_global  
from models.activation_functions import identity  
  
class Model(base.Model):  
    '''支持自定义输入维度的并行学生网络'''  
  
    class InitialParallelFCModule(nn.Module):  
        """第一层:支持自定义输入维度和选择性激活函数"""  
        def __init__(self, d_in, d_out, N, act_fun, fixed_indices=None):  
            super(Model.InitialParallelFCModule, self).__init__()  
            self.fc = nn.Parameter(torch.zeros(d_in, d_out, N))  
            self.b = nn.Parameter(torch.zeros(d_out, N))  
            self.act_fun = act_fun  
            self.fixed_indices = fixed_indices or []  
    
        def forward(self, x):  
            # 计算线性变换  
            linear_out = torch.einsum('bi,ihn->bhn', x, self.fc) + self.b.expand([x.shape[0]] + list(self.b.shape))  
            
            # 只对固定神经元应用激活函数  
            if self.fixed_indices:  
                activated_out = linear_out.clone()  
                for idx in self.fixed_indices:  
                    activated_out[:, idx, :] = self.act_fun(linear_out[:, idx, :])  
                return activated_out  
            else:  
                return self.act_fun(linear_out)  
    
    class ParallelFCModule(nn.Module):  
        """后续层:支持选择性激活函数"""  
        def __init__(self, d_in, d_out, N, act_fun, fixed_indices=None):  
            super(Model.ParallelFCModule, self).__init__()  
            self.fc = nn.Parameter(torch.zeros(d_in, d_out, N))  
            self.b = nn.Parameter(torch.zeros(d_out, N))  
            self.act_fun = act_fun  
            self.fixed_indices = fixed_indices or []  
    
        def forward(self, x):  
            # 计算线性变换  
            linear_out = torch.einsum('bin,ihn->bhn', x, self.fc) + self.b.expand([x.shape[0]] + list(self.b.shape))  
            
            # 只对固定神经元应用激活函数  
            if self.fixed_indices:  
                activated_out = linear_out.clone()  
                for idx in self.fixed_indices:  
                    activated_out[:, idx, :] = self.act_fun(linear_out[:, idx, :])  
                return activated_out  
            else:  
                return self.act_fun(linear_out) 
  
    def __init__(self, plan, d_in, initializer, act_fun, outputs=1, fixed_weights=None):  
        super(Model, self).__init__()  
        self.act_fun = act_fun  
        self.plan = plan  
        self.N = plan[0]  
        self.d_in = d_in  
        self.initializer = initializer  
        self.outputs = outputs  
        self.fixed_weights = fixed_weights  
        
        # 提取每层的固定神经元索引  
        self.fixed_indices_per_layer = {}  
        if fixed_weights:  
            for layer_idx in fixed_weights:  
                self.fixed_indices_per_layer[layer_idx] = list(fixed_weights[layer_idx].keys())  
    
        layers = []  
        current_size = d_in  
        for i, size in enumerate(self.plan[1:]):  
            fixed_indices = self.fixed_indices_per_layer.get(i, [])  
            if i == 0:  
                layers.append(self.InitialParallelFCModule(current_size, size, self.N, self.act_fun, fixed_indices))  
            else:  
                layers.append(self.ParallelFCModule(current_size, size, self.N, self.act_fun, fixed_indices))  
            current_size = size  
        
        # 输出层不应用选择性激活  
        layers.append(self.ParallelFCModule(current_size, outputs, self.N, identity()))  
        self.fc_layers = nn.ModuleList(layers)  
    
        self.criterion = Model.loss_fn  
        print(f"Model.__init__ 收到 fixed_weights: {fixed_weights}")  
        
        self.apply(self.initializer)  
        
        # 应用固定权重  
        if fixed_weights:  
            print("应用固定权重...")  
            self.apply_fixed_weights()  
        else:  
            print("没有固定权重需要应用")
        
        if fixed_weights:  
            self.apply_fixed_weights()
    
    def apply_fixed_weights(self): 
            """应用固定的神经元权重，并注册 Hook 使得梯度永远为 0""" 
            if not self.fixed_weights: 
                return 
                
            for layer_idx, layer_fixed_weights in self.fixed_weights.items(): 
                if layer_idx < len(self.fc_layers): 
                    layer = self.fc_layers[layer_idx] 
                    
                    # 获取当前层的设备（此时通常是 CPU）
                    device = layer.fc.device
                    
                    # 1. 准备掩码 (Mask)，初始全为 1
                    fc_mask = torch.ones_like(layer.fc, device=device)
                    b_mask = torch.ones_like(layer.b, device=device)
                    
                    for neuron_idx, (w, b) in layer_fixed_weights.items(): 
                        # 2. 设置固定神经元的权重和偏置
                        # 注意：这里也需要 .to(device)，尽管此时 device 可能是 CPU
                        w_tensor = torch.tensor(w, device=device).unsqueeze(-1).repeat(1, 1, self.N)
                        b_tensor = torch.tensor(b, device=device).repeat(self.N)
                        
                        layer.fc.data[:, neuron_idx, :] = w_tensor
                        layer.b.data[neuron_idx, :] = b_tensor
                        
                        # 3. 将固定位置的掩码设为 0
                        fc_mask[:, neuron_idx, :] = 0
                        b_mask[neuron_idx, :] = 0
                    
                    # 4. 定义 Hook 函数 (关键修改)
                    def get_mask_hook(mask_tensor):
                        def hook(grad):
                            # --- 修复核心: 动态对齐设备 ---
                            # 如果 grad 在 GPU 而 mask 在 CPU，这里会自动将 mask 移到 GPU
                            if mask_tensor.device != grad.device:
                                return grad * mask_tensor.to(grad.device)
                            return grad * mask_tensor
                        return hook

                    # 5. 注册 Hook
                    layer.fc.register_hook(get_mask_hook(fc_mask))
                    layer.b.register_hook(get_mask_hook(b_mask))
                    
            print("已应用固定权重并注册自适应设备梯度的 Hook")
  
    def forward(self, x):  
        x = x.view(x.size(0), -1)  
        for layer in self.fc_layers:  
            x = layer(x)  
        return x  
  
    @property  
    def output_layer_names(self):  
        out_name = list(self.named_modules())[-1][0]  
        return [f'{out_name}.fc', f'{out_name}.b']  
  
    @staticmethod  
    def is_valid_model_name(model_name):  
        return (model_name.startswith('students_custom(') and  
                model_name.find(")") != -1)  
  
    @staticmethod  
    def get_model_from_name(model_name, initializer, act_fun, outputs=None, d_in=None, fixed_weights=None):  
        outputs = outputs or 1  
        if not Model.is_valid_model_name(model_name):  
            raise ValueError('Invalid model name: {}'.format(model_name))  
            
        # 解析: students_custom(N)_d_in_W1_W2...    
        N = int(model_name[model_name.find("(")+1:model_name.find(")")])    
        parts = model_name.split('_')[2:]  # ['d', 'in', 'W1', 'W2', ...]    
            
        # 第一个数字是 d_in    
        if d_in is None:    
            d_in = int(parts[0])    
            
        # 其余是隐藏层维度    
        plan = [N]    
        plan.extend([int(n) for n in parts[1:]])    
            
        return Model(plan, d_in, initializer, act_fun, outputs, fixed_weights) 
    
    @staticmethod  
    def loss_fn(y_hat, y):  
        overall_loss = Model.individual_losses(y_hat, y).sum()  
        return overall_loss
    
    @staticmethod  
    def individual_losses(y_hat, y):  
        # y_hat: [batch_size, 1, N] - N 个学生网络的输出  
        # y: [batch_size, 1] 或 [batch_size] - 教师标签  
        
        if y.dim() == 1:  
            y = y.unsqueeze(1)  # 确保是 [batch_size, 1]  
        
        # 扩展 y 以匹配学生网络数量  
        y_repeats = y.unsqueeze(-1).repeat(1, 1, y_hat.shape[-1])  # [batch_size, 1, N]  
        
        # 计算每个学生的 MSE  
        return (y_hat - y_repeats).square().mean(dim=(0, 1)).squeeze()
  
    @property  
    def loss_criterion(self):  
        return self.criterion  
  
    @property  
    def prunable_layer_names(self) -> typing.List[str]:  
        return [name + '.fc' for name, module in self.named_modules() if  
                isinstance(module, self.InitialParallelFCModule) or  
                isinstance(module, self.ParallelFCModule)]  
  
    @staticmethod  
    def default_hparams():  
        model_hparams = hparams.ModelHparams(  
            model_name='students_custom(20)_2_300_100',  
            model_init='kaiming_normal',  
            batchnorm_init='uniform'  
        )  
        dataset_hparams = hparams.DatasetHparams(  
            dataset_name='teacher',  
            batch_size=512  
        )  
        training_hparams = hparams.TrainingHparams(  
            optimizer_name='adam',  
            lr=0.001,  
            training_steps='10000ep',  
        )  
        pruning_hparams = sparse_global.PruningHparams(  
            pruning_strategy='sparse_global',  
            pruning_fraction=0.2,  
        )  
        extraction_hparams = hparams.ExtractionHparams(  
            gamma=0.5,  
            beta=6,  
        )  
        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams, extraction_hparams)