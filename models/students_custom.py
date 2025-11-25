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
        """第一层:支持自定义输入维度"""  
        def __init__(self, d_in, d_out, N, act_fun):  
            super(Model.InitialParallelFCModule, self).__init__()  
            self.fc = nn.Parameter(torch.zeros(d_in, d_out, N))  
            self.b = nn.Parameter(torch.zeros(d_out, N))  
            self.act_fun = act_fun  
  
        def forward(self, x):  
            return self.act_fun(torch.einsum('bi,ihn->bhn', x, self.fc) +  
                   self.b.expand([x.shape[0]] + list(self.b.shape)))  
  
    class ParallelFCModule(nn.Module):  
        """后续层"""  
        def __init__(self, d_in, d_out, N, act_fun):  
            super(Model.ParallelFCModule, self).__init__()  
            self.fc = nn.Parameter(torch.zeros(d_in, d_out, N))  
            self.b = nn.Parameter(torch.zeros(d_out, N))  
            self.act_fun = act_fun  
  
        def forward(self, x):  
            return self.act_fun(torch.einsum('bin,ihn->bhn', x, self.fc) +  
                   self.b.expand([x.shape[0]] + list(self.b.shape)))  
  
    def __init__(self, plan, d_in, initializer, act_fun, outputs=1):  
        super(Model, self).__init__()  
        self.act_fun = act_fun  
        self.plan = plan  
        self.N = plan[0]  
        self.d_in = d_in  
        self.initializer = initializer  
        self.outputs = outputs  
  
        layers = []  
        current_size = d_in  # 使用自定义输入维度  
        for i, size in enumerate(self.plan[1:]):  
            if i == 0:  
                layers.append(self.InitialParallelFCModule(current_size, size, self.N, self.act_fun))  
            else:  
                layers.append(self.ParallelFCModule(current_size, size, self.N, self.act_fun))  
            current_size = size  
        layers.append(self.ParallelFCModule(current_size, outputs, self.N, identity()))  
        self.fc_layers = nn.ModuleList(layers)  
  
        self.criterion = self.loss_fn  
        self.apply(self.initializer)  
  
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
    def get_model_from_name(model_name, initializer, act_fun, outputs=None, d_in=None):  
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
          
        return Model(plan, d_in, initializer, act_fun, outputs)  
  
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