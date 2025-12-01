"""自定义教师神经网络模型"""  
import torch  
import torch.nn as nn  
from models import base  
from foundations import hparams  
from lottery.desc import LotteryDesc  
from pruning import sparse_global  
  
class Model(base.Model):  
    """自定义的全连接教师网络"""  
      
    def __init__(self, plan, initializer, act_fun, outputs=10):  
        super(Model, self).__init__()  
        self.act_fun = act_fun  
          
        layers = []  
        current_size = plan[0]  # 输入维度  
        for size in plan[1:]:  
            layers.append(nn.Linear(current_size, size))  
            current_size = size  
          
        self.fc_layers = nn.ModuleList(layers)  
        self.fc = nn.Linear(current_size, outputs)  
        self.criterion = nn.CrossEntropyLoss()  
          
        self.apply(initializer)  
      
    def forward(self, x):  
        x = x.view(x.size(0), -1)  
        for layer in self.fc_layers:  
            x = self.act_fun(layer(x))  
        return self.fc(x)  
      
    @property  
    def output_layer_names(self):  
        return ['fc.weight', 'fc.bias']  
      
    @property  
    def loss_criterion(self):  
        return self.criterion  
      
    @staticmethod  
    def is_valid_model_name(model_name):  
        return model_name.startswith('custom_teacher')  
      
    @staticmethod  
    def get_model_from_name(model_name, initializer, act_fun, outputs=None):  
        outputs = 1  # 强制使用单输出  
        plan = [2, 4,4]  
        return Model(plan, initializer, act_fun, outputs)
      
    @staticmethod  
    def default_hparams():  
        model_hparams = hparams.ModelHparams(  
            model_name='custom_teacher',  
            model_init='kaiming_normal',  
            batchnorm_init='uniform'  
        )  
        dataset_hparams = hparams.DatasetHparams(  
            dataset_name='mnist',  
            batch_size=128  
        )  
        training_hparams = hparams.TrainingHparams(  
            optimizer_name='relu',  
            lr=0.1,  
            training_steps='40ep',  
        )  
        pruning_hparams = sparse_global.PruningHparams(  
            pruning_strategy='sparse_global',  
            pruning_fraction=0.2,  
        )  
        extraction_hparams = hparams.ExtractionHparams(  
            gamma=None,  
            beta=None  
        )  
        return LotteryDesc(model_hparams, dataset_hparams, training_hparams,   
                          pruning_hparams, extraction_hparams)
    
