import torch
from models import registry
from foundations import hparams
import numpy as np
import os

# ==============================================================================
# 配置部分
# ==============================================================================
# 请确保路径正确
TEACHER_CKPT = '/home/alvin/expand-and-cluster/data/sims/train_custom_teacher_deep/seed_0/main/model_ep0_it0.pth'
STUDENT_CKPT = '/home/alvin/expand-and-cluster/data/sims/ec_4bdc938a0f/seed_-1/main/model_ep6000_it0.pth'

# 你希望查看第几个学生的权重？(通常看第0个即可)
STUDENT_INDEX = 0 

# 设置打印精度
np.set_printoptions(precision=7, suppress=True)

# ==============================================================================
# 核心工具函数：格式化打印
# ==============================================================================
def print_styled_model(model_label, layers_data):
    """
    model_label: 模型名称 (如 'Original', 'Splitted')
    layers_data: 一个列表，包含每层的 (Weights, Bias) 元组。
                 要求 Weights 的形状统一为 [Neurons, Inputs] (即 [行, 列])
    """
    print("\n" + "="*60)
    print(f"重建模型 {model_label} 的权重")
    print("="*60 + "\n")

    num_layers = len(layers_data)

    for i, (w, b) in enumerate(layers_data):
        layer_num = i + 1
        num_neurons, num_inputs = w.shape
        
        # 判断是否为最后一层
        is_output_layer = (i == num_layers - 1)

        if is_output_layer:
            print(f"--- 第 {layer_num} 层 (输出层) ---")
            print(f"权重 (来自上一层的 {num_inputs} 个神经元，连接到唯一的输出):")
            # 输出层通常是 1 个神经元，flatten 后打印
            w_flat = w.flatten()
            # 拼接权重字符串
            w_str = "[" + "  ".join(f"{val: .7f}" for val in w_flat) + "]"
            print(w_str)
            print(f"\n偏置 (本层的 {num_neurons} 个输出):")
            print(f"   {b[0]: .7f}")
            print("-" * 60)
        
        else:
            print(f"--- 第 {layer_num} 层 ({num_neurons} 个神经元) ---")
            
            # 动态生成表头: [Input 1] [Input 2] ...
            header_inputs = " ".join([f"[Input {k+1} 权重]" for k in range(num_inputs)])
            print(f"神经元: {header_inputs}   [偏置]")
            
            # 动态生成分隔线
            print("-" * (12 * num_inputs + 30)) 

            # 遍历每个神经元 (行)
            for n_idx in range(num_neurons):
                # 提取权重和偏置
                weights = w[n_idx]
                bias = b[n_idx]
                
                # 格式化权重数值
                w_str = "  ".join(f"{val: .7f}" for val in weights)
                print(f"神经元 {n_idx+1:02d}:    {w_str}    {bias: .7f}")
            
            print("-" * 60 + "\n")


# ==============================================================================
# 1. 处理 Teacher (Original)
# ==============================================================================
teacher_data = []

# 定义 Hparams
teacher_hparams = hparams.ModelHparams(
    model_name='custom_teacher',
    model_init='kaiming_normal',
    batchnorm_init='uniform',
    act_fun='relu'
)

try:
    teacher_model = registry.get(teacher_hparams, outputs=1, d_in=2)
    if os.path.exists(TEACHER_CKPT):
        teacher_model.load_state_dict(torch.load(TEACHER_CKPT, map_location='cpu'))
    
    # 提取权重
    if hasattr(teacher_model, 'fc_layers'):
        # 1. 隐藏层
        for layer in teacher_model.fc_layers:
            # PyTorch Linear 默认形状 [Out, In]，直接使用
            w = layer.weight.detach().cpu().numpy()
            b = layer.bias.detach().cpu().numpy()
            teacher_data.append((w, b))
        
        # 2. 输出层 (假设叫 fc)
        if hasattr(teacher_model, 'fc'):
            w = teacher_model.fc.weight.detach().cpu().numpy()
            b = teacher_model.fc.bias.detach().cpu().numpy()
            teacher_data.append((w, b))

    # 执行打印
    if teacher_data:
        print_styled_model("Original", teacher_data)
    else:
        print("❌ 无法提取教师权重 (结构不匹配)")

except Exception as e:
    print(f"❌ 加载教师模型出错: {e}")


# ==============================================================================
# 2. 处理 Student (Splitted)
# ==============================================================================
student_data = []

student_hparams = hparams.ModelHparams( 
    model_name='students_custom(20)_2_12_12', # 你的新模型名称
    model_init='kaiming_normal', 
    batchnorm_init='uniform', 
    act_fun='relu' 
) 

try:
    students_model = registry.get(student_hparams, outputs=1, d_in=2) 
    if os.path.exists(STUDENT_CKPT):
        students_model.load_state_dict(torch.load(STUDENT_CKPT, map_location='cpu'))

    if hasattr(students_model, 'fc_layers'):
        for layer in students_model.fc_layers:
            # 学生模型权重形状通常是 [Input, Neurons, Students]
            # 我们需要取出指定学生，并转置为 [Neurons, Input] 以便逐行打印神经元
            
            # 1. 取出特定学生
            w_raw = layer.fc.detach().cpu().numpy()[:, :, STUDENT_INDEX] # shape: [In, Out]
            b_raw = layer.b.detach().cpu().numpy()[:, STUDENT_INDEX]     # shape: [Out]
            
            # 2. 转置权重矩阵：从 [In, Out] -> [Out, In] (即 [Neurons, Inputs])
            w_transposed = w_raw.T 
            
            student_data.append((w_transposed, b_raw))

    # 执行打印
    if student_data:
        print_styled_model("Splitted", student_data)
    else:
        print("❌ 无法提取学生权重 (结构不匹配)")

except Exception as e:
    print(f"❌ 加载学生模型出错: {e}")


# ==============================================================================
# 3. 输出对比 (教师 vs 学生)  [新增部分]
# ==============================================================================
print("\n\n" + "="*80)
print(f"PART 3: 前向传播对比 (Teacher vs Student #{STUDENT_INDEX})")
print("="*80)

# 生成测试数据 (5个样本, 2维输入)
torch.manual_seed(42)
test_input = torch.randn(5, 2)

# 切换到评估模式
if 'teacher_model' in locals():
    teacher_model.eval()
if 'students_model' in locals():
    students_model.eval()

# 检查模型是否都加载成功
if 'teacher_model' in locals() and 'students_model' in locals():
    with torch.no_grad():
        # 教师输出
        # [Batch, 1] -> flatten -> [Batch]
        t_out = teacher_model(test_input).detach().cpu().numpy().flatten()

        # 学生输出 
        # [Batch, 1, Students] -> 取出指定学生 -> flatten -> [Batch]
        s_out_all = students_model(test_input).detach().cpu().numpy()
        s_out = s_out_all[:, 0, STUDENT_INDEX].flatten()

    print(f"\n输入数据 (5个样本):\n{test_input.numpy()}\n")

    print(f"{'样本':<5} | {'教师输出':<15} | {'学生输出':<15} | {'差异 (Abs Diff)':<15}")
    print("-" * 60)

    for i in range(len(t_out)):
        diff = abs(t_out[i] - s_out[i])
        print(f"{i:<5} | {t_out[i]:<15.7f} | {s_out[i]:<15.7f} | {diff:<15.7f}")

    # 计算平均差异
    avg_diff = np.mean(np.abs(t_out - s_out))
    print("-" * 60)
    print(f"平均绝对误差 (MAE): {avg_diff:.7f}")

else:
    print("❌ 无法进行对比：教师模型或学生模型未能成功加载。")