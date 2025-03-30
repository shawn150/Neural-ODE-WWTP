import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取CSV文件中的数据并转换为Float类型
data = np.genfromtxt("aved_train_smooth.csv", delimiter=",", dtype=np.float32)

# 生成掩码矩阵，标记缺失值
mask = ~np.isnan(data)
data = np.nan_to_num(data)  # 将缺失值填充为0

# 数据标准化：减去均值，除以标准差
data_mean = data.mean(axis=0)
data_std = data.std(axis=0)
data_normalized = (data - data_mean) / data_std

# 数据集划分
train_ratio = 1
train_size = int(train_ratio * data.shape[0])

train_y = data_normalized[:train_size]
train_mask = mask[:train_size]

train_y = torch.tensor(train_y).view(-1, 1, 12).to(device)  ########根据数据文件修改
train_mask = torch.tensor(train_mask).view(-1, 1, 12).to(device)  ########根据数据文件修改

# 时间数据
train_t = torch.linspace(0., (train_size - 1) / 720, train_size).to(device)  ########根据数据文件修改


# 定义Neural ODE模型
class neuralODE(nn.Module):
    def __init__(self):
        super(neuralODE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 50),  ########根据数据文件修改
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 12),  ########根据数据文件修改
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # 改为Xavier初始化
                nn.init.zeros_(m.bias)

    def forward(self, t, y):
        return self.net(y.float())  # 确保输入数据是float类型


# 训练参数
niters = 100  # 训练迭代次数
batch_time = 2  # 批次时间步长
batch_size = 256  # 批次样本数
lr = 1e-4  # 调低学习率

# 初始化模型和优化器
model = neuralODE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# 损失记录
loss_history = []

print("Starting training.")

for epoch in range(niters):
    model.train()
    total_loss = 0  # 用于累计所有批次的损失

    # 遍历每个批次
    for start_idx in range(0, train_size, batch_size):
        end_idx = min(start_idx + batch_size, train_size)
        batch_y0 = train_y[start_idx:end_idx]
        batch_t = train_t[:batch_time]
        batch_y = torch.stack([train_y[start_idx:end_idx] for _ in range(batch_time)], dim=0).to(device)
        batch_mask = torch.stack([train_mask[start_idx:end_idx] for _ in range(batch_time)], dim=0).to(device)

        # 求解 ODE
        pred_y = odeint(model, batch_y0, batch_t, method='rk4')

        # 计算损失
        batch_loss = torch.mean(torch.abs((pred_y - batch_y) * batch_mask))
        batch_loss.backward()  # 累积梯度

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新模型参数
        optimizer.step()
        optimizer.zero_grad()

        total_loss += batch_loss.item()  # 累加损失

    scheduler.step(total_loss)

    # 记录损失
    loss_history.append(total_loss / (train_size // batch_size))

    # 打印 GPU 利用率
    gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
    gpu_memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # MB
    print(f'Epoch: {epoch + 1}/{niters}, Loss: {total_loss / (train_size // batch_size):.6f}')
    print(f'GPU Memory Allocated: {gpu_memory_allocated:.2f} MB, GPU Memory Reserved: {gpu_memory_reserved:.2f} MB')

# 保存模型
torch.save(model.state_dict(), 'NODE_aved1.pth')
print("Model saved.")

# 可视化训练损失
plt.figure()
plt.plot(range(1, niters + 1), loss_history, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()


# 定义 Integrated Gradients 方法
def integrated_gradients(model, input_tensor, target_output_idx, baseline=None, steps=500):
    if baseline is None:
        baseline = torch.zeros_like(input_tensor).to(input_tensor.device)

    # 线性插值：生成 steps 个输入点
    interpolated_inputs = [
        baseline + (float(i) / steps) * (input_tensor - baseline)
        for i in range(steps + 1)
    ]
    interpolated_inputs = torch.stack(interpolated_inputs, dim=0)

    interpolated_inputs.requires_grad = True

    outputs = model(interpolated_inputs)
    target_outputs = outputs[:, 0, target_output_idx]

    gradients = []
    for i in range(steps + 1):
        grad = torch.autograd.grad(
            target_outputs[i], interpolated_inputs, retain_graph=True
        )[0][i]
        gradients.append(grad)
    gradients = torch.stack(gradients, dim=0)

    avg_gradients = gradients.mean(dim=0)
    integrated_gradients = (input_tensor - baseline) * avg_gradients

    return integrated_gradients


# 定义输入和基线
example_input = train_y[0, 0, :].unsqueeze(0).to(device)
baseline_input = torch.zeros_like(example_input).to(device)

# 初始化 self.net 部分的模型
net_model = model.net

# 对每个输出特征计算 IG
output_dim = 12
ig_results = []
for target_idx in range(output_dim):
    ig_result = integrated_gradients(
        model=net_model,
        input_tensor=example_input,
        target_output_idx=target_idx,
        baseline=baseline_input,
        steps=500
    )
    ig_results.append(ig_result.cpu().detach().numpy())

ig_results = np.concatenate(ig_results, axis=0)

# 可视化特征重要性
plt.figure(figsize=(12, 6))
plt.imshow(ig_results, aspect='auto', cmap='viridis')
plt.colorbar(label='Integrated Gradients')
plt.xlabel('Input Features')
plt.ylabel('Output Features')
plt.title('Feature Importances using Integrated Gradients')
plt.show()

import pandas as pd

columns = [f'Input Feature {i+1}' for i in range(ig_results.shape[1])]
rows = [f'Output Feature {j+1}' for j in range(ig_results.shape[0])]
ig_df = pd.DataFrame(ig_results, index=rows, columns=columns)

output_csv_path = 'integrated_gradients_node_aved.csv'
ig_df.to_csv(output_csv_path)
print(f'Feature importance saved to {output_csv_path}')
