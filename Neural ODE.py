import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt

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
train_t = torch.linspace(0., (train_size - 1) / 720, train_size).to(device)   ########根据数据文件修改

# 选择特征索引（例如，我们选择特征 0, 2, 4 等）
selected_features = [0, 1, 2, 3]  # 选择要计算损失的特征索引（你可以根据需要修改）
selected_features_mask = torch.zeros(12)  # 假设一共有12个特征
selected_features_mask[selected_features] = 1  # 选定特征的位置设为1，其余为0
selected_features_mask = selected_features_mask.to(device)

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
                nn.init.normal_(m.weight, mean=0, std=0.5)

    def forward(self, t, y):
        return self.net(y.float())  # 确保输入数据是float类型


# 训练参数
niters = 100  # 训练迭代次数
batch_time = 2  # 批次时间步长
batch_size = 256  # 批次样本数
lr = 1e-4  # 学习率

# 初始化模型和优化器
model = neuralODE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# 修改后的训练过程
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

        # 计算损失时只考虑选定的特征
        # batch_loss = torch.mean(torch.abs((pred_y - batch_y) * batch_mask))
        batch_loss = torch.mean(torch.abs((pred_y - batch_y) * batch_mask * selected_features_mask))  # 只计算选定特征的损失
        batch_loss.backward()  # 累积梯度
        total_loss += batch_loss.item()  # 累加损失

    # 更新模型参数
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step(total_loss)

    # 打印 GPU 利用率
    gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
    gpu_memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # MB
    print(f'Epoch: {epoch + 1}/{niters}, Loss: {total_loss / (train_size // batch_size):.6f}')
    print(f'GPU Memory Allocated: {gpu_memory_allocated:.2f} MB, GPU Memory Reserved: {gpu_memory_reserved:.2f} MB')

# 保存模型
torch.save(model.state_dict(), 'NODE_lucerne_30%missing.pth')
print("Model saved.")

# import numpy as np
# import torch
# import torch.nn as nn
# import pandas as pd
# from torchdiffeq import odeint
#
# # 设备设置
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# # 加载训练集数据（用于计算均值和标准差）
# train_data = np.genfromtxt("uster_train_smooth.csv", delimiter=",", dtype=np.float32)
# train_data = np.nan_to_num(train_data)  # 将训练数据中的缺失值填充为0
#
# # 计算训练集均值和标准差
# data_mean = train_data.mean(axis=0)
# data_std = train_data.std(axis=0)
#
# # 加载测试数据
# test_data = np.genfromtxt("uster_test_smooth.csv", delimiter=",", dtype=np.float32)
# test_mask = ~np.isnan(test_data)
# test_data = np.nan_to_num(test_data)  # 将缺失值填充为0
#
# # 数据标准化
# test_data_normalized = (test_data - data_mean) / data_std  # 使用训练集的均值和标准差
#
# # 转换为 PyTorch 张量
# test_y = torch.tensor(test_data_normalized).view(-1, 1, 13).to(device)  # 根据数据文件修改维度
#
# # 时间数据
# test_size = test_y.shape[0]
# time_step = 1 / 72  # 假设时间步长为 1/96
# future_steps = 1  # 预测未来 n 个时间步
#
#
# # 定义 NeuralODE 模型（与训练时相同）
# class neuralODE(nn.Module):
#     def __init__(self):
#         super(neuralODE, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(13, 50),  ########根据数据文件修改
#             nn.Tanh(),
#             nn.Linear(50, 50),
#             nn.Tanh(),
#             nn.Linear(50, 50),
#             nn.Tanh(),
#             nn.Linear(50, 13),  ########根据数据文件修改
#         )
#         for m in self.net.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean=0, std=0.5)
#
#     def forward(self, t, y):
#         return self.net(y.float())  # 确保输入数据是float类型
#
#
# # 加载训练好的模型
# model = neuralODE().to(device)
# model.load_state_dict(torch.load('NODE_lucerne_30%missing.pth'))
# model.eval()
# print("Model loaded.")
#
# # 滚动预测
# with torch.no_grad():
#     results = []
#     # 滚动预测每个时间步
#     for i in range(test_size - future_steps):
#         # 当前真实值作为初值
#         current_y = test_y[i]
#         # 时间区间 [0, n * time_step]
#         future_t = torch.linspace(0, time_step * future_steps, future_steps + 1).to(device)
#
#         # 使用模型预测未来 n 个时间步
#         pred_y = odeint(model, current_y, future_t, method='rk4')
#         # 记录真实值和最后一个预测值
#         last_pred = pred_y[-1].cpu().numpy() * data_std + data_mean  # 还原预测数据
#         true_value = test_y[i + future_steps].cpu().numpy() * data_std + data_mean  # 还原真实值
#         time_stamp = time_step * (i + future_steps)
#
#         # 保存时间戳、真实值和预测值
#         results.append({
#             "Time": time_stamp,
#             **{f"True_Dim{j + 1}": true_value[0, j] for j in range(true_value.shape[-1])},
#             **{f"Prediction_Dim{j + 1}": last_pred[0, j] for j in range(last_pred.shape[-1])}
#         })
#
# # 转换为 DataFrame 并保存
# df = pd.DataFrame(results)
# df.to_csv("NODE_lucerne_30%missing.csv", index=False)
# print("Predictions with truth saved.")
#
# ##模型评估
# import pandas as pd
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, max_error
# import numpy as np
#
# # Load the dataset
# data_path = 'NODE_lucerne_30%missing1.csv'  # Update with your file path
# data = pd.read_csv(data_path)
#
# # Initialize dictionaries to store error metrics
# r2_scores_corrected = {}
# adjusted_r2_scores = {}
# mae_scores = {}
# mse_scores = {}
# rmse_scores = {}
# mape_scores = {}
# max_errors = {}
#
# # Number of predictors (Adjusted R²)
# n_predictors = 12  # Update based on your dataset's dimensions
#
# # Iterate over columns to calculate metrics
# for i in range(1, n_predictors + 1):  # Adjust range for the number of dimensions
#     predicted_col = f'Prediction_Dim{i}'
#     true_col = f'True_Dim{i}'
#
#     if predicted_col in data.columns and true_col in data.columns:
#         # Drop rows with NaN values in the current columns
#         valid_data = data.dropna(subset=[predicted_col, true_col])
#         n = len(valid_data)  # Number of valid samples
#
#         if n > 0:
#             # Calculate R²
#             r2 = r2_score(valid_data[predicted_col], valid_data[true_col])
#             r2_scores_corrected[predicted_col] = round(r2, 3)
#
#             # Calculate Adjusted R²
#             adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - n_predictors - 1)
#             adjusted_r2_scores[predicted_col] = round(adj_r2, 3)
#
#             # Calculate Mean Absolute Error (MAE)
#             mae = mean_absolute_error(valid_data[true_col], valid_data[predicted_col])
#             mae_scores[predicted_col] = round(mae, 3)
#
#             # Calculate Mean Squared Error (MSE)
#             mse = mean_squared_error(valid_data[true_col], valid_data[predicted_col])
#             mse_scores[predicted_col] = round(mse, 3)
#
#             # Calculate Root Mean Squared Error (RMSE)
#             rmse = np.sqrt(mse)
#             rmse_scores[predicted_col] = round(rmse, 3)
#
#             # Calculate Mean Absolute Percentage Error (MAPE)
#             try:
#                 mape = np.mean(np.abs((valid_data[true_col] - valid_data[predicted_col]) / valid_data[true_col])) * 100
#                 mape_scores[predicted_col] = round(mape, 3)
#             except ZeroDivisionError:
#                 mape_scores[predicted_col] = float('inf')
#
#             # Calculate Max Error
#             max_err = max_error(valid_data[true_col], valid_data[predicted_col])
#             max_errors[predicted_col] = round(max_err, 3)
#
# # Combine metrics into a DataFrame
# combined_metrics_df = pd.DataFrame({
#     "Variable": list(r2_scores_corrected.keys()),
#     "R2 Value": list(r2_scores_corrected.values()),
#     "Adjusted R2": list(adjusted_r2_scores.values()),
#     "MAE": list(mae_scores.values()),
#     "MSE": list(mse_scores.values()),
#     "RMSE": list(rmse_scores.values()),
#     "MAPE": list(mape_scores.values()),
#     "Max Error": list(max_errors.values())
# })
#
# # Save metrics to a CSV file
# output_path = 'combined_metrics_NODE_aved_noisy1.csv'
# combined_metrics_df.to_csv(output_path, index=False)
#
# print(f"Metrics saved to {output_path}")

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