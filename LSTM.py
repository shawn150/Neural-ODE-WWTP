import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, max_error
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
import seaborn as sns

# 1. 加载训练和测试数据
train_data = pd.read_csv('aved_train_smooth.csv')
test_data = pd.read_csv('aved_test_smooth.csv')

# 提取所有特征数据
train_values = train_data.values
test_values = test_data.values

# 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_values)
test_scaled = scaler.transform(test_values)

# 2. 创建输入输出数据对，选择特定的输出特征（0, 2, 4）
output_feature_indices = [0, 1, 2, 3]  # 选择特定特征作为输出（例如第1, 第3, 第5个）

def create_sequences_with_target(data, n_steps, target_indices):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps, :])  # 输入为所有特征
        y.append(data[i + n_steps, target_indices])  # 选择指定的输出特征
    return np.array(X), np.array(y)

# 使用新的函数生成数据
n_steps = 1  # 选择时间步长
X_train, y_train = create_sequences_with_target(train_scaled, n_steps, output_feature_indices)
X_test, y_test = create_sequences_with_target(test_scaled, n_steps, output_feature_indices)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 3. 检查 GPU 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 4. 创建 Dataset 和 DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 256
train_dataset = TimeSeriesDataset(X_train_tensor.to(device), y_train_tensor.to(device))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TimeSeriesDataset(X_test_tensor.to(device), y_test_tensor.to(device))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 5. 创建模型 (LSTM)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # 输出尺寸为指定特征数

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 定义模型参数
input_size = X_train.shape[2]  # 输入特征数量
hidden_size = 50  # 隐藏层单元数
output_size = len(output_feature_indices)  # 输出特征数量为指定特征数
num_epochs = 100

# 6. 实例化模型
model = LSTMModel(input_size, hidden_size, output_size).to(device)

criterion = nn.MSELoss()  # 损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器

# 7. 训练模型
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0  # 记录总损失
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()  # 累加每个 batch 的损失

    print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.6f}')

# 8. 预测与评估模型
model.eval()
y_pred_scaled = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        y_pred = model(X_batch)
        y_pred_scaled.append(y_pred)

# 将预测结果拼接起来并转移回 CPU
y_pred_scaled = torch.cat(y_pred_scaled, dim=0).cpu().numpy()

# 逆标准化处理
y_pred_full = np.zeros((y_pred_scaled.shape[0], train_scaled.shape[1]))
y_pred_full[:, output_feature_indices] = y_pred_scaled  # 填充预测值
y_pred = scaler.inverse_transform(y_pred_full)[:, :len(output_feature_indices)]  # 逆标准化并提取输出特征

# 逆标准化真实值
y_test_full = np.zeros((y_test_tensor.shape[0], train_scaled.shape[1]))  # 创建一个全零的数组来容纳真实数据
y_test_full[:, output_feature_indices] = y_test_tensor.cpu().numpy()  # 填充真实值
y_test_true = scaler.inverse_transform(y_test_full)[:, :len(output_feature_indices)]  # 逆标准化并提取输出特征

# 保存预测值和真实值到CSV文件
aligned_results = pd.DataFrame({
    **{f'True_Dim{i + 1}': y_test_true[:, i] for i in range(len(output_feature_indices))},
    **{f'Prediction_Dim{i + 1}': y_pred[:, i] for i in range(len(output_feature_indices))}
})
aligned_results.to_csv('LSTM_lucerne.csv', index=False)

print(f'Aligned predictions and true values saved')

# 9. 添加可解释性分析
# 将模型切换到 train 模式，以便计算梯度
model.train()

# 定义 Integrated Gradients 方法
ig = IntegratedGradients(model)

# 选择部分测试样本进行解释性分析
X_sample = X_test_tensor[:batch_size].to(device)

# 初始化存储特征重要性的矩阵
feature_importance_matrix = np.zeros((input_size, output_size))

# 遍历每个输出特征，分别计算 Integrated Gradients
for output_idx in range(output_size):
    print(f'Calculating contributions for Output {output_idx + 1}')
    attributions, _ = ig.attribute(X_sample, target=output_idx, return_convergence_delta=True)

    # 计算当前输出特征的特征重要性
    attributions_np = attributions.detach().cpu().numpy()  # 转为 NumPy
    feature_importance_matrix[:, output_idx] = attributions_np.mean(axis=(0, 1))  # 对时间步和样本取平均

# 保存特征重要性到 CSV 文件
feature_importance_df = pd.DataFrame(feature_importance_matrix,
                                     columns=[f'Output_{i + 1}' for i in range(output_size)],
                                     index=[f'Input_{i + 1}' for i in range(input_size)])
feature_importance_df.to_csv('integrated_gradients_LSTM_aved.csv')

print('Feature importance matrix saved')

# 可视化特征对每个输出的贡献
plt.figure(figsize=(12, 8))
sns.heatmap(feature_importance_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=[f'Output {i + 1}' for i in range(output_size)],
            yticklabels=[f'Input {i + 1}' for i in range(input_size)])
plt.title('Feature Contribution to Each Output Feature')
plt.xlabel('Output Features')
plt.ylabel('Input Features')
plt.show()


import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, max_error
import numpy as np

# Load the dataset
data_path = 'LSTM_lucerne.csv'  # Update with your file path
data = pd.read_csv(data_path)

# Initialize dictionaries to store error metrics
r2_scores_corrected = {}
adjusted_r2_scores = {}
mae_scores = {}
mse_scores = {}
rmse_scores = {}
mape_scores = {}
max_errors = {}

# Number of predictors (Adjusted R²)
n_predictors = len(output_feature_indices)  # Update based on your dataset's dimensions

# Iterate over columns to calculate metrics
for i in range(1, n_predictors + 1):  # Adjust range for the number of dimensions
    predicted_col = f'Prediction_Dim{i}'
    true_col = f'True_Dim{i}'

    if predicted_col in data.columns and true_col in data.columns:
        # Drop rows with NaN values in the current columns
        valid_data = data.dropna(subset=[predicted_col, true_col])
        n = len(valid_data)  # Number of valid samples

        if n > 0:
            # Calculate R²
            r2 = r2_score(valid_data[predicted_col], valid_data[true_col])
            r2_scores_corrected[predicted_col] = round(r2, 3)

            # Calculate Adjusted R²
            adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - n_predictors - 1)
            adjusted_r2_scores[predicted_col] = round(adj_r2, 3)

            # Calculate Mean Absolute Error (MAE)
            mae = mean_absolute_error(valid_data[true_col], valid_data[predicted_col])
            mae_scores[predicted_col] = round(mae, 3)

            # Calculate Mean Squared Error (MSE)
            mse = mean_squared_error(valid_data[true_col], valid_data[predicted_col])
            mse_scores[predicted_col] = round(mse, 3)

            # Calculate Root Mean Squared Error (RMSE)
            rmse = np.sqrt(mse)
            rmse_scores[predicted_col] = round(rmse, 3)

            # Calculate Mean Absolute Percentage Error (MAPE)
            try:
                mape = np.mean(np.abs((valid_data[true_col] - valid_data[predicted_col]) / valid_data[true_col])) * 100
                mape_scores[predicted_col] = round(mape, 3)
            except ZeroDivisionError:
                mape_scores[predicted_col] = float('inf')

            # Calculate Max Error
            max_err = max_error(valid_data[true_col], valid_data[predicted_col])
            max_errors[predicted_col] = round(max_err, 3)

# Combine metrics into a DataFrame
combined_metrics_df = pd.DataFrame({
    "Variable": list(r2_scores_corrected.keys()),
    "R2 Value": list(r2_scores_corrected.values()),
    "Adjusted R2": list(adjusted_r2_scores.values()),
    "MAE": list(mae_scores.values()),
    "MSE": list(mse_scores.values()),
    "RMSE": list(rmse_scores.values()),
    "MAPE": list(mape_scores.values()),
    "Max Error": list(max_errors.values())
})

# Save metrics to a CSV file
output_path = 'combined_metrics_LSTM_uster_1h.csv'
combined_metrics_df.to_csv(output_path, index=False)

print(f"Metrics saved to {output_path}")