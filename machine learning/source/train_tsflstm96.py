import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import datetime
from optuna.trial import Trial



device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

# 加载训练数据和测试数据
train_data = pd.read_csv('/home/xykang/machine learning/data/train_data.csv')
test_data = pd.read_csv('/home/xykang/machine learning/data/test_data.csv')

# 提取日期特征
def extract_date_features(df):
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['day'] = df['dteday'].dt.day
    day_series = df.pop('day')
    df.insert(0, 'day', day_series)
    df.drop(['dteday'], axis=1, inplace=True)
    return df

train_data = extract_date_features(train_data)
test_data = extract_date_features(test_data)

# 处理缺失值
def handle_outliers(df):
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
            df[col] = np.where(df[col] == np.inf, mean_val, df[col])
            df[col] = np.where(df[col] == -np.inf, mean_val, df[col])
    return df

train_data = handle_outliers(train_data)
test_data = handle_outliers(test_data)

# 去掉不必要的列
train_data.drop(['instant'], axis=1, inplace=True)
train_data.drop(['registered'], axis=1, inplace=True)
train_data.drop(['casual'], axis=1, inplace=True)
test_data.drop(['instant'], axis=1, inplace=True)
test_data.drop(['registered'], axis=1, inplace=True)
test_data.drop(['casual'], axis=1, inplace=True)
# 对数据进行归一化
scaler = MinMaxScaler()
scaler_output = MinMaxScaler()

train_data_tensor = torch.tensor(train_data.values, dtype=torch.float32).to(device)
test_data_tensor = torch.tensor(test_data.values, dtype=torch.float32).to(device)

train_data_scaled = scaler.fit_transform(train_data_tensor.cpu().numpy())
train_data_scaled = torch.tensor(train_data_scaled, dtype=torch.float32).to(device)
test_data_scaled = scaler.transform(test_data_tensor.cpu().numpy())
test_data_scaled = torch.tensor(test_data_scaled, dtype=torch.float32).to(device)

train_last_column = train_data_tensor[:, -1].cpu().numpy().reshape(-1, 1)
test_last_column = test_data_tensor[:, -1].cpu().numpy().reshape(-1, 1)
train_last_column_scaled = scaler_output.fit_transform(train_last_column)
test_last_column_scaled = scaler_output.transform(test_last_column)

# 数据转化为时间序列输入输出对
def create_sequences(data, input_length, output_length):
    sequences = []
    targets = []
    for i in range(len(data) - input_length - output_length + 1):
        seq = data[i:i + input_length, :]
        target = data[i + input_length:i + input_length + output_length, -1]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

input_length = 96  # 输入序列长度
output_length = 96  # 输出序列长度

# 创建时间序列数据
X_train, y_train = create_sequences(train_data_scaled.cpu().numpy(), input_length=input_length, output_length=output_length)
X_test, y_test = create_sequences(test_data_scaled.cpu().numpy(), input_length=input_length, output_length=output_length)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
input_size = X_train.shape[2]
# 将训练数据分为训练集和验证集
train_size = int(0.8 * len(X_train))  # 80% 训练集
val_size = len(X_train) - train_size  # 20% 验证集
X_train, X_val = X_train[:train_size], X_train[train_size:]
y_train, y_val = y_train[:train_size], y_train[train_size:]

# 定义PyTorch Dataset类
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# 创建数据集
train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
test_dataset = TimeSeriesDataset(X_test, y_test)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义Transformer模型
class TransformerLSTMModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_transformer_layers, hidden_size, num_lstm_layers, output_length):
        super(TransformerLSTMModel, self).__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.output_length = output_length

        if input_size != d_model:
            self.fc_in = nn.Linear(input_size, d_model)
        else:
            self.fc_in = nn.Identity()

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer Encoder 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # LSTM 
        self.lstm = nn.LSTM(d_model, hidden_size, num_lstm_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_length)

    def forward(self, x):
        # : [batch_size, sequence_length, input_size]
        x = self.fc_in(x)  # [batch_size, sequence_length, d_model]
        x = self.pos_encoder(x)  
        x = self.transformer_encoder(x)  # [batch_size, sequence_length, d_model]

        # LSTM处理
        lstm_out, _ = self.lstm(x)  # [batch_size, sequence_length, hidden_size]
        output = self.fc_out(lstm_out[:, -1, :])  # 使用最后一个时间步的输出
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
# 定义超参数搜索空间
def define_model(trial: Trial, input_size: int, output_length: int):
    """
    定义模型和优化器。
    
    参数:
        trial (Trial): Optuna 的 Trial 对象。
        input_size (int): 输入特征数。
        output_length (int): 输出序列长度。
    """
    # 定义超参数范围
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    nhead = trial.suggest_categorical("nhead", [4, 8, 16])
    num_transformer_layers = trial.suggest_int("num_transformer_layers", 2, 6)
    hidden_size = trial.suggest_categorical("hidden_size", [32, 64, 128])
    num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 3)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # 初始化模型
    model = TransformerLSTMModel(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_transformer_layers=num_transformer_layers,
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        output_length=output_length
    ).to(device)

    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, optimizer
criterion=nn.MSELoss()

d_model = 128  
nhead = 8      
num_layers = 3
output_length = output_length
hidden_size=64
num_lstm_layers=2
# 初始化模型
model = TransformerLSTMModel(input_size, d_model, nhead, num_layers,hidden_size,
        num_lstm_layers, output_length).to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 保存模型的目录
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

# 多次实验训练和评估
num_experiments = 5
mse_scores = []
mae_scores = []
all_predictions = []
all_true_values = []

for experiment in range(num_experiments):
    # 重新初始化模型
    model.apply(lambda m: m.reset_parameters() if hasattr(m,'reset_parameters') else None)

    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 训练循环
    epochs = 200
    for epoch in tqdm(range(epochs), desc=f"训练实验 {experiment + 1}/{num_experiments} 输出长度 {output_length}"):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # 正向传播
            y_pred = model(X_batch).squeeze()

            loss = criterion(y_pred, y_batch)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    # 验证集评估
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            predictions.extend(y_pred.cpu().numpy())
            true_values.extend(y_batch.cpu().numpy())

    # 计算评估指标
    predictions = np.array(predictions).reshape(-1, output_length)
    true_values = np.array(true_values).reshape(-1, output_length)

    predictions_original = scaler_output.inverse_transform(predictions)
    true_values_original = scaler_output.inverse_transform(true_values)

    mse = np.mean((predictions_original - true_values_original) ** 2)
    mae = np.mean(np.abs(predictions_original - true_values_original))

    mse_scores.append(mse)
    mae_scores.append(mae)

    all_predictions.append(predictions_original)
    all_true_values.append(true_values_original)

    # 保存模型
    model_path = os.path.join(model_dir, f"transformerLSTM_output_{output_length}_mse_{mse:.4f}_mae_{mae:.4f}_experiment_{experiment + 1}_{current_time}.pth")
    torch.save(model.state_dict(), model_path)

    # 可视化预测结果
    image_dir = "saved_images"
    os.makedirs(image_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    x_true = [i for i in range(0,192)]
    x_pred = [i for i in range(96,192)]
    plt.plot(x_pred, predictions_original[404], label="Predictions (Original Scale)", color="blue")
    plt.plot(x_true, (list(true_values_original[404-96])+list(true_values_original[404])), label="True Values (Original Scale)", color="orange")
    plt.title(f"Predictions vs True Values for Output Length {output_length}")
    plt.legend()

    # 保存图像到文件夹
    image_filename = os.path.join(image_dir, f"prediction_plot_output_length_{output_length}_{current_time}.png")
    plt.savefig(image_filename)

    # 显示图像
    plt.show()

# 输出最终的评估结果
    results = {
    'mse_mean': np.mean(mse_scores),
    'mse_std': np.std(mse_scores),
    'mae_mean': np.mean(mae_scores),
    'mae_std': np.std(mae_scores),
    }

print(f"Output Length: {output_length} hours")
print(f"MSE Mean: {results['mse_mean']:.4f}, MSE Std: {results['mse_std']:.4f}")
print(f"MAE Mean: {results['mae_mean']:.4f}, MAE Std: {results['mae_std']:.4f}")