import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pyupbit

# 업비트 API로 비트코인 가격 데이터 가져오기
# 업비트 API 사용을 위한 API Key, Secret Key 발급 필요

# 차트 데이터 가져오기
data = pyupbit.get_ohlcv("KRW-BTC", interval="minute1")

# 데이터 전처리
df = pd.DataFrame(data)
# df['date'] = pd.to_datetime(df['candle_date_time_kst'], format='%Y-%m-%dT%H:%M:%S')
df['date'] = pd.to_datetime(df.index, format='%Y-%m-%dT%H:%M:%S')
df = df.set_index('date')
# df = df[['opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']]
df = df[['open', 'high', 'low', 'close', 'volume']]

# print(df)

# 학습 데이터와 테스트 데이터 분리
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# 데이터 스케일링
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

# LSTM 모델 구현
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        # print(input_dim, hidden_dim, num_layers, output_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)  # hidden state 초기화
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)  # cell state 초기화

        out, (h, c) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 하이퍼파라미터 설정
input_size = 5
hidden_size = 64
num_layers = 2
output_size = 1
learning_rate = 0.001
num_epochs = 100

# 모델 초기화 및 CUDA 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# print(train_scaled)

# 모델 학습
for epoch in range(num_epochs):
    model.train()
    print(train_scaled)
    train_inputs = torch.from_numpy(train_scaled[:-1]).float().to(device)
    # print(train_inputs)
    train_labels = torch.from_numpy(train_scaled[1:, 3]).float().to(device)
    optimizer.zero_grad()
    outputs = model(train_inputs)
    loss = criterion(outputs, train_labels.unsqueeze(1))
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 예측
model.eval()
test_inputs = torch.from_numpy(test_scaled[:-1]).float().to(device)
with torch.no_grad():
    test_outputs = model(test_inputs)
test_predictions = test_outputs.cpu().numpy()

# 예측 결과 시각화
test_df['predicted_price'] = scaler.inverse_transform(test_predictions)[:, 0]
plt.figure(figsize=(16, 6))
plt.plot(train_df['trade_price'], label='Train')
plt.plot(test_df['trade_price'], label='Test')
plt.plot(test_df['predicted_price'], label='Predicted')
plt.legend(loc='best')
plt.show()
