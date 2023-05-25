import yfinance as yf

# AAPL(Appl Inc.)의 주식 데이터를 가져오기
data = yf.download("AAPL", start="2021-01-01", end="2023-04-18")

import numpy as np
import pandas as pd

# Adj Close 컬럼을 기준으로 데이터프레임 정렬
data = data.sort_values('Date')

# 필요한 컬럼만 추출
data = data[['Adj Close']]

# 데이터 정규화
data = (data - data.mean()) / data.std()

import torch
from torch import nn

# LSTM 모델 생성
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# 하이퍼 파라미터 설정
input_size = 1
hidden_size = 16
num_layers = 2
output_size = 1

# 모델 생성
model = LSTM(input_size, hidden_size, num_layers, output_size)

# 입력 데이터와 출력 데이터 나누기
inputs = data[:-1].values.reshape(-1, 1, input_size)
outputs = data[1:].values.reshape(-1, 1, output_size)

# 훈련용 데이터와 검증용 데이터 나누기
train_size = int(len(inputs) * 0.8)
train_inputs, train_outputs = inputs[:train_size], outputs[:train_size]
val_inputs, val_outputs = inputs[train_size:], outputs[train_size:]

# 하이퍼 파라미터 설정
lr = 0.001
epochs = 1000

# 손실 함수와 최적화 함수 설정
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 학습 진행
for epoch in range(epochs):
    train_loss = 0.0

    # 훈련용 데이터 학습
    model.train()
    optimizer.zero_grad()
    train_pred = model(train_inputs)
    loss = criterion(train_pred, train_outputs)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()

    # 검증용 데이터로 검증
    with torch.no_grad():
        model.eval()
        val_pred = model(val_inputs)
        val_loss = criterion(val_pred, val_outputs)

    # 결과 출력
    if (epoch + 1) % 100 == 0:
        print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')

# 모델 테스트
model.eval()
test_inputs = inputs[-1].reshape(1, 1, input_size)
predictions = []

for _ in range(len(outputs)):
    with torch.no_grad():
        test_pred = model(test_inputs)
        predictions.append(test_pred.numpy().flatten()[0])
        test_inputs = test_pred.reshape(1, 1, input_size)

# 실제 데이터와 예측한 데이터 시각화
import matplotlib.pyplot as plt

plt.plot(outputs.reshape(-1), label='actual')
plt.plot(predictions, label='prediction')
plt.legend()
plt.show()
