import pyupbit
import pandas as pd
from fbprophet import Prophet

def calculate_rsi(data, window=14):
    close_delta = data['close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ema_up = up.ewm(com=window - 1, min_periods=window).mean()
    ema_down = down.ewm(com=window - 1, min_periods=window).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def predict_rsi_future(data, window=14, periods=10):
    # RSI 계산

    data['RSI'] = calculate_rsi(data, window)
    data = data[14:]
    print(data)
    # Prophet 입력 형식에 맞게 데이터 전처리

    df = df.rename(columns={"index": "ds", "rsi": "y"})
    prophet_data = data[['index', 'RSI']].rename(columns={'index': 'ds', 'RSI': 'y'})

    # Prophet 모델 초기화 및 학습
    model = Prophet()
    model.fit(prophet_data)

    # 10일 후를 예측하는 데이터프레임 생성
    future = model.make_future_dataframe(periods=periods)

    # 예측 수행
    forecast = model.predict(future)

    # 마지막 10일 후의 예측값 반환
    predicted_value = forecast['yhat'].iloc[-1]
    return predicted_value


# 예측에 사용할 주식 가격 데이터를 불러옵니다. (예시 데이터 사용)
# stock_data = pd.read_csv('stock_data.csv')

stock_data = pyupbit.get_ohlcv("KRW-BTC", interval="minute10", count=1000)
# df = df.reset_index()


# RSI 값을 예측하고 10일 후의 값을 예측합니다.
predicted_rsi = predict_rsi_future(stock_data)

# 예측 결과를 출력합니다.
print('10일 후 예측된 RSI 값:', predicted_rsi)
