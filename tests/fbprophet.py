import pyupbit
from prophet import Prophet
import pandas as pd
import pymysql
#
# con = pymysql.connect(
#     host='localhost',
#     user='root',
#     password='1111',
#     db='upbit',
#     charset='utf8',
#     # autocommit = True,  # 결과 DB 반영 (Insert or update)
#     # cursorclass=pymysql.cursors.DictCursor  # DB조회시 컬럼명을 동시에 보여줌
# )
#
# cur = con.cursor()
#
# sql = 'SELECT * FROM `market_chart`'
# cur.execute(sql)
# rows = cur.fetchall()
# print(rows)
# con.close()

# 시계열 데이터 분석해서 미래 예측해주는 라이브러리래요..
# 페북에서 만든거.

# MACD
def MACD(df, short=12, long=26, signal=9):
    # df['MACD'] = df['close'].ewm(span=short, min_periods=long-1, adjust=False).mean() - df['close'].ewm(span=long, min_periods=long-1, adjust=False).mean()
    df['MACD'] = df['close'].ewm(span=short, adjust=False).mean() - df['close'].ewm(span=long, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_OSC'] = df['MACD'] - df['MACD_Signal']
    return df
# end def

# RSI
def RSI(df, period=14):
    delta = df['close'].diff()
    gains, declines = delta.copy(), delta.copy()
    gains[gains < 0] = 0
    declines[declines > 0] = 0
    _gain = gains.ewm(com=(period-1), min_periods=period).mean()
    _loss = declines.abs().ewm(com=(period-1), min_periods=period).mean()
    RS = _gain / _loss
    return pd.Series(100 - (100/(1+RS)), name='RSI')
#end def

def calculate_rsi(data, window=14):
    close_delta = data['close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ema_up = up.ewm(com=window - 1, min_periods=window).mean()
    ema_down = down.ewm(com=window - 1, min_periods=window).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 업비트에서 BTC/KRW 데이터 가져오기
market = 'KRW-BTC'
# df = pyupbit.get_ohlcv(market, interval="minute10", count=1000)
# df = pyupbit.get_ohlcv(market, interval="day", count=1000)
df = pyupbit.get_ohlcv(market, interval="minute60", count=1000)
df = df.reset_index()
df = df.iloc[:-1]
# print(df)

# train = df.iloc[:-30]
# test = df.iloc[-30:]
df['rsi'] = calculate_rsi(df)
df = MACD(df)
df['rsi'] = df['rsi'].round(2)

# Prophet의 입력 데이터 형식에 맞게 컬럼 이름 변경
df = df.rename(columns={"index":"ds", "rsi":"y"})
df.reset_index(inplace=True)

train = df.iloc[14:]
print(train)

# Prophet 모델 객체 생성
model = Prophet(
    interval_width=0.95,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.3,
    yearly_seasonality=True,
)

# 모델 학습
model.fit(train)

# 예측
# future = model.make_future_dataframe(periods=5, freq='D')
future = model.make_future_dataframe(periods=24, freq='60min')

# 예측
forecast = model.predict(future)
forecast['yhat'] = forecast['yhat'].round(2)
# 데이터 시각화
# fig = model.plot(forecast)
# plt.show()

# 예측 결과 출력
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(24))
# print(forecast.tail(24))

sql = 'INSERT INTO `rsi_testdata`(market, candle_kst_date, close, origin_rsi, predict_rsi) VALUES(%s, %s, %s, %s, %s)' % (market, '2', '3', '4', '5')
print(sql)
