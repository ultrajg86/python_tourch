import requests, json, datetime, os, hashlib, uuid, jwt
import pandas as pd
from urllib.parse import urlencode


class Upbit:
    Access_key = False
    Secret_key = False
    BASE_URL = 'https://api.upbit.com'

    URL_MARKET_ALL = BASE_URL + '/v1/market/all'
    URL_CANDLES_MINUTES = BASE_URL + '/v1/candles/minutes/'
    URL_CANDLES_DAYS = BASE_URL + '/v1/candles/days/'
    URL_CANDLES_WEEKS = BASE_URL + '/v1/candles/weeks/'
    URL_CANDLES_MONTHS = BASE_URL + '/v1/candles/months/'

    # va
    chart_count_limit = 200

    # def __init__(self, config=None):
        # if config == None:
        #     config = {
        #         'upbit': {
        #             'Access_key': '-',
        #             'Secret_key': '-'
        #         }
        #     }
        # else:
        #     config = config['exchange']['upbit']
        #     # end if
        # self.Access_key = config['Access_key']
        # self.Secret_key = config['Secret_key']

    # end __init__

    def get_market(self, isDetails='true', filter_market='ALL'):
        """
        마켓 코드 조회 (업비트에서 거래 가능한 마켓 목록 조회)
        :return:
        """
        market_info = []
        try:
            params = {'isDetails': isDetails}
            market_json = self._request_json(self.URL_MARKET_ALL, params)
            if filter_market == 'ALL':
                market_info = market_json
            else:
                for m in market_json:
                    fiat, coin = m['market'].split('-')
                    if fiat == filter_market:
                        market_info.append(m['market'])
            return market_info
        except Exception as x:
            print(x.__class__.__name__)
            return None

    def get_tick_size(self, price):
        """
        호가 조회
        :return:
        """
        if price >= 2000000:
            tick_size = round(price / 1000) * 1000
        elif price >= 1000000:
            tick_size = round(price / 500) * 500
        elif price >= 500000:
            tick_size = round(price / 100) * 100
        elif price >= 100000:
            tick_size = round(price / 50) * 50
        elif price >= 10000:
            tick_size = round(price / 10) * 10
        elif price >= 1000:
            tick_size = round(price / 5) * 5
        elif price >= 100:
            tick_size = round(price / 1) * 1
        elif price >= 10:
            tick_size = round(price / 0.1) * 0.1
        else:
            tick_size = round(price / 0.01) * 0.01
        return tick_size

    def get_ohlcv(self, ticker='KRW-BTC', interval="day", count=200):
        """
        캔들 조회
        :return:
        """
        ticker_data = []
        try:
            url = self._get_url_ohlcv(interval=interval)
            if url == None:
                raise Exception('Unknow Interval!')

            to_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            while count > 0:
                if count > self.chart_count_limit:
                    params = {'market': ticker, 'count': self.chart_count_limit, 'to': to_date}
                    count = count - self.chart_count_limit
                else:
                    params = {'market': ticker, 'count': count, 'to': to_date}
                    count = 0

                ticker_json = self._request_json(url, params)

                to_date = ticker_json[len(ticker_json) - 1]['candle_date_time_utc'].replace('T', ' ')
                # ticker_json.append(ticker_json)
                # ticker_data.append(ticker_json)
                ticker_data = ticker_data + ticker_json

            dt_list = [datetime.datetime.strptime(x['candle_date_time_kst'], "%Y-%m-%dT%H:%M:%S") for x in ticker_data]
            df = pd.DataFrame(ticker_data, columns=['opening_price', 'high_price', 'low_price', 'trade_price',
                                                    'candle_acc_trade_volume', 'candle_acc_trade_price'], index=dt_list)
            df = df.rename(
                columns={"opening_price": "open", "high_price": "high", "low_price": "low", "trade_price": "close",
                         "candle_acc_trade_volume": "volume", "candle_acc_trade_price": "trade_price"})
            # df = pd.DataFrame(ticker_data, columns=['candle_date_time_kst', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume'], index=dt_list)
            # df = df.rename(columns={"candle_date_time_kst":"ds","opening_price": "open", "high_price": "high", "low_price": "low", "trade_price": "y", "candle_acc_trade_volume": "volume"})

            return df.iloc[::-1]
        except Exception as x:
            # print(ticker, x, __class__, __name__)
            return None

    def _get_url_ohlcv(self, interval):
        if interval == "day":
            url = self.URL_CANDLES_DAYS
        elif interval == "min1" or interval == "1min":
            url = self.URL_CANDLES_MINUTES + '1'
        elif interval == "min3" or interval == "3min":
            url = self.URL_CANDLES_MINUTES + '3'
        elif interval == "min5" or interval == "5min":
            url = self.URL_CANDLES_MINUTES + '5'
        elif interval == "min10" or interval == "10min":
            url = self.URL_CANDLES_MINUTES + '10'
        elif interval == "min15" or interval == "15min":
            url = self.URL_CANDLES_MINUTES + '15'
        elif interval == "min30" or interval == "30min":
            url = self.URL_CANDLES_MINUTES + '30'
        elif interval == "min60" or interval == "60min":
            url = self.URL_CANDLES_MINUTES + '60'
        elif interval == "min240" or interval == "240min":
            url = self.URL_CANDLES_MINUTES + '240'
        elif interval == "min480" or interval == "480min":
            url = self.URL_CANDLES_MINUTES + '240'
        elif interval == "week" or interval == "weeks":
            url = self.URL_CANDLES_WEEKS
        elif interval == "month":
            url = self.URL_CANDLES_MONTHS
        else:
            url = self.URL_CANDLES_DAYS
        return url

    ##
    def getbalance(self):
        access_key = self.Access_key
        secret_key = self.Secret_key

        payload = {
            'access_key': access_key,
            'nonce': str(uuid.uuid4()),
        }

        jwt_token = jwt.encode(payload, secret_key, algorithm='HS256')
        authorize_token = 'Bearer {}'.format(jwt_token)
        headers = {"Authorization": authorize_token}

        res = requests.get(self.BASE_URL + "/v1/accounts", headers=headers)
        return res.json()
        # print(res.json())

    # end def getbalance():

    def buy_market(self, market, price, volume):
        return self.trade_market(market, price, volume, 'bid')

    # end def

    def sell_market(self, market, price, volume):
        return self.trade_market(market, price, volume, 'ask')

    # end def

    def trade_market(self, market, price, volume, side, ord_type='limit'):
        access_key = self.Access_key
        secret_key = self.Secret_key
        server_url = self.BASE_URL

        """
        주문 타입 (필수)
        - limit : 지정가 주문
        - price : 시장가 주문(매수)
        - market : 시장가 주문(매도)
        """
        query = {
            'market': market,
            'side': side,
            'volume': volume,
            'price': price,
            'ord_type': ord_type,
        }

        if ord_type == 'market' and side == 'ask':
            del query['price']
        elif ord_type == 'price' and side == 'bid':
            del query['volume']
        # end if

        query_string = urlencode(query).encode()

        m = hashlib.sha512()
        m.update(query_string)
        query_hash = m.hexdigest()

        payload = {
            'access_key': access_key,
            'nonce': str(uuid.uuid4()),
            'query_hash': query_hash,
            'query_hash_alg': 'SHA512',
        }

        jwt_token = jwt.encode(payload, secret_key, algorithm='HS256')
        authorize_token = 'Bearer {}'.format(jwt_token)
        headers = {"Authorization": authorize_token}

        res = requests.post(server_url + "/v1/orders", params=query, headers=headers)
        return res

    # end def

    def _request_json(self, url, params):
        '''
        headers = {'Content-Type': 'application/json; charset=utf-8'}
        cookies = {'session_id': 'sorryidontcare'}
        res = requests.get(URL, headers=headers, cookies=cookies)
        '''
        res = requests.get(url, params)

        try:
            return res.json()
        except Exception as x:
            return {}
