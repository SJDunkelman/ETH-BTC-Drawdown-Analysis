import finnhub
from datetime import datetime
import time

API_KEY = 'br4je57rh5r8ufeothr0'  # DELETE
finnhub_client = finnhub.Client(api_key=API_KEY)

symbols = finnhub_client.crypto_symbols('BINANCE')

btc_ticker = 'BINANCE:BTCUSDT'
eth_ticker = 'BINANCE:ETHUSDT'
start_date = datetime(2016, 1, 1)
end_date = datetime(2016, 1, 1)

btc_data = finnhub_client.crypto_candles(btc_ticker, 'D',
                                         int(time.mktime(start_date.timetuple())),
                                         int(time.mktime(end_date.timetuple())))

eth_data = finnhub_client.crypto_candles(eth_ticker, 'D',
                                         int(time.mktime(start_date.timetuple())),
                                         int(time.mktime(end_date.timetuple())))
