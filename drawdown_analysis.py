import pandas as pd
from datetime import datetime, timedelta
from data import eth_price, btc_price
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


eth_price.dropna(inplace=True)
eth_price.reset_index(inplace=True)


THREE_MONTHS = 91
SIX_MONTHS = 182
NINE_MONTHS = 274
TWELVE_MONTHS = 365

WINDOW_DAYS = {'3_mth': THREE_MONTHS,
               '6_mth': SIX_MONTHS,
               '9_mth': NINE_MONTHS,
               '12_mth': TWELVE_MONTHS}


DRAWDOWN_PERCENTAGES = {'ten': 0.9,
                        'twenty': 0.8,
                        'thirty': 0.7,
                        'fourty': 0.6}


CRYPTOCURRENCIES = {'btc': btc_price,
                    'eth': eth_price}


def return_pct(t0: float, t1: float) -> float:
    return ((t1 - t0) / t0)


# RECENT_HIGH_PRICE = 'high'
BUY_IN_PRICE = 'open'

eth_price = eth_price.iloc[1:]


for curr_name, curr_data in tqdm(CRYPTOCURRENCIES.items()):
    correction_win_rate = pd.DataFrame(columns=['date', '3_mth', '6_mth', '9_mth', '12_mth'])
    correction_pcts = []
    dates = list(curr_data['date'])
    for correction_name, drawdown_pct in DRAWDOWN_PERCENTAGES.items():
        correction_market = []
        recent_high = curr_data.iloc[0]['open']
        for day in curr_data.itertuples():
            if day.low < (recent_high * drawdown_pct):
                correction_market.append(1)
                forward_percentages = {'date': day.date}
                for range_name, window in WINDOW_DAYS.items():
                    forward_date = datetime.combine(day.date, datetime.min.time()) + timedelta(days=window)
                    forward_date = forward_date.date()
                    if forward_date in dates:
                        forward_price = curr_data[BUY_IN_PRICE].loc[curr_data.date == forward_date].iloc[0]
                        forward_percentages[range_name] = return_pct(day.low, forward_price)
                correction_pcts.append(forward_percentages)
            else:
                correction_market.append(0)
                if day.high > recent_high:
                    recent_high = day.high

        corrections_df = pd.DataFrame(correction_pcts)
        corrections_df.dropna(inplace=True, thresh=4)
        corrections_df.to_csv(f'output/{curr_name}_{correction_name}_correction.csv')

        curr_data[f'correction_{correction_name}'] = correction_market
        days_in_downturn = len(curr_data[f'correction_{correction_name}'].loc[curr_data[f'correction_{correction_name}']==1])
        total_days = len(curr_data)
        print(f'For {curr_name} {days_in_downturn} out of {total_days} days were {correction_name}% lower than their recent peak')
        curr_data.to_csv(f'output/{curr_name}_with_corrections.csv')

        plt.style.use("seaborn-bright")
        plt.figure(figsize=(10, 5))
        plt.xlabel("Date")
        plt.ylabel(f"log(USD / {curr_name.upper()})")
        plt.title(f"{curr_name.upper()} with {correction_name.title()} % correction highlighted")
        plt.scatter(curr_data.date, curr_data.low, s=0.1, c=curr_data[f'correction_{correction_name}'], cmap='rainbow')
        plt.yscale('log')
        plt.savefig(f'graphs/{curr_name}_{correction_name}.png')
        plt.show()

    df = pd.DataFrame()
    for window_name, w_range in WINDOW_DAYS.items():
        window_avgs = []
        for correction_name, drawdown_pct in DRAWDOWN_PERCENTAGES.items():
            csv_i = pd.read_csv(f'output/{curr_name}_{correction_name}_correction.csv')
            window_avgs.append(csv_i[window_name].mean())
        df[window_name] = window_avgs

    index = pd.Index(['10%', '20%', '30%', '40%'])
    df = df.set_index(index)
    df.to_csv(f'output/output_{curr_name}.csv')

total_eth = return_pct(eth_price.iloc[0].open, eth_price.iloc[-1].close)
print(f'Total ETH return from {str(eth_price.iloc[0].date)} to {str(eth_price.iloc[-1].date)}  period: {total_eth}')

total_btc = return_pct(btc_price.iloc[0].open, btc_price.iloc[-1].close)
print(f'Total BTC return from {str(btc_price.iloc[0].date)} to {str(btc_price.iloc[-1].date)} period: {total_btc}')

# btc_price['correction'] = correction_market
# plt.scatter(btc_price.date, btc_price.open, s=0.1, c=btc_price.correction, cmap='rainbow')
# plt.savefig('fourty_percent_correct.png')
# plt.show()
