import pandas as pd
from datetime import datetime, timedelta
from data import eth_price, btc_price
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

THREE_MONTHS = 91
SIX_MONTHS = 182
NINE_MONTHS = 274
TWELVE_MONTHS = 365

WINDOW_DAYS = {'3_mth': THREE_MONTHS,
               '6_mth': SIX_MONTHS,
               '9_mth': NINE_MONTHS,
               '12_mth': TWELVE_MONTHS}


EMA_RANGES = {'9_ema': 9,
              '20_ema': 20}


CRYPTOCURRENCIES = {'btc': btc_price,
                    'eth': eth_price}

lower_ema_idx = list(ema_ranges.values()).index(min(ema_ranges.values()))
lower_ema_name = list(ema_ranges.keys())[lower_ema_idx]
higher_ema_name = [a for a in ema_ranges.keys() if a != lower_ema_name][0]
for curr_name, curr_data in tqdm(cryptocurrency_data.items()):
    dates = list(curr_data['date'])
    for ema_name, ema_range in ema_ranges.items():
        curr_data[ema_name] = curr_data['open'].ewm(span=ema_range, min_periods=ema_range).mean()
    curr_data['ema_correction'] = np.where(curr_data[lower_ema_name] < curr_data[higher_ema_name], 1, 0)

    curr_data.dropna(inplace=True, thresh=len(window_days))
    curr_data.to_csv(f'{output_dir}/{curr_name}_{lower_ema_name}_{higher_ema_name}_correction.csv')

    correction_pcts = []
    for day in curr_data.itertuples():
        forward_percentages = {'date': day.date}
        if day.ema_correction:
            for range_name, window in window_days.items():
                forward_date = datetime.combine(day.date, datetime.min.time()) + timedelta(days=window)
                forward_date = forward_date.date()
                if forward_date in dates:
                    forward_price = curr_data[buy_in_column].loc[curr_data.date == forward_date].iloc[0]
                    forward_percentages[range_name] = return_pct(day.low, forward_price)
            correction_pcts.append(forward_percentages)

    corrections_df = pd.DataFrame(correction_pcts)
    corrections_df.dropna(inplace=True, thresh=len(window_days))
    corrections_df.to_csv(f'{output_dir}/{curr_name}_ema_correction.csv')

    days_in_downturn = len(
        curr_data[f'ema_correction'].loc[curr_data[f'ema_correction'] == 1])
    total_days = len(curr_data)
    print(
        f'For {curr_name} {days_in_downturn} out of {total_days} days where the shorter ema crossed the longer')
    curr_data.to_csv(f'{output_dir}/{curr_name}_with_corrections.csv')

    plot_corrections(currency_name=curr_name,
                          correction_name='ema crossover',
                          currency_data=curr_data)

    # Create returns matrix
    df = pd.DataFrame()
    for window_name, w_range in window_days.items():
        csv_i = pd.read_csv(f'{output_dir}/{curr_name}_ema_correction.csv')
        df[window_name] = csv_i[window_name].mean()

    index = pd.Index([f"{lower_ema_name} below {higher_ema_name}"])
    returns_matrix = returns_matrix.set_index(index)
    returns_matrix.to_csv(f'{output_dir}/output_{curr_name}.csv')

    #     corrections_df = pd.DataFrame(correction_pcts)
    #     corrections_df.dropna(inplace=True, thresh=4)
    #     corrections_df.to_csv(f'output/{curr_name}_{correction_name}_correction.csv')
    # 
    #     curr_data[f'correction_{correction_name}'] = correction_market
    #     days_in_downturn = len(curr_data[f'correction_{correction_name}'].loc[curr_data[f'correction_{correction_name}']==1])
    #     total_days = len(curr_data)
    #     print(f'For {curr_name} {days_in_downturn} out of {total_days} days were {correction_name}% lower than their recent peak')
    #     curr_data.to_csv(f'output/{curr_name}_with_corrections.csv')
    # 
    #     plt.style.use("seaborn-bright")
    #     plt.figure(figsize=(10, 5))
    #     plt.xlabel("Date")
    #     plt.ylabel(f"log(USD / {curr_name.upper()})")
    #     plt.title(f"{curr_name.upper()} with {correction_name.title()} % correction highlighted")
    #     plt.scatter(curr_data.date, curr_data.low, s=0.1, c=curr_data[f'correction_{correction_name}'], cmap='rainbow')
    #     plt.yscale('log')
    #     plt.savefig(f'graphs/{curr_name}_{correction_name}.png')
    #     plt.show()
    # 
    # df = pd.DataFrame()
    # for window_name, w_range in WINDOW_DAYS.items():
    #     window_avgs = []
    #     for correction_name, drawdown_pct in DRAWDOWN_PERCENTAGES.items():
    #         csv_i = pd.read_csv(f'output/{curr_name}_{correction_name}_correction.csv')
    #         window_avgs.append(csv_i[window_name].mean())
    #     df[window_name] = window_avgs
    # 
    # index = pd.Index(['10%', '20%', '30%', '40%'])
    # df = df.set_index(index)
    # df.to_csv(f'output/output_{curr_name}.csv')