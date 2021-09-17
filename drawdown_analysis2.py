import pandas as pd
from datetime import datetime, timedelta
from data import eth_price, btc_price
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path


def return_pct(t0: float, t1: float) -> float:
    return ((t1 - t0) / t0)


class DrawdownAnalysis:
    def __init__(self,
                 analysis_dir_name: str,
                 window_days: dict,
                 buy_in_column: str,
                 cryptocurrency_data: dict):
        self.analysis_dir = Path(f'{os.getcwd()}/{analysis_dir_name}')
        self.output_dir = self.analysis_dir / 'output'
        self.graphs_dir = self.analysis_dir / 'graphs'
        if not os.path.isdir(self.analysis_dir):
            os.mkdir(self.analysis_dir)
            os.mkdir(self.output_dir)
            os.mkdir(self.graphs_dir)
            os.mkdir(self.analysis_dir / 'tables')

        self.buy_in_column = buy_in_column
        self.window_days = window_days
        self.cryptocurrency_data = cryptocurrency_data

        self.calculate_drawdowns()

    def calculate_drawdowns(self):
        raise NotImplementedError('Need drawdown calculation func.')

    def plot_corrections(self, currency_name: str,
                         correction_name: str,
                         currency_data: pd.DataFrame,
                         percentages: bool = True):
        try:
            plt.style.use("seaborn-bright")
            plt.figure(figsize=(10, 5))
            plt.xlabel("Date")
            plt.ylabel(f"log(USD / {currency_name.upper()})")
            if percentages:
                plt.title(f"{currency_name.upper()} with {correction_name} % correction highlighted")
            else:
                plt.title(f"{currency_name.upper()} with {correction_name} correction highlighted")
            plt.scatter(currency_data.date, currency_data.low, s=0.1, c=currency_data[f'correction_{correction_name}'],
                        cmap='rainbow')
            plt.yscale('log')
            plt.savefig(f'{self.graphs_dir}/{currency_name}_{correction_name}.png')
            plt.show()
        except:
            raise ValueError("Could not plot graphs")


class SimpleCorrection(DrawdownAnalysis):
    def __init__(self,
                 drawdown_percentages: dict,
                 drawdown_names_in_pct: bool,
                 **kwargs):
        self.drawdown_percentages = drawdown_percentages
        self.drawdown_names_in_pct = drawdown_names_in_pct
        super().__init__(*kwargs.values())

    def calculate_drawdowns(self):
        for curr_name, curr_data in tqdm(self.cryptocurrency_data.items()):
            correction_pcts = []
            dates = list(curr_data['date'])
            for correction_name, drawdown_pct in self.drawdown_percentages.items():
                correction_market = []
                recent_high = curr_data.iloc[0]['open']
                for day in curr_data.itertuples():
                    if day.low < (recent_high * drawdown_pct):
                        correction_market.append(1)
                        forward_percentages = {'date': day.date}
                        for range_name, window in self.window_days.items():
                            forward_date = datetime.combine(day.date, datetime.min.time()) + timedelta(days=window)
                            forward_date = forward_date.date()
                            if forward_date in dates:
                                forward_price = curr_data[self.buy_in_column].loc[curr_data.date == forward_date].iloc[
                                    0]
                                forward_percentages[range_name] = return_pct(day.low, forward_price)
                        correction_pcts.append(forward_percentages)
                    else:
                        correction_market.append(0)
                        if day.high > recent_high:
                            recent_high = day.high

                corrections_df = pd.DataFrame(correction_pcts)
                corrections_df.dropna(inplace=True, thresh=len(self.window_days))
                corrections_df.to_csv(f'{self.output_dir}/{curr_name}_{correction_name}_correction.csv')

                curr_data[f'correction_{correction_name}'] = correction_market
                days_in_downturn = len(
                    curr_data[f'correction_{correction_name}'].loc[curr_data[f'correction_{correction_name}'] == 1])
                total_days = len(curr_data)
                print(
                    f'For {curr_name} {days_in_downturn} out of {total_days} days were {correction_name}% lower than their recent peak')
                curr_data.to_csv(f'{self.output_dir}/{curr_name}_with_corrections.csv')

                self.plot_corrections(currency_name=curr_name,
                                      correction_name=correction_name,
                                      currency_data=curr_data)

            self.returns_matrix = pd.DataFrame()
            for window_name, w_range in self.window_days.items():
                window_avgs = []
                for correction_name, drawdown_pct in self.drawdown_percentages.items():
                    csv_i = pd.read_csv(f'{self.output_dir}/{curr_name}_{correction_name}_correction.csv')
                    window_avgs.append(csv_i[window_name].mean())
                self.returns_matrix[window_name] = window_avgs

            if self.drawdown_names_in_pct:
                index_names = [f'{i} %' for i in self.drawdown_percentages.keys()]
            else:
                index_names = [f'{i} days' for i in self.drawdown_percentages.keys()]
            index = pd.Index(index_names)
            self.returns_matrix = self.returns_matrix.set_index(index)
            self.returns_matrix.to_csv(f'{self.output_dir}/output_{curr_name}.csv')


class ExpMovingAverage(DrawdownAnalysis):
    def __init__(self, ema_ranges: dict, **kwargs):
        self.ema_ranges = ema_ranges
        super().__init__(*kwargs.values())

    def calculate_drawdowns(self):
        lower_ema_idx = list(self.ema_ranges.values()).index(min(self.ema_ranges.values()))
        lower_ema_name = list(self.ema_ranges.keys())[lower_ema_idx]
        higher_ema_name = [a for a in self.ema_ranges.keys() if a != lower_ema_name][0]
        for curr_name, curr_data in tqdm(self.cryptocurrency_data.items()):
            dates = list(curr_data['date'])
            for ema_name, ema_range in self.ema_ranges.items():
                curr_data[ema_name] = curr_data['open'].ewm(span=ema_range, min_periods=ema_range).mean()
            curr_data['correction_ema'] = np.where(curr_data[lower_ema_name] < curr_data[higher_ema_name], 1, 0)

            curr_data.dropna(inplace=True, thresh=len(self.window_days))
            curr_data.to_csv(f'{self.output_dir}/{curr_name}_{lower_ema_name}_{higher_ema_name}_correction.csv')

            correction_pcts = []
            for day in curr_data.itertuples():
                forward_percentages = {'date': day.date}
                if day.correction_ema:
                    for range_name, window in self.window_days.items():
                        forward_date = datetime.combine(day.date, datetime.min.time()) + timedelta(days=window)
                        forward_date = forward_date.date()
                        if forward_date in dates:
                            forward_price = curr_data[self.buy_in_column].loc[curr_data.date == forward_date].iloc[0]
                            forward_percentages[range_name] = return_pct(day.low, forward_price)
                    correction_pcts.append(forward_percentages)

            corrections_df = pd.DataFrame(correction_pcts)
            corrections_df.dropna(inplace=True, thresh=len(self.window_days))
            corrections_df.to_csv(f'{self.output_dir}/{curr_name}_ema_correction.csv')

            days_in_downturn = len(
                curr_data[f'correction_ema'].loc[curr_data[f'correction_ema'] == 1])
            total_days = len(curr_data)
            print(
                f'For {curr_name} {days_in_downturn} out of {total_days} days where the shorter ema crossed the longer')
            curr_data.to_csv(f'{self.output_dir}/{curr_name}_with_corrections.csv')

            self.plot_corrections(currency_name=curr_name,
                                  correction_name='ema',
                                  currency_data=curr_data,
                                  percentages=False)

            # Create returns matrix
            self.returns_matrix = pd.DataFrame(index=pd.Index([f"{lower_ema_name} below {higher_ema_name}"]))
            for window_name, w_range in self.window_days.items():
                csv_i = pd.read_csv(f'{self.output_dir}/{curr_name}_ema_correction.csv')
                self.returns_matrix[window_name] = csv_i[window_name].mean()
            self.returns_matrix.to_csv(f'{self.output_dir}/output_{curr_name}.csv')


if __name__ == "__main__":
    eth_price.dropna(inplace=True)
    eth_price.reset_index(inplace=True, drop=True)

    WINDOW_DAYS = {'one': 1,
                   'three': 3,
                   'seven': 7,
                   'fourteen': 14,
                   'twenty_eight': 28,
                   'three_months': 91,
                   'six_months': 182,
                   'nine_months': 273,
                   'twelve_months': 365}

    DRAWDOWN_PERCENTAGES = {'five': 0.95,
                            'ten': 0.9,
                            'twenty': 0.8,
                            'thirty': 0.7,
                            'fourty': 0.6}

    CRYPTOCURRENCIES = {'btc': btc_price,
                        'eth': eth_price}

    EMA_RANGES = {'9_ema': 9,
                  '20_ema': 20}

    da = SimpleCorrection(analysis_dir_name='long_term',
                          window_days=WINDOW_DAYS,
                          buy_in_column='open',
                          drawdown_percentages=DRAWDOWN_PERCENTAGES,
                          drawdown_names_in_pct=True,
                          cryptocurrency_data=CRYPTOCURRENCIES)

    da_ema = ExpMovingAverage(analysis_dir_name='ema',
                              window_days=WINDOW_DAYS,
                              buy_in_column='open',
                              cryptocurrency_data=CRYPTOCURRENCIES,
                              ema_ranges=EMA_RANGES)

# # RECENT_HIGH_PRICE = 'high'
# BUY_IN_PRICE = 'open'
#
# eth_price = eth_price.iloc[1:]
#
#
#
# total_eth = return_pct(eth_price.iloc[0].open, eth_price.iloc[-1].close)
# print(f'Total ETH return from {str(eth_price.iloc[0].date)} to {str(eth_price.iloc[-1].date)}  period: {total_eth}')
#
# total_btc = return_pct(btc_price.iloc[0].open, btc_price.iloc[-1].close)
# print(f'Total BTC return from {str(btc_price.iloc[0].date)} to {str(btc_price.iloc[-1].date)} period: {total_btc}')
#
# # btc_price['correction'] = correction_market
# # plt.scatter(btc_price.date, btc_price.open, s=0.1, c=btc_price.correction, cmap='rainbow')
# # plt.savefig('fourty_percent_correct.png')
# # plt.show()

# eth_price_ema['ema_9'] = eth_price_ema['open'].ewm(span=9, min_periods=9).mean()
# eth_price_ema['ema_20'] = eth_price_ema['open'].ewm(span=20, min_periods=20).mean()
# eth_price_ema['ema_correction'] = np.where(eth_price['ema_9'] > eth_price['ema_20'], 0, 1)
