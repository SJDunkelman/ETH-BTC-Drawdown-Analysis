import pandas as pd
from datetime import datetime
import re
import numpy as np
from typing import Union

# Price data
# eth_price = pd.read_csv('eth_price-2015-2021.csv')
eth_price = pd.read_csv('input/ETH-USD_2015-2021.csv')
btc_price = pd.read_csv('input/btc_price_2016-2021.csv')

# Transaction fee data
eth_fees = pd.read_csv('input/eth_gas_fees.csv')
btc_fees = pd.read_csv('input/btc_transaction_fees.csv')

# Preprocess data
data = [eth_price, eth_fees, btc_price, btc_fees]

# Standardise column names
for df in data:
    df.columns = ['date' if re.search(r'(?i)date', c) is not None or re.search(r'(?i)timestamp', c) \
                      else c.replace(' ', '_').lower() for c in df.columns]

eth_fees.rename(columns={'value_(wei)': 'value'}, inplace=True)

# Convert dates into datetime.date
# eth_price['date'] = pd.to_datetime(eth_price['date'], unit='s').dt.date
eth_price['date'] = eth_price['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
eth_price['date'] = eth_price['date'].dt.date
btc_price['date'] = pd.to_datetime(btc_price['date'], format="%b-%d-%Y").dt.date
eth_fees['date'] = pd.to_datetime(eth_fees['date'], format="%m/%d/%Y").dt.date
btc_fees['date'] = btc_fees['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
btc_fees['date'] = btc_fees['date'].dt.date


# # Add suffix to column names for clarity
# btc_price.columns = [f'{name}_btc' if name != 'date' else name for name in btc_price.columns]
# eth_price.columns = [f'{name}_eth' if name != 'date' else name for name in eth_price.columns]

def sort_chronologically(df: pd.DataFrame):
    df.sort_values(by='date', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)


for df in data:
    sort_chronologically(df)

eth_price.replace('undefined', np.nan, inplace=True)


def preprocess_numeric(number: Union[int, float, str]) -> float:
    if type(number) == str:
        number = number.replace(',', '')
        number = re.sub(r'\s+', '', number)
        numeric = float(re.search(r'[\d.]+', number).group())
        if re.search(r'(?i)m', number) is not None:
            numeric = numeric * 1E6
        elif re.search(r'(?i)k', number) is not None:
            numeric = numeric * 1E3
        return numeric
    return number


eth_price = eth_price.applymap(lambda x: preprocess_numeric(x))
eth_price.dropna(inplace=True)
eth_price.reset_index(inplace=True, drop=True)