import pandas as pd
from datetime import datetime, timedelta
from data import eth_fees, eth_price, btc_fees, btc_price
import config
from typing import Union, List, Optional
import matplotlib.pyplot as plt

# Conversions
WEI_TO_ETH = 1 / 1E18
SAT_TO_BTC = 1 / 1E8
THREE_MONTHS_DAYS = 91
SIX_MONTHS_DAYS = 182
NINE_MONTHS_DAYS = 273
TWELVE_MONTH_DAYS = 365

# Fee Calculations
BYTES_PER_TRANSACTION = 259  # (input_total * 180) + (output_total * 34) + 10 (+/- input_total); we assume 1 in 2 out
GAS_UNITS_PER_TRANSACTION = 21000

# Backtest config
BACKTEST_LENGTH_YEARS = 4
END_DATE = datetime(2021, 9, 9).date()
# START_DATE = END_DATE - timedelta(days=round(BACKTEST_LENGTH_YEARS * 365.25))
START_DATE = datetime(2017, 9, 11).date()

# Strategy config
STARTING_DEPOSIT = 10000  # USD
DEPOSITS_PER_YEAR = 1


class Portfolio:
    def __init__(self,
                 deposits_per_year=0,
                 recurring_deposit=0,
                 starting_deposit=STARTING_DEPOSIT,
                 start_date=START_DATE,
                 end_date=END_DATE,
                 eth_prices=eth_price,
                 btc_prices=btc_price,
                 eth_fees=eth_fees,
                 btc_fees=btc_fees):
        dates = pd.date_range(start_date, end_date).to_pydatetime().tolist()
        self.dates = [d.date() for d in dates]

        self.eth_prices = eth_prices
        self.btc_prices = btc_prices
        self.eth_fees = eth_fees
        self.btc_fees = btc_fees

        self.daily_returns = pd.DataFrame()
        self.portfolio = pd.DataFrame(columns=['date', 'usd', 'btc', 'eth'])

        self.cash_balance = starting_deposit
        if deposits_per_year:
            deposit_per_days = round(365 / deposits_per_year)
        else:
            deposit_per_days = False
        days_since_deposit = 0

        # DEBUG
        self.failed_dates = []

        for date in self.dates:
            if deposit_per_days:
                if days_since_deposit % deposit_per_days == 0:
                    self.cash_balance += recurring_deposit
            self.generate_portfolio(date)
            days_since_deposit += 1

        self.returns = self.calculate_returns()

    def generate_portfolio(self, date: datetime.date):
        raise NotImplementedError('Must implement portfolio indicator/strategy')

    def allocate_to_portfolio(self, date: Union[datetime.date, List[datetime.date]],
                              eth: Union[float, List[float]] = 0,
                              btc: Union[float, List[float]] = 0):
        self.portfolio = self.portfolio.append({'date': date,
                                                'eth': eth,
                                                'btc': btc,
                                                'usd': self.cash_balance},
                                               ignore_index=True)

    def calculate_returns(self):
        # Convert daily portfolio value into USD
        daily_value_usd = []
        for row in self.portfolio.itertuples():
            daily_btc_usd = self._get_btc_price(row.date)
            daily_eth_usd = self._get_eth_price(row.date)
            daily_usd = row.usd + (row.btc * daily_btc_usd) + (row.eth * daily_eth_usd)
            daily_value_usd.append({'date': row.date,
                                    'value': daily_usd})

        portfolio_returns = pd.DataFrame(daily_value_usd)
        portfolio_returns['daily_return'] = portfolio_returns['value'].pct_change()

        return portfolio_returns

    def convert_btc(self, date: datetime.date, amount: float, usd_to_btc: bool = True) -> float:
        """
        Returns the amount of BTC or USD the user can get on conversion, taking in to account transaction costs
        :param date:
        :param amount: BTC / USD
        :param usd_to_btc: Whether the user is converting USD -> BTC (True) or reverse (False)
        :return: USD or BTC user has after transaction
        """
        btc_price_usd = self._get_btc_price(date)
        transaction_fee_btc = self._get_btc_fee(date)
        if usd_to_btc:
            transaction_fee_usd = transaction_fee_btc / btc_price_usd
            amount_available = amount - transaction_fee_usd
            return amount_available / btc_price_usd
        amount_available = amount - transaction_fee_btc
        return amount_available * btc_price_usd

    def convert_eth(self, date: datetime.date, amount: float, usd_to_eth: bool = True) -> float:
        """
        Returns the amount of ETH or USD the user can get on conversion, taking in to account transaction costs
        :param date:
        :param amount: BTC / USD
        :param usd_to_eth: Whether the user is converting USD -> ETH (True) or reverse (False)
        :return: USD or ETH user has after transaction
        """
        eth_price_usd = self._get_eth_price(date)
        transaction_fee_eth = self._get_eth_fee(date)
        if usd_to_eth:
            transaction_fee_usd = transaction_fee_eth / eth_price_usd
            amount_available = amount - transaction_fee_usd
            return amount_available / eth_price_usd
        amount_available = amount - transaction_fee_eth
        return amount_available * eth_price_usd

    # Getters / Setters
    def _get_btc_price(self, date: datetime.date) -> float:
        btc_usd = self.btc_prices['open'].loc[self.btc_prices.date == date]
        if len(btc_usd):
            return btc_usd.iloc[0]
        previous_day = date - timedelta(days=1)
        if previous_day not in self.dates:
            raise ValueError('No data for BTC price')
        print(f'BTC price for {date} does not exist')
        self.failed_dates.append(('BTC', date))
        return self._get_btc_price(previous_day)

    def _get_eth_price(self, date: datetime.date) -> float:
        eth_usd = self.eth_prices['open'].loc[self.eth_prices.date == date]
        if len(eth_usd):
            return eth_usd.iloc[0]
        print(f'ETH price for {date} does not exist')
        previous_day = date - timedelta(days=1)
        if previous_day not in self.dates:
            raise ValueError('No data for ETH price')
        self.failed_dates.append(('ETH', date))
        return self._get_eth_price(previous_day)

    def _get_btc_fee(self, date: datetime.date) -> float:
        """
        Returns the fee in BTC for a transaction
        :param date:
        :return:
        """
        fee_sat_per_byte = self.btc_fees['value'].loc[self.btc_fees.date == date]
        if len(fee_sat_per_byte):
            return fee_sat_per_byte.iloc[0] * BYTES_PER_TRANSACTION * SAT_TO_BTC
        print(f'BTC fee for {date} does not exist')
        previous_day = date - timedelta(days=1)
        if previous_day not in self.dates:
            raise ValueError('No data for BTC fee')
        return self._get_btc_fee(previous_day)

    def _get_eth_fee(self, date: datetime.date) -> float:
        """
        Returns the fee in ETH for a transaction
        :param date:
        :return:
        """
        fee_wei_per_gas = self.eth_fees['value'].loc[self.eth_fees.date == date]
        if len(fee_wei_per_gas):
            return fee_wei_per_gas.iloc[0] * GAS_UNITS_PER_TRANSACTION * WEI_TO_ETH
        print(f'ETH fee for {date} does not exist')
        previous_day = date - timedelta(days=1)
        if previous_day not in self.dates:
            raise ValueError('No data for ETH fee')
        return self._get_eth_fee(previous_day)

    def _get_previous_allocation(self) -> pd.Series:
        return self.portfolio.iloc[-1]


class BenchmarkBuyHold(Portfolio):
    def __init__(self, eth_weight, btc_weight):
        assert eth_weight + btc_weight == 1
        assert 0 <= eth_weight <= 1
        assert 0 <= btc_weight <= 1
        self.eth_weight = eth_weight
        self.btc_weight = btc_weight
        super(BenchmarkBuyHold, self).__init__(recurring_deposit=0)

    def generate_portfolio(self, date: datetime.date):
        if self.cash_balance:
            eth_alloc_usd = self.cash_balance * self.eth_weight
            if eth_alloc_usd:
                eth = self.convert_eth(date=date, amount=eth_alloc_usd, usd_to_eth=True)
            else:
                eth = 0

            btc_alloc_usd = self.cash_balance * self.btc_weight
            if btc_alloc_usd:
                btc = self.convert_btc(date=date, amount=btc_alloc_usd, usd_to_btc=True)
            else:
                btc = 0
            self.cash_balance = self.cash_balance - (eth_alloc_usd + btc_alloc_usd)
            self.allocate_to_portfolio(date=date, eth=eth, btc=btc)
            # print(f'Added {eth} eth, {btc} btc with {self.cash_balance} usd remaining')  # Debug
        else:
            # Allocation remains the same as previous day
            previous_alloc = self._get_previous_allocation()
            self.allocate_to_portfolio(date=date,
                                       eth=previous_alloc['eth'],
                                       btc=previous_alloc['btc'])


class Naive(Portfolio):
    def __init__(self, position_duration, eth_weight, btc_weight):
        assert eth_weight + btc_weight == 1
        assert 0 <= eth_weight <= 1
        assert 0 <= btc_weight <= 1
        self.eth_weight = eth_weight
        self.btc_weight = btc_weight

        self.position_duration = position_duration
        self.days_since_purchase = 0
        super(Naive, self).__init__(recurring_deposit=0)

    def generate_portfolio(self, date: datetime.date):
        if self.days_since_purchase % self.position_duration == 0:
            print(f'rebalance on {date}')
            self.days_since_purchase = 0
            if self.dates.index(date) != 0:
            # Check whether investor has positions to sell
                previous_alloc = self._get_previous_allocation()
                eth_holdings = previous_alloc['eth']
                if eth_holdings:
                    usd_from_eth = self.convert_eth(date=date, amount=eth_holdings, usd_to_eth=False)
                else:
                    usd_from_eth = 0

                btc_holdings = previous_alloc['btc']
                if btc_holdings:
                    usd_from_btc = self.convert_btc(date=date, amount=btc_holdings, usd_to_btc=False)
                else:
                    usd_from_btc = 0

                self.cash_balance = self.cash_balance + usd_from_btc + usd_from_eth

            if self.cash_balance:
                eth_alloc_usd = self.cash_balance * self.eth_weight
                if eth_alloc_usd:
                    eth = self.convert_eth(date=date, amount=eth_alloc_usd, usd_to_eth=True)
                else:
                    eth = 0

                btc_alloc_usd = self.cash_balance * self.btc_weight
                if btc_alloc_usd:
                    btc = self.convert_btc(date=date, amount=btc_alloc_usd, usd_to_btc=True)
                else:
                    btc = 0
                self.cash_balance = self.cash_balance - (eth_alloc_usd + btc_alloc_usd)
                self.allocate_to_portfolio(date=date, eth=eth, btc=btc)
        self.days_since_purchase += 1
        print(f'{self.days_since_purchase} days since rebalance')


class Opportunistic(Portfolio):
    def __init__(self, position_duration, eth_weight, btc_weight):
        assert eth_weight + btc_weight == 1
        assert 0 <= eth_weight <= 1
        assert 0 <= btc_weight <= 1
        self.eth_weight = eth_weight
        self.btc_weight = btc_weight

        self.position_duration = position_duration
        self.days_since_purchase = 0
        super(Opportunistic, self).__init__(recurring_deposit=0)

    def generate_portfolio(self, date: datetime.date):
        pass


if __name__ == "__main__":
    # Calculate buy and hold benchmarks
    even_split_benchmark = BenchmarkBuyHold(eth_weight=0.5, btc_weight=0.5)
    eth_benchmark = BenchmarkBuyHold(eth_weight=1, btc_weight=0)
    btc_benchmark = BenchmarkBuyHold(eth_weight=0, btc_weight=1)

    # Calculate rebalanced strategies
    naive_eth = Naive(30, 0.5, 0.5)
    naive_eth2 = Naive(60, 0.5, 0.5)
    naive_eth3 = Naive(90, 0.5, 0.5)

    # Plot benchmarks
    plt.plot(btc_benchmark.returns.date, btc_benchmark.returns.value, label="BTC Buy & Hold")
    plt.plot(eth_benchmark.returns.date, eth_benchmark.returns.value, label="ETH Buy & Hold")

    plt.legend()
    plt.show()

    # Plot strategies
    plt.plot(even_split_benchmark.returns.date, eth_benchmark.returns.value, label="ETH/BTC Buy & Hold")
    plt.plot(naive_eth.returns.date, naive_eth.returns.value, label="naive 30 days")
    plt.plot(naive_eth2.returns.date, naive_eth2.returns.value, label="naive 60 days")
    plt.plot(naive_eth3.returns.date, naive_eth3.returns.value, label="naive 90 days")

    plt.legend()
    plt.show()