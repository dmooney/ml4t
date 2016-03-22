# author: David Mooney (dmooney3)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

if __name__ == "__main__":
    prices = get_data(['IBM'], pd.date_range(dt.datetime(2007,12,31), dt.datetime(2009,12,31)), False)
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

    prices['SMA'] = pd.rolling_mean(prices['IBM'], 20, 20)
    prices['Upper'] = prices['SMA'] + 2 * pd.rolling_std(prices['IBM'], 20, 20)
    prices['Lower'] = prices['SMA'] - 2 * pd.rolling_std(prices['IBM'], 20, 20)

    chart = prices.plot(title="IBM")
    chart.set_xlabel("Date")
    chart.set_ylabel("Price")
    chart.axvline(pd.datetime(2009,01,01))
    plt.show()

