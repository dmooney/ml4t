# author: David Mooney (dmooney3)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

if __name__ == "__main__":
    prices = get_data(['IBM'], pd.date_range(dt.datetime(2007,12,31), dt.datetime(2009,12,31)), False)
    # prices.fillna(method='ffill', inplace=True)
    # prices.fillna(method='bfill', inplace=True)
    prices.dropna(inplace=True)

    prices['SMA'] = pd.rolling_mean(prices['IBM'], 20, 20)
    prices['Upper'] = prices['SMA'] + 2 * pd.rolling_std(prices['IBM'], 20, 20)
    prices['Lower'] = prices['SMA'] - 2 * pd.rolling_std(prices['IBM'], 20, 20)
    prices['_above_upper'] = np.nan
    prices['_above_sma'] = np.nan
    prices['_below_lower'] = np.nan
    prices['_below_sma'] = np.nan
    prices['_above_upper'][19:] = prices['IBM'][19:] > prices['Upper'][19:]
    prices['_above_sma'][19:] =   prices['IBM'][19:] > prices['SMA'][19:]
    prices['_below_lower'][19:] = prices['IBM'][19:] < prices['Lower'][19:]
    prices['_below_sma'][19:] =   prices['IBM'][19:] < prices['SMA'][19:]
    #
    # prices['_long_entry'][20:] =  np.logical_and(prices['_below_lower'][19:-1],  np.logical_not(prices['_below_lower'][20:]))
    # prices['_long_exit'][20:] =   np.logical_and(prices['_below'][19:-1],  np.logical_not(prices['_below'][20:]))
    # prices['_short_entry'][20:] = np.logical_and(prices['_below'][19:-1],  np.logical_not(prices['_below'][20:]))
    # prices['_short_exit'][20:] =  np.logical_and(prices['_below'][19:-1],  np.logical_not(prices['_below'][20:]))

    chart = prices.plot(title="IBM")
    chart.set_xlabel("Date")
    chart.set_ylabel("Price")

    state = 'cash'

    for i in xrange(20,prices.shape[0]):
        # chart.axvline(pd.datetime(2009,01,01), color="g")
        # print(prices.index[i], prices.iloc[i])
        today = prices.index[i]
        if state == 'cash':
            if np.logical_and(prices['_below_lower'][i - 1],  np.logical_not(prices['_below_lower'][i])):
                state = 'long'
                chart.axvline(today, color="g")
                print("Going long")
            elif np.logical_and(prices['_above_upper'][i - 1],  np.logical_not(prices['_above_upper'][i])):
                state = 'short'
                chart.axvline(today, color="r")
        elif state == 'long':
            if np.logical_and(prices['_below_sma'][i - 1],  np.logical_not(prices['_below_sma'][i])):
                state = 'cash'
                chart.axvline(today, color="k")
        else:
            if np.logical_and(prices['_above_sma'][i - 1],  np.logical_not(prices['_above_sma'][i])):
                state = 'cash'
                chart.axvline(today, color="k")



    plt.show()

