# author: David Mooney (dmooney3)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def add_bb(prices, symbol):
    sma = pd.rolling_mean(prices[symbol], 20, 20)
    prices['bb_normed'] = (prices[symbol] - sma)/(2 * pd.rolling_std(prices[symbol], 20, 20))
    # prices['Lower_normed'] = prices[symbol] + prices['SMA']/-2 * pd.rolling_std(prices[symbol], 20, 20)
    return prices

if __name__ == "__main__":
    symbol = 'IBM'


    prices = get_data([symbol], pd.date_range(dt.datetime(2007,12,31), dt.datetime(2011,12,31)), False)
    prices.dropna(inplace=True)
    prices = add_bb(prices, symbol)

    # chart = prices.plot(title="IBM")
    # chart.set_xlabel("Date")
    # chart.set_ylabel("Price")
    # fig = chart.get_figure()
    # fig.savefig('bollinger.png')

    # bb_value = (prices[symbol] - prices['SMA'])/(2 * prices[symbol].std())
    # momentum = ((prices[symbol]/prices[symbol].shift(-5)) - 1.0).shift(5)
    # prices["Momentum"] = momentum + prices[symbol]

    smaX = 15
    smaY = 75
    indcX = 'SMA' + str(smaX)
    indcY = 'SMA' + str(smaY)

    sma15 = pd.rolling_mean(prices[symbol], smaX, smaX)
    sma75 = pd.rolling_mean(prices[symbol], smaY, smaY)
    # print(prices[indcX].head(100))
    # print(prices[indcY].head(100))
    prices['15div75sma'] = (sma15/sma75) - 1.0
    prices[symbol + '_normed'] = prices[symbol] / np.mean(prices[symbol]) - 1.0
    X = prices.copy()
    del X[symbol]
    Y = (prices[symbol].shift(-5) / prices[symbol]) - 1.0
    # print(prices['XDIVY'].head(100))






    chart = X.plot(title=symbol)
    chart.set_xlabel("Date")
    chart.set_ylabel("Price")
    fig = chart.get_figure()
    plt.show()

    # chart2 = momentum.plot(title="momentum")
    # chart2.set_xlabel("Date")
    # chart2.set_ylabel("momentum")
    # fig = chart2.get_figure()
    # fig.savefig('momentum.png')
    # plt.show(1)



    # print(bb_value.tail())
    # print(prices['IBM'][5:].head())
    # print(prices['IBM'][:-5].head())
    # print((prices['IBM']/prices['IBM'].shift(5)).head(10))
    # momentum = (prices['IBM']/prices['IBM'].shift(5)) - 1
    # print(momentum.tail(30))


