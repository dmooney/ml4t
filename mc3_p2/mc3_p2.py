# author: David Mooney (dmooney3)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import LinRegLearner as lrl
import KNNLearner as knn
import BagLearner as bl

def add_bb(prices, symbol):
    sma = pd.rolling_mean(prices[symbol], 20, 20)
    prices['bb_normed'] = (prices[symbol] - sma)/(2 * pd.rolling_std(prices[symbol], 20, 20))
    # prices['Lower_normed'] = prices[symbol] + prices['SMA']/-2 * pd.rolling_std(prices[symbol], 20, 20)
    return prices

if __name__ == "__main__":
    symbol = 'IBM'
    symbol = 'ML4T-220'


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
    prices['momentum'] = prices[symbol] / prices[symbol].shift(5) - 1.0
    X = prices.copy()
    del X[symbol]

    # Note to self: shifting backwards 5 days pulls the future (t+5) prices back to line up with the current price
    Y = (prices[symbol].shift(-5) / prices[symbol]) - 1.0
    # print(prices['XDIVY'].head(100))






    # chart = X.plot(title=symbol)
    # chart.set_xlabel("Date")
    # chart.set_ylabel("Price")
    # fig = chart.get_figure()
    # plt.show()

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

    # compute how much of the data is training and testing
    # train_rows = math.floor(0.6* data.shape[0])
    # test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = X['20071231':'20091231']
    trainY = Y['20071231':'20091231']
    testX = X['20091231':'20111231']
    testY = Y['20091231':'20111231']

    # print testX.shape
    # print testY.shape

    # create a learner and train it
    # learners = []
    # learners.append(lrl.LinRegLearner(verbose = True)) # create a LinRegLearner
    # learners.append(knn.KNNLearner(k = 3, verbose = False)) # constructor
    # learners.append(bl.BagLearner(learner = knn.KNNLearner, kwargs = {"k":3}, bags = 2, boost = False, verbose = False))
    # learners.append(bl.BagLearner(learner = knn.KNNLearner, kwargs = {"k":3}, bags = 5, boost = False, verbose = False))
    # learners.append(bl.BagLearner(learner = knn.KNNLearner, kwargs = {"k":3}, bags = 10, boost = False, verbose = False))
    # learners.append(bl.BagLearner(learner = knn.KNNLearner, kwargs = {"k":3}, bags = 15, boost = False, verbose = False))
    # for learner in learners:
    learner = lrl.LinRegLearner(verbose = True)
    print type(learner)
    learner.addEvidence(trainX, trainY) # train it

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = np.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]

    plot1 = prices.copy()
    del plot1['bb_normed']
    del plot1['15div75sma']
    del plot1[symbol + '_normed']
    del plot1['momentum']
    plot1["Training Y"] = ((1.0 + Y)  *  prices[symbol])
    plot1["Predicted Y"] = ((1.0 + predY) *  prices[symbol])
    print(plot1['20071231':'20091231'].tail())
    chart = plot1['20071231':'20091231'].plot(title=symbol)
    chart.set_xlabel("Date")
    chart.set_ylabel("Price")
    fig = chart.get_figure()
    fig.savefig(symbol + "_plot1.png")
    # plt.show()


    # evaluate out of sample
    # predY = learner.query(testX) # get the predictions
    # rmse = np.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    # print
    # print "Out of sample results"
    # print "RMSE: ", rmse
    # c = np.corrcoef(predY, y=testY)
    # print "corr: ", c[0,1]

