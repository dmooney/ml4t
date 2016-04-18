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


def execute_order(cash, holdings, order, prices):
    date = dt.datetime.strptime(order['Date'], '%Y-%m-%d')
    symbol = order['Symbol']
    order_type = order['Order']
    shares = order['Shares']

    new_holdings = holdings.copy()

    if date in prices.index:
        price = prices[[symbol]].loc[date][0]
        if order_type == 'BUY':
            new_cash = cash - (price * shares)
            new_holdings[symbol] = holdings[symbol] + shares
        else:
            new_cash = cash + (price * shares)
            new_holdings[symbol] = holdings[symbol] - shares

    longs = 0
    shorts = 0
    for holding in new_holdings:
        price = prices[[holding]].loc[date][0]
        if (new_holdings[holding] > 0):
            longs = longs + price * new_holdings[holding]
        elif (new_holdings[holding] < 0):
            shorts = shorts + price * new_holdings[holding]

    leverage = (longs + abs(shorts)) / (longs + shorts + new_cash)
    # if leverage > 2.0:
    #     # print("rejected: ", date, symbol, order_type, shares, holdings, cash, leverage, longs, shorts)
    #     return cash, holdings
    # else:
        # print("executed: ", date, symbol, order_type, shares, new_holdings, new_cash, leverage, longs, shorts)
    return new_cash, new_holdings

def print_stats(start_date, end_date, prices, portvals, sf=252.0, rfr=0.0, baseline_perf=0.3524):

    cum_ret = portvals["Value"].iget(-1)/portvals["Value"].iget(0) - 1

    dr = portvals.copy()
    dr[1:] = portvals[1:].values / portvals[:-1].values - 1.0
    dr = dr[1:]
    avg_daily_ret = dr.mean()
    std_daily_ret = dr.std()
    sharpe_ratio = np.sqrt(sf) * np.mean(dr - rfr) / std_daily_ret
    ev = portvals["Value"].iget(-1)

    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    # print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    # print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    # print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    # print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals["Value"].iget(-1))

    # print
    # print "Performance vs baseline: {}".format(cum_ret / baseline_perf)
    # return cum_ret/baseline_perf



def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, baseline_perf=0.3524):
    # this is the function the autograder will call to test your code
    orders = pd.read_csv(orders_file, parse_dates=True)

    start_date = orders['Date'].iloc[0]
    end_date = orders['Date'].iloc[len(orders)-1]
    symbols = sorted(set(orders['Symbol']))
    prices = get_data(symbols, pd.date_range(start_date, end_date))

    cash = start_val
    holdings = {}
    for symbol in symbols:
        holdings[symbol] = 0

    portvals = pd.DataFrame(index=prices.index, columns=['Value'], dtype=np.float64)
    # print(type(portvals))

    for index in portvals.index:

        # If there are orders on this date, execute them
        date_orders = orders[orders['Date'] == str(index.date())]
        for row in date_orders.iterrows():
            order = row[1]
            cash, holdings = execute_order(cash, holdings, order, prices)

        # Calculate port val for this date
        portvals.ix[index] = cash
        for symbol in holdings:
            if (holdings[symbol] != 0):
                price = prices[[symbol]].loc[index.date()][0]
                portvals.ix[index] = portvals.ix[index] + (price * holdings[symbol])
        # print(portvals.ix[index])

    performance = print_stats(start_date, end_date, prices, portvals, baseline_perf=baseline_perf)

    return portvals, performance

def add_bb(prices, symbol):
    sma = pd.rolling_mean(prices[symbol], 20, 20)
    prices['bb_normed'] = (prices[symbol] - sma)/(2 * pd.rolling_std(prices[symbol], 20, 20))
    # prices['Lower_normed'] = prices[symbol] + prices['SMA']/-2 * pd.rolling_std(prices[symbol], 20, 20)
    return prices

def mc3p2(symbol):
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

    train_start = '20071231'
    train_end = '20091231'
    test_start = '20091231'
    test_end = '20111231'
    trainX = X[train_start:train_end]
    trainY = Y[train_start:train_end]
    testX = X[test_start:test_end]
    testY = Y[test_start:test_end]

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
    # learner = lrl.LinRegLearner(verbose = True)
    learner = knn.KNNLearner(k = 3, verbose = False)
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
    # print(plot1['20071231':'20091231'].tail())
    chart = plot1[train_start:train_end].plot(title=symbol + " in sample prediction")
    chart.set_xlabel("Date")
    chart.set_ylabel("Price")
    fig = chart.get_figure()
    fig.savefig(symbol + "_in_plot1.png")
    # plt.show()

    sim(predY, prices, train_start, train_end, symbol, "in sample")

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = np.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]

    sim(predY, prices, test_start, test_end, symbol, "out of sample")


def sim(predY, prices, start, end, symbol, sample_str):
    orders_chart_prices = prices[start:end].copy()
    del orders_chart_prices['bb_normed']
    del orders_chart_prices['15div75sma']
    del orders_chart_prices[symbol + '_normed']
    del orders_chart_prices['momentum']
    order_chart = orders_chart_prices.plot(title=symbol + " " + sample_str + " entry/exit")
    order_chart.set_xlabel("Date")
    order_chart.set_ylabel("Price")
    order_cols = ("Date", "Symbol", "Order", "Shares")
    orders = pd.DataFrame(columns=order_cols)
    state = 'cash'
    hold = 0
    for i in xrange(predY.shape[0]):
        today = orders_chart_prices.index[i]
        # print(i, today, predY[i])

        if hold > 0:
            hold = hold - 1
            continue

        if state == 'cash':
            if predY[i] > 0.01:
                state = 'long'
                order_chart.axvline(today, color="g")
                orders.loc[len(orders)] = [today, symbol, "BUY", 100]
                hold = 5
            elif predY[i] < -0.01:
                state = 'short'
                order_chart.axvline(today, color="r")
                orders.loc[len(orders)] = [today, symbol, "SELL", 100]
                hold = 5

        elif state == 'short':
            if predY[i] > 0 and predY[i] < 0.01:
                state = 'cash'
                order_chart.axvline(today, color="k")
                orders.loc[len(orders)] = [today, symbol, "BUY", 100]
                hold = 5
            elif predY[i] > 0.01:
                state = 'long'
                order_chart.axvline(today, color="g")
                orders.loc[len(orders)] = [today, symbol, "BUY", 100]
                orders.loc[len(orders)] = [today, symbol, "BUY", 100]
                hold = 5

        else:  # long
            if predY[i] < 0.01 and predY[i] > 0:
                state = 'cash'
                order_chart.axvline(today, color="k")
                orders.loc[len(orders)] = [today, symbol, "SELL", 100]
                hold = 5
            elif predY[i] < -0.01:
                state = 'short'
                order_chart.axvline(today, color="r")
                orders.loc[len(orders)] = [today, symbol, "SELL", 100]
                orders.loc[len(orders)] = [today, symbol, "SELL", 100]
                hold = 5
    if state == 'long':
        orders.loc[len(orders)] = [today, symbol, "SELL", 100]
        order_chart.axvline(today, color="k")
    elif state == 'short':
        orders.loc[len(orders)] = [today, symbol, "BUY", 100]
        order_chart.axvline(today, color="k")
    fig = order_chart.get_figure()
    fig.savefig(symbol + '_' + sample_str + '_entry_exit.png')
    # plt.show()
    baseline_perf = 0.3524
    if len(orders) > 1:
        orders.to_csv('orders.csv', index=False)
        portvals, performance = compute_portvals('orders.csv', start_val=10000, baseline_perf=baseline_perf)
    else:
        performance = 0.0
    print(performance)
    start_date = orders['Date'].iloc[0]
    end_date = orders['Date'].iloc[len(orders) - 1]
    prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date), addSPY=False)
    prices_SPX.dropna(inplace=True)
    port_val_normed = portvals / portvals.ix[0]
    prices_SPX_normed = prices_SPX / prices_SPX.ix[0]
    df_temp = pd.concat([port_val_normed, prices_SPX_normed], keys=['Portfolio_' + symbol, '$SPX'], axis=1)
    plot_data(df_temp, title=symbol + " " + sample_str + " backtest")


if __name__ == "__main__":
    symbols = ['IBM', 'ML4T-220']

    for symbol in symbols:
        mc3p2(symbol)