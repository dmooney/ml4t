"""
Test a Strategy Learner.  (c) 2016 Tucker Balch
"""

import pandas as pd
import datetime as dt
import util as ut
import StrategyLearner as sl
import numpy as np

def execute_order(cash, holdings, date, order, symbol, prices):
    # date = dt.datetime.strptime(order['Date'], '%Y-%m-%d')
    # symbol = order['Symbol']
    if order == 100:
        order_type = 'BUY'
    else:
        order_type = 'SELL'
    # order_type = order['Order']
    # shares = order['Shares']
    shares = 100

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



def compute_portvals(df_trades, symbol, start_val = 1000000, baseline_perf=0.3524):
    # this is the function the autograder will call to test your code
    # orders = pd.read_csv(orders_file, parse_dates=True)


    start_date = df_trades.index[0]
    end_date = df_trades.index[-1]
    symbols = [symbol]
    prices = ut.get_data(symbols, pd.date_range(start_date, end_date))

    cash = start_val
    holdings = {}
    for symbol in symbols:
        holdings[symbol] = 0

    portvals = pd.DataFrame(index=prices.index, columns=['Value'], dtype=np.float64)
    # print(type(portvals))

    for index in portvals.index:

        # If there are orders on this date, execute them
        trade = df_trades.ix[index][symbol]
        cash, holdings = execute_order(cash, holdings, index, trade, symbol, prices)

        # Calculate port val for this date
        portvals.ix[index] = cash
        for symbol in holdings:
            if (holdings[symbol] != 0):
                price = prices[[symbol]].loc[index.date()][0]
                portvals.ix[index] = portvals.ix[index] + (price * holdings[symbol])
    # print(portvals.ix[index])

    performance = print_stats(start_date, end_date, prices, portvals, baseline_perf=baseline_perf)

    return portvals, performance


def test_code(verb = True):

    # instantiate the strategy learner
    learner = sl.StrategyLearner(verbose = verb)

    # set parameters for training the learner
    sym = "ML4T-220"
    stdate =dt.datetime(2007,12,31)
    enddate =dt.datetime(2009,12,31) # just a few days for "shake out"

    # train the learner
    learner.addEvidence(symbol = sym, sd = stdate, \
        ed = enddate, sv = 10000) 

    # set parameters for testing
    sym = "ML4T-220"
    stdate =dt.datetime(2009,12,31)
    enddate =dt.datetime(2011,12,31)

    # get some data for reference
    syms=[sym]
    dates = pd.date_range(stdate, enddate)
    prices_all = ut.get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    if verb: print prices

    # test the learner
    df_trades = learner.testPolicy(symbol = sym, sd = stdate, \
        ed = enddate, sv = 10000)

    # a few sanity checks
    # df_trades should be a single column DataFrame (not a series)
    # including only the values 100, 0, -100
    if isinstance(df_trades, pd.DataFrame) == False:
        print "Returned result is not a DataFrame"
    if prices.shape != df_trades.shape:
        print "Returned result is not the right shape"
    tradecheck = abs(df_trades.cumsum()).values
    tradecheck[tradecheck<=100] = 0
    tradecheck[tradecheck>0] = 1
    if tradecheck.sum(axis=0) > 0:
        print "Returned result violates holding restrictions (more than 100 shares)"

    if verb: print df_trades
    # we will add code here to evaluate your trades
    # df_trades.to_csv('orders.csv', index=False)
    portvals, performance = compute_portvals(df_trades, sym, start_val=10000)
    print(performance)
    start_date = df_trades.iloc[0]
    end_date = df_trades.iloc[-1]
    prices_SPX = ut.get_data(['$SPX'], pd.date_range(start_date, end_date), addSPY=False)
    prices_SPX.dropna(inplace=True)
    port_val_normed = portvals / portvals.ix[0]
    prices_SPX_normed = prices_SPX / prices_SPX.ix[0]
    df_temp = pd.concat([port_val_normed, prices_SPX_normed], keys=['Portfolio_' + sym, '$SPX'], axis=1)


if __name__=="__main__":
    test_code(verb = True)
