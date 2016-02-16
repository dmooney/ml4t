"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

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
    if leverage > 2.0:
        print("rejected: ", date, symbol, order_type, shares, holdings, cash, leverage, longs, shorts)
        return cash, holdings
    else:
        print("executed: ", date, symbol, order_type, shares, new_holdings, new_cash, leverage, longs, shorts)
        return new_cash, new_holdings

def print_stats(start_date, end_date, prices, portvals, sf=252.0, rfr=0.0):

    cum_ret = (portvals["Value"].iget(-1)/portvals["Value"].iget(0)) - 1


    dr = portvals.copy()
    dr[1:] = portvals[1:].values / portvals[:-1].values - 1
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




def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
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

    portvals = pd.DataFrame(index=prices.index, columns=['Value'])

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
        print(portvals.ix[index])

    print_stats(start_date, end_date, prices, portvals)

    return portvals

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-leverage-2.csv"
    print(of)
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    # plot_data(portvals)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # print(portvals)
    
    # # Get portfolio stats
    # # Here we just fake the data. you should use your code from previous assignments.
    # start_date = dt.datetime(2008,1,1)
    # end_date = dt.datetime(2008,6,1)
    # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    # cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]
    #
    # # Compare portfolio against $SPX
    # print "Date Range: {} to {}".format(start_date, end_date)
    # print
    # print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    # print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    # print
    # print "Cumulative Return of Fund: {}".format(cum_ret)
    # print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    # print
    # print "Standard Deviation of Fund: {}".format(std_daily_ret)
    # print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    # print
    # print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    # print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    # print
    # print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
