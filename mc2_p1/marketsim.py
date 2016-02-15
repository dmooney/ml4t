"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here
    orders = pd.read_csv(orders_file, parse_dates=True)
    # print(orders)

    start_date = orders['Date'][0]
    end_date = orders['Date'][len(orders)-1]
    symbols = sorted(set(orders['Symbol']))
    prices = get_data(symbols, pd.date_range(start_date, end_date))

    cash = start_val
    holdings = {}
    for symbol in symbols:
        holdings[symbol] = 0

    for row in orders.iterrows():
        order = row[1]
        date = dt.datetime.strptime(order['Date'], '%Y-%m-%d')
        symbol = order['Symbol']
        order_type = order['Order']
        shares = order['Shares']
        if date in prices.index:
            price = prices[[symbol]].loc[date][0]
            if order_type == 'BUY':
                cash = cash - (price * shares)
                holdings[symbol] = holdings[symbol] + shares
            else:
                cash = cash + (price * shares)
                holdings[symbol] = holdings[symbol] - shares
        else:
            print("Rejecting trade on ", date)
        print(order, cash, holdings)


    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    portvals = get_data(['IBM'], pd.date_range(start_date, end_date))
    portvals = portvals[['IBM']]  # remove SPY

    return portvals

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

if __name__ == "__main__":
    test_code()
