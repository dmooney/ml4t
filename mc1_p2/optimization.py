"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as opt

def daily_values(prices, allocs):
    normed = prices/prices.ix[0]
    alloced = normed * allocs
    return alloced.sum(axis=1)

def daily_returns(port_val):
    dr = port_val.copy()
    dr[1:] = (port_val[1:] / port_val[:-1].values) - 1
    dr = dr[1:]
    return dr

def sharpe_ratio(dr):
    return np.sqrt(252) * dr.mean() / dr.std()

def stats(port_val, dr):
    cr = (port_val[-1]/port_val[0]) - 1
    adr = dr.mean()
    sddr = dr.std()
    sr = sharpe_ratio(dr)
    return cr, adr, sddr, sr

def f(allocs, prices):
    port_val = daily_values(prices, allocs)
    dr = daily_returns(port_val)
    cr, adr, sddr, sr = stats(port_val, daily_returns(port_val))
    return -float(sr)


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    # initial_alloc = [1.0/len(syms)] * len(syms)
    initial_alloc = np.fromiter((1.0/len(syms) for i in xrange(len(syms))), dtype='float')
    port_val = daily_values(prices, initial_alloc)

    bounds = [(0.0, 1.0)] * len(syms)
    constraints = ({ 'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs) })

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    result = opt.minimize(f, initial_alloc, args=(prices,), method='SLSQP', options={'disp':True}, bounds=bounds, constraints=constraints)
    allocs = result.x
    cr, adr, sddr, sr = stats(port_val, daily_returns(port_val))

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        prices_SPY_normed = prices_SPY / prices_SPY.ix[0]
        opt_val = daily_values(prices, allocs)
        opt_val_normed = opt_val / opt_val.ix[0]
        df_temp = pd.concat([opt_val_normed, prices_SPY_normed], keys=[','.join(syms), 'SPY'], axis=1)
        plot_data(df_temp)

    return allocs, cr, adr, sddr, sr

if __name__ == "__main__":
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2009,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['IBM', 'AAPL', 'HNZ', 'XOM', 'GLD']


    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date, syms = symbols, gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations, np.sum(allocations)
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr
    #
    # start_date = dt.datetime(2010,1,1)
    # end_date = dt.datetime(2010,12,31)
    # symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    #
    #
    # # Assess the portfolio
    # allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date, syms = symbols, gen_plot = True)
    #
    # # Print statistics
    # print "Start Date:", start_date
    # print "End Date:", end_date
    # print "Symbols:", symbols
    # print "Allocations:", allocations
    # print "Sharpe Ratio:", sr
    # print "Volatility (stdev of daily returns):", sddr
    # print "Average Daily Return:", adr
    # print "Cumulative Return:", cr
    #
    # start_date = dt.datetime(2004,1,1)
    # end_date = dt.datetime(2006,1,1)
    # symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']
    #
    #
    # # Assess the portfolio
    # allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date, syms = symbols, gen_plot = True)
    #
    # # Print statistics
    # print "Start Date:", start_date
    # print "End Date:", end_date
    # print "Symbols:", symbols
    # print "Allocations:", allocations
    # print "Sharpe Ratio:", sr
    # print "Volatility (stdev of daily returns):", sddr
    # print "Average Daily Return:", adr
    # print "Cumulative Return:", cr
    #
    #
    # start_date = dt.datetime(2004,12,1)
    # end_date = dt.datetime(2006,5,31)
    # symbols = ['YHOO', 'XOM', 'GLD', 'HNZ']
    #
    #
    # # Assess the portfolio
    # allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date, syms = symbols, gen_plot = True)
    #
    # # Print statistics
    # print "Start Date:", start_date
    # print "End Date:", end_date
    # print "Symbols:", symbols
    # print "Allocations:", allocations
    # print "Sharpe Ratio:", sr
    # print "Volatility (stdev of daily returns):", sddr
    # print "Average Daily Return:", adr
    # print "Cumulative Return:", cr
    #
    #
    # start_date = dt.datetime(2005,12,1)
    # end_date = dt.datetime(2006,5,31)
    # symbols = ['YHOO', 'HPQ', 'GLD', 'HNZ']
    #
    #
    # # Assess the portfolio
    # allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date, syms = symbols, gen_plot = True)
    #
    # # Print statistics
    # print "Start Date:", start_date
    # print "End Date:", end_date
    # print "Symbols:", symbols
    # print "Allocations:", allocations
    # print "Sharpe Ratio:", sr
    # print "Volatility (stdev of daily returns):", sddr
    # print "Average Daily Return:", adr
    # print "Cumulative Return:", cr