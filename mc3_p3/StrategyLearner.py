"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import QLearner as ql
import pandas as pd
import util as ut
import numpy as np

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose

    def discretize(self, symbol, data, steps):
        sorted_data = data.sort_values(symbol)
        num_rows = sorted_data.shape[0]
        stepsize = num_rows/steps
        bins = []
        for i in xrange(steps):
            offset = min((i + 1) * stepsize, num_rows - 1)
            bins.append(sorted_data[symbol].iloc[offset])
        data[symbol] = np.digitize(data[symbol], bins)
        return data

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000): 

        self.learner = ql.QLearner()
        # add your code to do learning here
        syms=[symbol]
        dates = pd.date_range(sd, ed)

        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        momentum_label = symbol + "_momentum"
        prices_all[momentum_label] = prices_all[symbol] / prices_all[symbol].shift(5) - 1.0
        prices_all[4:] = self.discretize(momentum_label, prices_all[4:], 9)

        volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume = self.discretize(symbol, volume, 9)
        prices_all[symbol + "_volume"] = volume

        prices_all[symbol + "_sma15"] = prices_all[symbol].rolling(15, 15).mean()
        prices_all[14:] = self.discretize(symbol + "_sma15", prices_all[14:], 5)
        # prices_all[symbol + "_sma15"][15:] = self.discretize(symbol + "_sma15", prices_all[15:], 9)
        print(prices_all[13:])


        # example usage of the old backward compatible util function
        # syms=[symbol]
        # dates = pd.date_range(sd, ed)
        # prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        # prices = prices_all[syms]  # only portfolio symbols
        # prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        # if self.verbose: print prices
        #
        # # example use with new colname
        # volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
        # volume = volume_all[syms]  # only portfolio symbols
        # volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        # if self.verbose: print volume

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        trades.values[:,:] = 0 # set them all to nothing
        trades.values[3,:] = 100 # add a BUY at the 4th date
        trades.values[5,:] = -100 # add a SELL at the 6th date
        trades.values[6,:] = -100 # add a SELL at the 7th date
        trades.values[8,:] = -100 # add a SELL at the 9th date
        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        return trades

if __name__=="__main__":
    print "One does not simply think up a strategy"
    print pd.__version__