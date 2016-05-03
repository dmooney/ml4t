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
        prices_all[14:] = self.discretize(symbol + "_sma15", prices_all[14:], 9)
        # prices_all[symbol + "_sma15"][15:] = self.discretize(symbol + "_sma15", prices_all[15:], 9)
        # print(prices_all[13:])

        sma = pd.rolling_mean(prices_all[symbol], 20, 20)
        prices_all['bb_normed'] = (prices_all[symbol] - sma)/(2 * pd.rolling_std(prices_all[symbol], 20, 20))
        prices_all[19:] = self.discretize('bb_normed', prices_all[19:], 9)

        sma15 = pd.rolling_mean(prices_all[symbol], 15, 15)
        sma75 = pd.rolling_mean(prices_all[symbol], 75, 75)
        prices_all['15div75sma'] = (sma15/sma75) - 1.0
        prices_all[74:] = self.discretize('15div75sma', prices_all[74:], 9)
        prices_all[symbol + '_normed'] = prices_all[symbol] / np.mean(prices_all[symbol]) - 1.0
        prices_all[:] = self.discretize(symbol + '_normed', prices_all[:], 9)

        if self.verbose: print(prices_all.head())

        self.learner = ql.QLearner(
            num_states = 10 * 10 * 10 * 3,
            num_actions = 3,
            # dyna = 25,
            rar = 0.5)

        best_final_value = -1000000
        previous_final_value = -1000000
        learn_iteration = 0
        self.learn_loop(prices_all, sv, symbol)
        converge_countdown = 10
        while converge_countdown > 0 and learn_iteration < 200:
            previous_final_value = self.current_value
            self.learn_loop(prices_all, sv, symbol)
            if self.current_value > sv:
                if self.current_value > best_final_value:
                    best_final_value = self.current_value
                    converge_countdown = 10
                else:
                    converge_countdown = converge_countdown - 1
            # if self.current_value - previous_final_value < 100:
            #     converge_countdown = converge_countdown - 1
            # else:
            #     converge_countdown = 5
            learn_iteration = learn_iteration + 1
            if self.verbose: print("Learning iteration: ", learn_iteration, "Convergence Countdown", converge_countdown)

    def learn_loop(self, prices_all, sv, symbol):
        # for each day in training data
        # compute current state (cumret + holding)
        # compute reward for last action
        # query learner with current stat and reward to get an action
        # implement the action the learner returned (BUY, SELL, NOTHING) and update port value
        portvals = []
        dr = []

        self.cash = sv
        self.current_value = self.cash
        self.pos = 1  # cash
        first = True
        penalty = 1.0
        for date in prices_all[74:].index:
            # print(date)
            # mom = prices_all.ix[date][2]
            # vol = prices_all.ix[date][3]
            # sma = prices_all.ix[date][4]
            # s_prime = self.pos * 1000 + mom * 100 + vol * 10 + sma
            s_prime = self.sprime(prices_all, date)
            if first:
                a = self.learner.querysetstate(s_prime)
                first = False
            else:
                # if (len(portvals) > 1):
                #     dr.append(portvals[-1]/portvals[-2] - 1.0)
                #     avg_daily_ret = ((len(dr) - 1) * avg_daily_ret + dr[-1]) / len(dr)
                # else:
                #     avg_daily_ret = 0.0
                # # print(avg_daily_ret)
                # a = self.learner.query(s, avg_daily_ret)
                # r = sv/self.current_value - 1.0
                a = self.learner.query(s_prime, self.current_value * penalty)

            price = prices_all.ix[date][symbol]

            penalty = 1.0
            # trade
            if a == 0:  # buy
                if self.pos == 0:
                    self.pos = 1
                    self.cash = self.cash - price * 100
                    # print("buy")
                elif self.pos == 1:
                    self.pos = 2
                    self.cash = self.cash - price * 100
                    # print("buy")
                else:
                    pass
                    # penalty = 0.8
            elif a == 1:  # sell
                if self.pos == 1:
                    self.pos = 0
                    self.cash = self.cash + price * 100
                    # print("sell")
                elif self.pos == 2:
                    self.pos = 1
                    self.cash = self.cash + price * 100
                    # print("sell")
                else:
                    pass
                    # penalty = 0.8
            else:  # nothing
                pass

            if self.pos == 0:
                self.current_value = self.cash - price * 100
            elif self.pos == 1:
                self.current_value = self.cash
            else:
                self.current_value = self.cash + price * 100

            portvals.append(self.current_value)


        if self.verbose: print(self.current_value)
        return self.current_value

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

    def sprime(self, prices_all, date):
        mom = prices_all.ix[date][2]
        vol = prices_all.ix[date][3]
        sma = prices_all.ix[date][4]
        bb = prices_all.ix[date][5]
        x = prices_all.ix[date][6]
        norm = prices_all.ix[date][7]
        s_prime = self.pos * 1000 + bb * 100 + x * 10 + norm
        return s_prime


    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        syms=[symbol]
        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        momentum_label = symbol + "_momentum"
        prices_all[momentum_label] = prices_all[symbol] / prices_all[symbol].shift(5) - 1.0
        prices_all[4:] = self.discretize(momentum_label, prices_all[4:], 9)

        volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume = self.discretize(symbol, volume, 9)
        prices_all[symbol + "_volume"] = volume

        prices_all[symbol + "_sma15"] = prices_all[symbol].rolling(15, 15).mean()
        prices_all[14:] = self.discretize(symbol + "_sma15", prices_all[14:], 9)


        sma = pd.rolling_mean(prices_all[symbol], 20, 20)
        prices_all['bb_normed'] = (prices_all[symbol] - sma)/(2 * pd.rolling_std(prices_all[symbol], 20, 20))
        prices_all[19:] = self.discretize('bb_normed', prices_all[19:], 9)

        sma15 = pd.rolling_mean(prices_all[symbol], 15, 15)
        sma75 = pd.rolling_mean(prices_all[symbol], 75, 75)
        prices_all['15div75sma'] = (sma15/sma75) - 1.0
        prices_all[74:] = self.discretize('15div75sma', prices_all[74:], 9)
        prices_all[symbol + '_normed'] = prices_all[symbol] / np.mean(prices_all[symbol]) - 1.0
        prices_all[:] = self.discretize(symbol + '_normed', prices_all[:], 9)




        trades = prices_all[[symbol,]]  # only portfolio symbols
        trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        trades.values[:,:] = 0 # set them all to nothing
        # trades.values[3,:] = 100 # add a BUY at the 4th date
        # trades.values[5,:] = -100 # add a SELL at the 6th date
        # trades.values[6,:] = -100 # add a SELL at the 7th date
        # trades.values[8,:] = -100 # add a SELL at the 9th date
        self.cash = sv
        self.current_value = self.cash
        self.pos = 1  # cash
        first = True
        i = 0
        for date in prices_all[74:].index:
            s_prime = self.sprime(prices_all, date)
            if first:
                a = self.learner.querysetstate(s_prime)
                first = False
            else:
                a = self.learner.query(s_prime, self.current_value)

            price = prices_all.ix[date][symbol]

            penalty = 1.0
            # trade
            if a == 0:  # buy
                if self.pos == 0:
                    self.pos = 1
                    self.cash = self.cash - price * 100
                    # print("buy")
                    trades.values[i,:] = 100
                elif self.pos == 1:
                    self.pos = 2
                    self.cash = self.cash - price * 100
                    trades.values[i,:] = 100
                    # print("buy")
                else:
                    pass
                    # penalty = 0.8
            elif a == 1:  # sell
                if self.pos == 1:
                    self.pos = 0
                    self.cash = self.cash + price * 100
                    # print("sell")
                    trades.values[i,:] = -100
                elif self.pos == 2:
                    self.pos = 1
                    self.cash = self.cash + price * 100
                    # print("sell")
                    trades.values[i,:] = -100
                else:
                    pass
                    # penalty = 0.8
            else:  # nothing
                pass

            if self.pos == 0:
                self.current_value = self.cash - price * 100
            elif self.pos == 1:
                self.current_value = self.cash
            else:
                self.current_value = self.cash + price * 100

            i = i + 1

        if self.pos == 0: # short
            trades.values[i,:] = 100
        elif self.pos == 2: # long
            trades.values[i,:] = -100

        # if self.verbose: print type(trades) # it better be a DataFrame!
        # if self.verbose: print trades
        # if self.verbose: print prices_all
        return trades

if __name__=="__main__":
    print "One does not simply think up a strategy"
    print pd.__version__