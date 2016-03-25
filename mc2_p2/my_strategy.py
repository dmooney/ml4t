# author: David Mooney (dmooney3)

import pandas as pd
import matplotlib.pyplot as plt
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
        # print("rejected: ", date, symbol, order_type, shares, holdings, cash, leverage, longs, shorts)
        return cash, holdings
    else:
        # print("executed: ", date, symbol, order_type, shares, new_holdings, new_cash, leverage, longs, shorts)
        return new_cash, new_holdings

def print_stats(start_date, end_date, prices, portvals, sf=252.0, rfr=0.0):

    cum_ret = portvals["Value"].iget(-1)/portvals["Value"].iget(0) - 1

    dr = portvals.copy()
    dr[1:] = portvals[1:].values / portvals[:-1].values - 1.0
    dr = dr[1:]
    avg_daily_ret = dr.mean()
    std_daily_ret = dr.std()
    sharpe_ratio = np.sqrt(sf) * np.mean(dr - rfr) / std_daily_ret
    ev = portvals["Value"].iget(-1)

    # print "Date Range: {} to {}".format(start_date, end_date)
    # print
    # print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    # # print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    # print
    # print "Cumulative Return of Fund: {}".format(cum_ret)
    # # print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    # print
    # print "Standard Deviation of Fund: {}".format(std_daily_ret)
    # # print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    # print
    # print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    # # print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    # print
    # print "Final Portfolio Value: {}".format(portvals["Value"].iget(-1))
    #
    # print
    # print "Performance vs baseline: {}".format(cum_ret / 0.3524)
    return cum_ret/0.3524



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

    performance = print_stats(start_date, end_date, prices, portvals)

    return portvals, performance

def bollinger_strat(sma_window, prices):
    print(sma_window)

    # sma_window = 21

    prices['SMA'] = pd.rolling_mean(prices['IBM'], sma_window, sma_window)
    prices['Upper'] = prices['SMA'] + 2 * pd.rolling_std(prices['IBM'], sma_window, sma_window)
    prices['Lower'] = prices['SMA'] - 2 * pd.rolling_std(prices['IBM'], sma_window, sma_window)
    # prices['MACD'] = pd.ewma(prices['IBM'], min_periods=20, span=12) - pd.ewma(prices['IBM'], min_periods=20, span=26)
    # prices['Signal'] = pd.ewma(prices['MACD'], min_periods=20, span=9)
    prices['_above_upper'] = np.nan
    prices['_above_sma'] = np.nan
    prices['_below_lower'] = np.nan
    prices['_below_sma'] = np.nan
    prices['_above_upper'][sma_window - 1:] = prices['IBM'][sma_window - 1:] > prices['Upper'][sma_window - 1:]
    prices['_above_sma'][sma_window - 1:] =   prices['IBM'][sma_window - 1:] > prices['SMA'][sma_window - 1:]
    prices['_below_lower'][sma_window - 1:] = prices['IBM'][sma_window - 1:] < prices['Lower'][sma_window - 1:]
    prices['_below_sma'][sma_window - 1:] =   prices['IBM'][sma_window - 1:] < prices['SMA'][sma_window - 1:]
    # prices['_macd_above_signal'] = prices['MACD'][19:] > prices['Signal'][19:]
    # prices['_macd_below_signal'] = prices['MACD'][19:] < prices['Signal'][19:]
    # prices['SMAVol'] = pd.rolling_mean(prices['Volume'] / 1000000, 20, 20)
    # prices['_volSpike'] = prices['Volume'] > 2 * prices['Volume'].mean()
    # prices['_upDay'] = np.nan
    # prices['_upDay'][1:] = prices['IBM'][1:] > prices['IBM'][:-1]
    # prices['_volSpikeUpDay'] = np.logical_and(prices['_volSpike'], prices['_upDay'])
    # prices['_volSpikeDownDay'] = np.logical_and(prices['_volSpike'], np.logical_not(prices['_upDay']))
    # prices['_VolSpike'] = np.nan
    # prices['_VolSpike'][19:] = prices['Volume'][19:] > prices['SMAVol'][19:] + 8 * pd.rolling_std(prices['Volume'][19:], 20, 20)

    # chart = prices.plot(title="IBM", subplots=True )
    # chart[1].set_xlabel("Date")
    # chart[1].set_ylabel("Price")

    order_cols = ("Date", "Symbol", "Order", "Shares")
    orders = pd.DataFrame(columns=order_cols)

    state = 'cash'

    for i in xrange(sma_window ,prices.shape[0]):
        today = prices.index[i]
        if state == 'cash':
            if np.logical_and(prices['_below_lower'][i - 1],  np.logical_not(prices['_below_lower'][i])):
                state = 'long'
                # chart[1].axvline(today, color="g")
                orders.loc[len(orders)]=[today, "IBM", "BUY", 100]
            elif np.logical_and(prices['_above_upper'][i - 1],  np.logical_not(prices['_above_upper'][i])):
                state = 'short'
                # chart[1].axvline(today, color="r")
                orders.loc[len(orders)]=[today, "IBM", "SELL", 100]
        elif state == 'long':
            if np.logical_and(prices['_below_sma'][i - 1],  np.logical_not(prices['_below_sma'][i])):
                state = 'cash'
                # chart[1].axvline(today, color="k")
                orders.loc[len(orders)]=[today, "IBM", "SELL", 100]
        else:
            if np.logical_and(prices['_above_sma'][i - 1],  np.logical_not(prices['_above_sma'][i])):
                state = 'cash'
                # chart[1].axvline(today, color="k")
                orders.loc[len(orders)]=[today, "IBM", "BUY", 100]



    # plt.show()

    orders.to_csv('orders.csv', index=False)
    portvals = compute_portvals('orders.csv', start_val=10000)

    start_date = orders['Date'].iloc[0]
    end_date = orders['Date'].iloc[len(orders)-1]
    prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date), addSPY=False)
    prices_SPX.dropna(inplace=True)

    port_val_normed = portvals / portvals.ix[0]
    prices_SPX_normed = prices_SPX / prices_SPX.ix[0]
    df_temp = pd.concat([port_val_normed, prices_SPX_normed], keys=['Portfolio', '$SPX'], axis=1)
    # plot_data(df_temp)

def sma_strat(smaX, smaY, prices):
    # print(smaX, smaY)

    sma_window = 20

    prices['SMAX'] = pd.rolling_mean(prices['IBM'], smaX, smaX)
    prices['SMAY'] = pd.rolling_mean(prices['IBM'], smaY, smaY)
    # prices['SMA'] = pd.rolling_mean(prices['IBM'], sma_window, sma_window)
    # prices['Upper'] = prices['SMA'] + 2 * pd.rolling_std(prices['IBM'], sma_window, sma_window)
    # prices['Lower'] = prices['SMA'] - 2 * pd.rolling_std(prices['IBM'], sma_window, sma_window)
    # prices['MACD'] = pd.ewma(prices['IBM'], min_periods=20, span=12) - pd.ewma(prices['IBM'], min_periods=20, span=26)
    # prices['Signal'] = pd.ewma(prices['MACD'], min_periods=20, span=9)
    prices['_x_above_y'] = np.nan
    prices['_y_above_x'] = np.nan
    prices['_x_above_y'][smaY - 1:] = prices['SMAX'][smaY - 1:] > prices['SMAY'][smaY - 1:]
    prices['_y_above_x'][smaY - 1:] = prices['SMAY'][smaY - 1:] > prices['SMAX'][smaY - 1:]
    # prices['_above_upper'] = np.nan
    # prices['_above_sma'] = np.nan
    # prices['_below_lower'] = np.nan
    # prices['_below_sma'] = np.nan
    # prices['_above_upper'][sma_window - 1:] = prices['IBM'][sma_window - 1:] > prices['Upper'][sma_window - 1:]
    # prices['_above_sma'][sma_window - 1:] =   prices['IBM'][sma_window - 1:] > prices['SMA'][sma_window - 1:]
    # prices['_below_lower'][sma_window - 1:] = prices['IBM'][sma_window - 1:] < prices['Lower'][sma_window - 1:]
    # prices['_below_sma'][sma_window - 1:] =   prices['IBM'][sma_window - 1:] < prices['SMA'][sma_window - 1:]
    # prices['_macd_above_signal'] = prices['MACD'][19:] > prices['Signal'][19:]
    # prices['_macd_below_signal'] = prices['MACD'][19:] < prices['Signal'][19:]
    # prices['SMAVol'] = pd.rolling_mean(prices['Volume'] / 1000000, 20, 20)
    # prices['_volSpike'] = prices['Volume'] > 2 * prices['Volume'].mean()
    # prices['_upDay'] = np.nan
    # prices['_upDay'][1:] = prices['IBM'][1:] > prices['IBM'][:-1]
    # prices['_volSpikeUpDay'] = np.logical_and(prices['_volSpike'], prices['_upDay'])
    # prices['_volSpikeDownDay'] = np.logical_and(prices['_volSpike'], np.logical_not(prices['_upDay']))
    # prices['_VolSpike'] = np.nan
    # prices['_VolSpike'][19:] = prices['Volume'][19:] > prices['SMAVol'][19:] + 8 * pd.rolling_std(prices['Volume'][19:], 20, 20)

    chart = prices.plot(title="IBM")
    chart.set_xlabel("Date")
    chart.set_ylabel("Price")

    order_cols = ("Date", "Symbol", "Order", "Shares")
    orders = pd.DataFrame(columns=order_cols)

    state = 'cash'

    for i in xrange(smaY + 10, prices.shape[0]):
        today = prices.index[i]
        if state == 'cash':
            if prices['_x_above_y'][i - 10] and not prices['_x_above_y'][i-9] and not prices['_x_above_y'][i]:
                state = 'short'
                chart.axvline(today, color="r")
                orders.loc[len(orders)]=[today, "IBM", "SELL", 100]
            # elif np.logical_and(prices['_below_lower'][i - 1],  np.logical_not(prices['_below_lower'][i])):
            #     state = 'long'
            #     chart.axvline(today, color="g")
            #     orders.loc[len(orders)]=[today, "IBM", "BUY", 100]
        elif state == 'long':
            if prices['_x_above_y'][i - 10] and not prices['_x_above_y'][i-9] and not prices['_x_above_y'][i]:
                state = 'short'
                chart.axvline(today, color="r")
                orders.loc[len(orders)]=[today, "IBM", "SELL", 100]
                orders.loc[len(orders)]=[today, "IBM", "SELL", 100]
            # if np.logical_and(prices['_below_sma'][i - 1],  np.logical_not(prices['_below_sma'][i])):
            #     state = 'cash'
            #     chart.axvline(today, color="k")
            #     orders.loc[len(orders)]=[today, "IBM", "SELL", 100]
        else:
            if np.logical_and(prices['_y_above_x'][i - 1],  np.logical_not(prices['_y_above_x'][i])):
                state = 'long'
                chart.axvline(today, color="g")
                orders.loc[len(orders)]=[today, "IBM", "BUY", 100]
                orders.loc[len(orders)]=[today, "IBM", "BUY", 100]


    if state == 'long':
        orders.loc[len(orders)]=[today, "IBM", "SELL", 100]
        chart.axvline(today, color="k")
    elif state == 'short':
        orders.loc[len(orders)]=[today, "IBM", "BUY", 100]
        chart.axvline(today, color="k")

    plt.show()

    orders.to_csv('orders.csv', index=False)
    portvals, performance = compute_portvals('orders.csv', start_val=10000)

    # start_date = orders['Date'].iloc[0]
    # end_date = orders['Date'].iloc[len(orders)-1]
    # prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date), addSPY=False)
    # prices_SPX.dropna(inplace=True)

    # port_val_normed = portvals / portvals.ix[0]
    # prices_SPX_normed = prices_SPX / prices_SPX.ix[0]
    # df_temp = pd.concat([port_val_normed, prices_SPX_normed], keys=['Portfolio', '$SPX'], axis=1)
    # plot_data(df_temp)

    print(smaX, smaY, performance)


if __name__ == "__main__":
    prices = get_data(['IBM'], pd.date_range(dt.datetime(2007,12,31), dt.datetime(2009,12,31)), False)
    prices.dropna(inplace=True)

    # for i in xrange(15,35):
    #     bollinger_strat(i, prices.copy())

    # for i in xrange(5,20):
    #     for j in xrange(50,150,25):
    #         sma_strat(i, j, prices.copy())

    sma_strat(14, 75, prices.copy())






