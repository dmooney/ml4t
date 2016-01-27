"""Fit a line to a given set of data points using optimization. """

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

def error(line, data): # error function
    """Computer error between given line model and observed data.

        Parameters
        ----------
        line: tuple/list/array (c0,c1) where c0 is slope and c1 is y-intercept
        data: 2D array where each row is a point (x,y)

        Returns error as a single real value.
    """
    #metric: Sum of squared Y-axis differences
    err = np.sum((data[:,1] - (line[0] * data[:,0] + line[1])) ** 2)
    return err

def fit_line(data, error_func):
    """Fit a line to given data, using a supplied error function.

    Parameters
    ----------
    data: 2D array where each row is a point (X0, Y)
    error_func: function that computer the error between a line and observed data

    Returns line that minimizes the error function.
    """
    # Generate initial guess for line model
    l = np.float32([0, np.mean(data[:, 1])]) # slope = 0, intercept = mean(y values)

    #Plot initial guess (optional)
    x_ends = np.float32([-5,5])
    plt.plot(x_ends, l[0] * x_ends + l[1], 'm--', linewidth=2.0, label="Initial guess")  # m-- = magenta dashed line

    #Call optimizer to minimize error function
    result = spo.minimize(error_func, l, args=(data,), method='SLSQP', options={'disp':True})
    return result.x

def test_run_line():
    # Define original line
    l_orig = np.float32([4,2])
    print "Original line: C0 = {}, C1 = {}".format(l_orig[0], l_orig[1])
    Xorig=np.linspace(0,10,21)
    Yorig = l_orig[0]*Xorig + l_orig[1]
    plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label="Original line")  #b- = blue solid line

    # Generate noisy data points
    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:,0], data[:,1], 'go', label = "Data points")

    # Try to fit a line to this data
    l_fit = fit_line(data,error)
    print "Fitted line: C0 = {}, C1 = {}".format(l_fit[0], l_fit[1])
    plt.plot(data[:,0], l_fit[0] * data[:,0] + l_fit[1], 'r--', linewidth=2.0, label="Fitted line") #r-- = red dashed line
    plt.title("Comparison of original line, scattered data and fitted line",fontsize=12)
    plt.legend(loc='upper left')
    plt.show()

def test_run_poly():
    # Define original polynomial
    l_orig = np.float32([1.5,-10,-5,60,50])
    print "Original polynomial: C0 = {}, C1 = {}, C2 = {}, C3 = {}, C4 = {}".format(l_orig[0], l_orig[1], l_orig[2], l_orig[3], l_orig[4])
    Xorig=np.linspace(-6,6,25)
    Yorig = l_orig[0]*Xorig ** 4 + l_orig[1]*Xorig ** 3 + l_orig[2] * Xorig **2 + l_orig[3] * Xorig  + l_orig[4]
    plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label="Original polynomial")  #b- = blue solid line

    # Generate noisy data points
    noise_sigma = 100.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:,0], data[:,1], 'go', label = "Data points")

    # Try to fit a polynomial to this data
    p_fit = fit_poly(data,error_poly)
    # Why are the returns for p_fit the exact opposite from that of l_fit above?
    # p_fit[0] was the coefficent for the highest degree in the line example
    # above... but for the polynomial, p_fit[4] is the coefficient for the highest degree.
    print "Fitted polynomial: C0 = {}, C1 = {}, C2 = {}, C3 = {}, C4 = {}".format(p_fit[4], p_fit[3], p_fit[2], p_fit[1], p_fit[0])
    plt.plot(data[:,0], p_fit[4] * data[:,0]**4 + p_fit[3] * data[:,0]**3 + p_fit[2] * data[:,0]**2 + p_fit[1] * data[:,0]  + p_fit[0], 'r--', linewidth=2.0, label="Fitted line") #r-- = red dashed line
    plt.title("Comparison of original polynomial, scattered data and fitted curve",fontsize=12)
    plt.legend(loc='upper left')
    plt.show()

def error_poly(C, data):
    """ Compute error between given polynomial and observed data.

    Parameters
    ----------
    C: numpy.poly1d object or equivalent array representing polynomial coefficients
    data: 2D array where each row is a point (x,y)

    Returns error as a single real value.
    """
    #Metric : Sum of squared Y--axis differences
    err = np.sum((data[:,1] - np.polyval(C, data[:, 0])) ** 2)
    return err

def fit_poly(data, error_func, degree = 4):
    #I changed the above from degree = 3 to degree = 4.
    """Fit a polynomial to a give data, using supplied error function.

    Parameters
    ----------
    data: 2D array where each row is a point (x,y)
    error_func: function that computers the error between a polynomial and observed data

    Returns polynomial that minimizes the error function.
    """
    #Generate initial guess for polynomial model (all coeffs = 1)
    Cguess = np.poly1d(np.ones(degree + 1, dtype=np.float32))

    #Plot initial guess (optional)
    x = np.linspace(-5, 5, 21)
    plt.plot(x, np.polyval(Cguess, x), 'm--', linewidth=2.0, label="Initial guess")

    # Call optimizer to minimize error function
    result = spo.minimize(error_func, Cguess, args=(data,), method='SLSQP', options={'disp':True})
    return np.poly1d(result.x) # convert optimal result into a poly1d object and return it."

if __name__ == "__main__":
    test_run_line()
    test_run_poly()