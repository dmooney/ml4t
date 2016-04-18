"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np
import pandas as pd

class KNNLearner(object):

    def __init__(self, k = 3, verbose = False):
        self.k = k
        self.verbose = verbose

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        self.dataX = dataX
        self.dataY = dataY
        
    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        # print(points.shape, self.dataY.shape)
        values = pd.Series(index=points.index)
        i = 0
        for point in points.values:
            w = self.dataX - point
            x = w**2
            y = np.sum(x, axis=1)
            z = np.argsort(y)[:self.k]
            foo = np.mean(self.dataY[z])
            values[i] = foo
            i = i + 1
        return values

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
