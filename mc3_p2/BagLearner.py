"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np
import KNNLearner as knn

class BagLearner(object):
    learners = []
    boost = False

    def __init__(self, learner = knn.KNNLearner, kwargs = {"k":3}, bags = 20, boost = False, verbose = False):
        for i in xrange(bags):
            self.learners.append(learner(**kwargs))
        boost = boost

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        for learner in self.learners:
            randIndices = np.random.randint(0, len(dataX), len(dataX))
            learner.addEvidence(dataX[randIndices], dataY[randIndices])

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        values = np.zeros((len(points), len(self.learners)))
        i = 0
        for learner in self.learners:
            values[:,i] = learner.query(points)
            i = i + 1
        x = np.mean(values, axis=1)
        return x

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"
