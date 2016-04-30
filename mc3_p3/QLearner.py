"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):
        """
        @param num_states:  number of states to consider
        @param num_actions: number of actions available
        @param alpha:       alpha float, the learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
        @param gamma:       gamma float, the discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
        @param rar:         rar float, random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
        @param radr:        radr float, random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
        @param dyna:        dyna integer, conduct this number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
        @param verbose:     verbose boolean, if True, your class is allowed to print debugging statements, if False, all printing is prohibited.
        """

        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose
        self.s = 0
        self.a = 0
        self.Q = np.random.rand(num_states, num_actions) * 2 - 1

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = self.potentially_random_action(s)
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        action = self.potentially_random_action(s_prime)

        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * \
                (r + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime])])
        self.a = action
        self.s = s_prime
        self.rar = self.rar * self.radr

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r,"rar = ",self.rar
        return action

    def potentially_random_action(self, s):
        if rand.randint(0, 1) * self.rar > 0.5:
            if self.verbose: print("Random action")
            action = rand.randint(0, self.num_actions-1)
        else:
            if self.verbose: print("Best action")
            action = np.argmax(self.Q[s])
        return action


if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
