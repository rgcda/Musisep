#!python3

"""
Module containing the modified ADAM algorithm.
"""

from __future__ import absolute_import, division, print_function

import numpy as np

class Adam_B:
    """
    Object for the ADAM algorithm with bounds, adapted for the update of
    instrument dictionaries.  Each column refers to one instruments, and the
    harmonics are in rows.

    Parameters
    ----------
    init : array-like
        Initial value for the dictionary
    lo : float
        Lower bound for the dictionary entries
    hi : float
        Upper bound for the dictionary entries
    alpha : float
        Global step-size
    beta1 : float
        Inertia of the first moment estimator
    beta2 : float
        Inertia of the second moment estimator
    eps : float
        Value to add in the denominator to avoid division by zero
    """

    def __init__(self, init, lo=0, hi=1,
                 alpha=1e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        init = np.asarray(init)
        n = init.shape[1]
        self.lo = lo
        self.hi = hi
        self.theta = init
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(init.shape, dtype='double')
        self.v = np.zeros(n, dtype='double')
        self.t = np.zeros(n, dtype='double')

    def reset(self, i):
        """
        Reset an instrument to its initial state.

        Parameters
        ----------
        i : int
            Number of the instrument
        """

        self.m[:,i] = 0
        self.v[i] = 0
        self.t[i] = 0

    def step(self, stepdir):
        """
        Update the dictionary.

        Parameters
        ----------
        stepdir : array-like
            Step direction (negative gradient)

        Returns
        -------
        theta : ndarray
            New value of the dictionary
        """

        stepdir = np.asarray(stepdir)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * stepdir
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.mean(stepdir**2,
                                                                  axis=0)
        mcorr = self.m / (1 - self.beta1**self.t)
        vcorr = self.v / (1 - self.beta2**self.t)
        upd = self.alpha * mcorr / (np.sqrt(vcorr) + self.eps)
        self.theta = self.theta + upd
        self.theta = np.maximum(self.theta, self.lo)
        self.theta = np.minimum(self.theta, self.hi)

        return self.theta
