#!python3

"""
Module for calculation of the performance measures for blind source separation.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import itertools as it

def orthogonalize(signals):
    """
    Orthogonalize the given signals.

    Parameters
    ----------
    signals : array_like
        Matrix with the signals in its rows

    Returns
    -------
    q_matrix : ndarray
        Matrix with the orthogonalized signals in its rows
    """

    signals = np.asarray(signals)
    q_matrix, _ = np.linalg.qr(signals.T)

    return q_matrix.T

def project(signals, q_matrix):
    """
    Project the given signals on the given space.

    Parameters
    ----------
    signals : array_like
        Matrix with the signals in its rows
    q_matrix : array_like
        Matrix with an orthonormal basis of the space in its rows


    Returns
    -------
    proj_signals : ndarray
        Matrix with the projected signals in its rows
    """

    signals = np.asarray(signals)
    q_matrix = np.asarray(q_matrix)
    return q_matrix.T.dot(q_matrix.dot(signals.T)).T

class MeasureAcc:
    """
    Container object for the computation of performance measures.

    Parameters
    ----------
    n : int
        Number of instruments to consider
    """

    def __init__(self, n):
        self.n = n
        self.target_sum = np.zeros((n, n))
        self.proj_sum = np.zeros(n)
        self.distortion = np.zeros((n, n))
        self.interference = np.zeros((n, n))
        self.artifacts = np.zeros(n)

    def add(self, synth_signals, orig_signals):
        """
        Add portions of signals to the container.

        Parameters
        ----------
        synth_signals : array_like
            Matrix with the synthesized signals in its rows
        orig_signals : array_like
            Matrix with the original signals in its rows
        """

        synth_signals = np.asarray(synth_signals)
        orig_signals = np.asarray(orig_signals)

        orig_signals = orig_signals / np.linalg.norm(orig_signals,
                                                     axis=1)[:, np.newaxis]
        proj_signals = project(synth_signals, orthogonalize(orig_signals))
        target_signals = (orig_signals.dot(synth_signals.T)[:, :, np.newaxis]
                          * orig_signals[:, np.newaxis, :])
        
        self.target_sum += np.sum(np.square(target_signals), axis=2)
        self.proj_sum += np.sum(np.square(proj_signals), axis=1)
        self.distortion += np.sum(np.square(target_signals - synth_signals),
                                  axis=2)
        self.interference += np.sum(np.square(target_signals - proj_signals),
                                    axis=2)
        self.artifacts += np.sum(np.square(proj_signals - synth_signals),
                                 axis=1)

    def perms(self):
        """
        Obtain all the permutations of the signals and the related
        performance measures

        Returns
        -------
        perms : list of ndarray
            Permutations of the indices of the signals
        measures : list of ndarray
            Arrays with SDR, SIR, and SAR for the signals in rows
        """

        perms = []
        measures = []

        sig_sar = 10 * np.log10(self.proj_sum / self.artifacts)
        for p in it.permutations(range(self.n)):
            idcs0 = np.asarray(p)
            idcs1 = np.arange(self.n)
            sig_sdr = 10 * np.log10(self.target_sum[idcs0, idcs1]
                                    / self.distortion[idcs0, idcs1])
            sig_sir = 10 * np.log10(self.target_sum[idcs0, idcs1]
                                    / self.interference[idcs0, idcs1])
            perms.append(idcs0)
            measures.append(np.vstack((sig_sdr, sig_sir, sig_sar)))

        return perms, measures

def measures(synth_signals, orig_signals, size=1048576):
    """
    Compute the SDR, SIR, and SAR in all permutations of the synthesized
    signals.

    Parameters
    ----------
    synth_signals : array_like
        Array with the synthesized signals in its rows
    orig_signals : array_like
        Array with the original signals in its rows
    size : int
        Length of the signal fragments to consider at once

    Returns
    -------
    perms : list of ndarray
        Permutations of the indices of the signals
    measures : list of ndarray
        Arrays with SDR, SIR, and SAR for the signals in rows
    """

    synth_signals = np.asarray(synth_signals)
    orig_signals = np.asarray(orig_signals)

    m = MeasureAcc(orig_signals.shape[0])
    pos = np.arange(0, orig_signals.shape[1], size)
    for p in pos:
        m.add(synth_signals[:, p:p+size], orig_signals[:, p:p+size])

    return m.perms()

def select_perm(perms, measures):
    """
    Select the permutation with the highest SIR sum.

    Parameters
    ----------
    perms : list of array_like
        Permutations of the indices of the signals
    measures : list of array_like
        Arrays with SDR, SIR, and SAR for the signals in rows

    Returns
    -------
    best_perm : ndarray
        Permutation of `synth_signals` with the lowest SIR sum
    best_measure : ndarray
        SDR, SIR, and SAR in the permutation with the lowest SIR sum
    """

    perms = [np.asarray(p) for p in perms]
    measures = [np.asarray(m) for m in measures]

    sig_sirs = [np.sum(m[1, :]) for m in measures]
    idx = np.argmax(np.asarray(sig_sirs))
    perm = perms[idx]
    idcs = np.zeros(perm.size, dtype='int')
    idcs[perm] = np.arange(perm.size)
    return perm, measures[idx][:, idcs]
