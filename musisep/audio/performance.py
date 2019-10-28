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
    n = orig_signals.shape[0]

    target_sum = np.zeros((n, n))
    proj_sum = np.zeros(n)
    distortion = np.zeros((n, n))
    interference = np.zeros((n, n))
    artifacts = np.zeros(n)

    orig_signals = orig_signals / np.linalg.norm(orig_signals,
                                                 axis=1)[:, np.newaxis]
    proj_signals = project(synth_signals, orthogonalize(orig_signals))
    target_signals = (orig_signals.dot(synth_signals.T)[:, :, np.newaxis]
                      * orig_signals[:, np.newaxis, :])

    pos = np.arange(0, orig_signals.shape[1], size)
    for p in pos:
        target_sum += np.sum(np.square(target_signals[:, :, p:p+size]), axis=2)
        proj_sum += np.sum(np.square(proj_signals[:, p:p+size]), axis=1)
        distortion += np.sum(np.square(target_signals[:, :, p:p+size]
                                       - synth_signals[:, p:p+size]), axis=2)
        interference += np.sum(np.square(target_signals[:, :, p:p+size]
                                         - proj_signals[:, p:p+size]), axis=2)
        artifacts += np.sum(np.square(proj_signals[:, p:p+size]
                                      - synth_signals[:, p:p+size]), axis=1)

    perms = []
    measures = []

    sig_sar = 10 * np.log10(proj_sum / (artifacts + 1e-40))
    for p in it.permutations(range(n)):
        idcs0 = np.asarray(p)
        idcs1 = np.arange(n)
        sig_sdr = 10 * np.log10(target_sum[idcs0, idcs1]
                                / (distortion[idcs0, idcs1] + 1e-40))
        sig_sir = 10 * np.log10(target_sum[idcs0, idcs1]
                                / (interference[idcs0, idcs1] + 1e-40))
        perms.append(idcs0)
        measures.append(np.vstack((sig_sdr, sig_sir, sig_sar)))

    return perms, measures

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
