#!python3

"""
Module for the training of the dictionary.  When invoked, a performance test
on artificial data is performed.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

from . import exptool
from . import pursuit
from . import adam_b
from ..audio import wav
from ..audio import performance

beta = 0.9

def make_closures(fsigma):
    """
    Build the functions that give the bounds and initial values.

    Parameters
    ----------
    fsigma : float
        Standard deviation of the Gaussian in the time domain.

    Returns
    -------
    make_bounds : lambda (length)
        Lambda that gives the bounds for `length` peaks
    make_inits : lambda (length)
        Lambda that gives the initial values `length` peaks
    """

    make_bounds = (lambda length :
                   [(0, None)] * length
                   + [(None, None)] * length
                   + [(fsigma*0.8, fsigma*1.5)] * length
                   + [(0, 2e-3)] * length)
    make_inits = (lambda length :
                  (np.repeat(fsigma, length), np.zeros(length)))

    return make_bounds, make_inits

def stoch_grad(y, inst_dict, tone_num, adam, fsigma, harscale, baseshift,
               inst_spect, pexp, qexp):
    """
    Perform a dictionary training step.

    Parameters
    ----------
    y : array_like
        Log-frequency spectrum to represent
    inst_dict : ndarray
        Dictionary containing the relative amplitudes of the harmonics
    tone_num : int
        Maximum number of simultaneous tones for each instrument
    adam : Adam_B
        Container object for the ADAM optimizer
    fsigma : float
        Standard deviation (frequency)
    harscale : float
        Scaling factor
    baseshift : int
        Length to add to the spectrum in order to avoid circular convolution
    inst_spect : array_like
        Spectra of the instruments, in the columns
    pexp : float
        Exponent for the addition of sinusoids
    qexp : float
        Exponent to be applied on the spectrum

    Returns
    -------
    inst_dict : ndarray
        Updated dictionary
    reconstruction : ndarray
        Synthesized spectrum
    inst_amps : ndarray
        Summed amplitudes for each instruments
    """

    y = np.asarray(y)

    fixed_params = (inst_dict, harscale)
    make_bounds, make_inits = make_closures(fsigma)
    
    peaks, reconstruction = pursuit.peak_pursuit(
        y, tone_num, 1, tone_num*inst_dict.shape[1]*2, inst_dict.shape[1],
        pursuit.inst_shift, pursuit.inst_shift_obj, pursuit.inst_shift_grad,
        make_bounds, make_inits, fixed_params, pursuit.fft_selector,
        (baseshift, inst_spect, qexp), pexp, qexp, beta)

    inst_amps = np.zeros(inst_dict.shape[1])
    for i in range(inst_dict.shape[1]):
        idcs, = np.where(peaks.insts == i)
        inst_amps[i] = np.sum(peaks.amps[idcs])

    grad = pursuit.inst_shift_dict_grad(peaks.get_array(), peaks.insts,
                                        fixed_params, pexp, qexp,
                                        y.size, y)
    abserr_old = np.linalg.norm(reconstruction**qexp - y**qexp)
    print("abserr (before): %g" % abserr_old)

    inst_dict = adam.step(-grad)

    reconstruction_new = pursuit.inst_shift(peaks, fixed_params, pexp, y.size)
    abserr_new = np.linalg.norm(reconstruction_new**qexp - y**qexp)
    print("abserr  (after): %g" % abserr_new)

    if abserr_new >= abserr_old:
        print("residual increased")

    return inst_dict, reconstruction, inst_amps

def gen_random_inst(har):
    """
    Generate random harmonic amplitudes according to a Par(1,2) distribution.

    Parameters
    ----------
    har : int
       Number of harmonics

    Returns
    -------
    inst : ndarray
        Harmonic amplitudes for one instrument, unified to an interval of [0,1]
    """

    e = - np.random.pareto(2)
    coeffs = np.random.rand(har)
    inst = np.arange(1, har+1) ** e * coeffs

    return inst / np.amax(inst)

def gen_random_inst_dict(har, inst_num):
    """
    Generate a random instrument dictionary according to a Par(1,2)
    distribution.

    Parameters
    ----------
    har : int
       Number of harmonics
    inst_num : int
       Number of instruments

    Returns
    -------
    inst_dict : ndarray
        Dictionary with instruments in columns, unified to an interval of [0,1]
    """

    inst_dict = np.zeros((har, inst_num))
    for j in range(inst_num):
        inst_dict[:, j] = gen_random_inst(har)

    return inst_dict

class Learner:
    """
    Container object for the dictionary learning process.

    Parameters
    ----------
    fsigma : float
        Standard deviation (frequency)
    tone_num : int
        Maximum number of simultaneous tones for each instrument
    inst_num : int
        Number of instruments in the dictionary
    har : int
        Number of harmonics
    m : int
        Height of the log-frequency spectrogram
    minfreq : float
        Minimum frequency to be represented (included)
    maxfreq : float
        Maximum frequency to be represented (excluded)
    lifetime : int
        Number of steps after which to renew the dictionary
    pexp : float
        Exponent for the addition of sinusoids
    qexp : float
        Exponent to be applied on the spectrum
    init : array_like
        Initial value for the dictionary
    """

    def __init__(self, fsigma, tone_num, inst_num, har, m,
                 minfreq, maxfreq, lifetime,
                 pexp, qexp, init=None):
        self.fsigma = fsigma
        self.tone_num = tone_num
        self.inst_num = inst_num
        self.har = har
        self.m = m
        self.lifetime = lifetime
        self.baseshift = m
        self.harscale = pursuit.calc_harscale(minfreq, maxfreq, m)

        if init is None:
            self.inst_dict = gen_random_inst_dict(har, inst_num)
        else:
            self.inst_dict = np.asarray(init)

        self.adam = adam_b.Adam_B(self.inst_dict, alpha=1e-3)

        self.inst_amps = np.zeros(inst_num)
        self.inst_cnt = np.zeros(inst_num)

        self.pexp = pexp
        self.qexp = qexp

        self.cnt = 0

    def learn(self, y):
        """
        Learning step.  Automatically renews the dictionary.

        Parameters
        ----------
        y : array_like
            Log-frequency spectrum

        Returns
        -------
        reconstruction : ndarray
            Synthesized spectrum
        """

        y = np.asarray(y)

        fixed_params = (self.inst_dict, self.harscale)
        inst_spect = pursuit.gen_inst_spect(
            self.baseshift, self.fsigma, fixed_params,
            self.pexp, self.qexp, self.baseshift + self.m, self.inst_dict.shape[1])
        self.inst_dict, reconstruction, inst_amps_new = \
            stoch_grad(y, self.inst_dict, self.tone_num, self.adam,
                       self.fsigma, self.harscale, self.baseshift,
                       inst_spect, self.pexp, self.qexp)

        self.inst_amps += inst_amps_new
        self.inst_cnt += 1

        self.cnt += 1

        if self.cnt % self.lifetime == 0:
            self.renew_dict(self.lifetime//2, self.inst_num//2)

        return reconstruction

    def renew_dict(self, headstart, newinsts):
        """
        Renew the dictionary.

        Parameters
        ----------
        headstart : int
            Headstart in the lifetime counter (to help new instruments)
        newinsts : int
            Number of instruments to be renewed
        """

        idcs = np.argsort(self.inst_amps
                          / (self.inst_cnt - headstart))[:newinsts]
        for idx in idcs:
            self.inst_amps[idx] = 0
            self.inst_cnt[idx] = 0
            self.inst_dict[:, idx] = gen_random_inst(self.har)
            self.adam.reset(idx)

    def get_dict(self):
        """
        Get the active part of the dictionary.

        Returns
        -------
        inst_dict : ndarray
            Dictionary with `inst_num` columns
        """

        idcs, = np.where(self.inst_cnt >= self.lifetime)
        return self.inst_dict[:, idcs]

def synth_spect(spect, tone_num, inst_dict, fsigma, spectheight, pexp, qexp,
                minfreq, maxfreq, stretch=1):
    """
    Separate and synthesize the spectrograms from the original spectrogram.

    Parameters
    ----------
    spect : array_like
        Original log-frequency spectrogram of the recording
    tone_num : int
        Maximum number of simultaneous tones for each instrument
    inst_dict : ndarray
        Dictionary containing the relative amplitudes of the harmonics
    fsigma : float
        Standard deviation (frequency)
    spectheight : int
        Height of the linear-frequency spectrograms
    pexp : float
        Exponent for the addition of sinusoids
    qexp : float
        Exponent to be applied on the spectrum
    minfreq : float
        Minimum frequency to be represented (included)
        (normalized to the sampling frequency)
    maxfreq : float
        Maximum frequency to be represented (excluded)
        (normalized to the sampling frequency)

    Returns
    -------
    dict_spectrum : ndarray
        Synthesized log-frequency spectrogram with all instruments
    inst_spectrums : list of ndarray
        List of synthesized log-frequency spectrograms for the instruments
    dict_spectrum_lin : ndarray
        Synthesized linear-frequency spectrogram with all instruments
    inst_spectrums_lin : list of ndarray
        List of synthesized linear-frequency spectrograms for the instruments
    """

    spect = np.asarray(spect)
    inst_dict = np.asarray(inst_dict)
    tone_num = np.asarray(tone_num)
    
    minfreq = minfreq * (2 * spectheight)
    maxfreq = maxfreq * (2 * spectheight)

    m = spect.shape[0]
    har = inst_dict.shape[0]
    baseshift = m
    harscale = pursuit.calc_harscale(minfreq, maxfreq, m)
    fixed_params = (inst_dict, harscale)
    inst_spect = pursuit.gen_inst_spect(
        baseshift, fsigma, fixed_params,
        pexp, qexp, baseshift + m, inst_dict.shape[1])

    dict_spectrum = np.zeros(spect.shape)
    dict_spectrum_lin = np.zeros((spectheight, spect.shape[1]))
    inst_dict_size = inst_dict.shape[1]
    inst_spectrums = []
    inst_spectrums_lin = []
    for i in range(inst_dict_size):
        inst_spectrums.append(np.zeros(spect.shape))
        inst_spectrums_lin.append(np.zeros((spectheight, spect.shape[1])))

    numfreqs = spect.shape[0]

    make_bounds, make_inits = make_closures(fsigma)

    for j in range(spect.shape[1]):
        print("reconstruction {} out of {}".format(j, spect.shape[1]))
        y = spect[:, j]
        peaks, reconstruction = \
            pursuit.peak_pursuit(y, tone_num, 1, tone_num*inst_dict.shape[1]*2,
                                 inst_dict.shape[1],
                                 pursuit.inst_shift, pursuit.inst_shift_obj,
                                 pursuit.inst_shift_grad,
                                 make_bounds, make_inits,
                                 fixed_params, pursuit.fft_selector,
                                 (baseshift, inst_spect, qexp), pexp, qexp, beta)
        dict_spectrum[:, j] = reconstruction
        lin_peaks = peaks.copy()
        print("params:")
        print(peaks.params)
        lin_peaks.shifts = (np.exp(peaks.shifts * (np.log(maxfreq / minfreq)
                                                   / numfreqs))
                            * minfreq)
        lin_peaks.params[0, :] = peaks.params[0, :] / stretch

        reconstruction = pursuit.inst_scale(lin_peaks, inst_dict,
                                            pexp, spectheight)
        dict_spectrum_lin[:, j] = reconstruction
        for i in range(inst_dict_size):
            idcs, = np.where(peaks.insts == i)
            reconstruction = pursuit.inst_shift(peaks[idcs], fixed_params,
                                                pexp, m)
            inst_spectrums[i][:, j] = reconstruction
            reconstruction = pursuit.inst_scale(lin_peaks[idcs], inst_dict,
                                                pexp, spectheight)
            inst_spectrums_lin[i][:, j] = reconstruction

    return dict_spectrum, inst_spectrums, dict_spectrum_lin, inst_spectrums_lin

def mask_spectrums(spects, orig_spect):
    """
    Mask the synthesized spectrograms with the original spectrogram.

    Parameters
    ----------
    spects : list of array_like
        List of synthesized spectrograms
    orig_spect : array_like
        Original spectrogram

    Returns
    -------
    spectrums : list of ndarray
        Masked spectrograms
    mask_spect : ndarray
        Array mask
    """

    spects = [np.asarray(s) for s in spects]

    total_spect = sum([s for s in spects])
    mask_spect = orig_spect / (total_spect + 1e-40)
    for i in range(len(spects)):
        spects[i] = spects[i] * mask_spect

    return spects, mask_spect

def learn_spect_dict(spect, fsigma, tone_num, inst_num, pexp, qexp,
                     har, minfreq, maxfreq, runs, lifetime):
    """
    Train the dictionary containing the relative amplitudes of the harmonics.

    Parameters
    ----------
    spect : array_like
        Original log-frequency spectrogram of the recording
    fsigma : float
        Standard deviation (frequency)
    tone_num : int
        Maximum number of simultaneous tones for each instrument
    inst_num : int
        Number of instruments in the dictionary
    pexp : float
        Exponent for the addition of sinusoids
    qexp : float
        Exponent to be applied on the spectrum
    har : int
        Number of harmonics
    minfreq : float
        Minimum frequency in Hz to be represented (included)
    maxfreq : float
        Maximum frequency in Hz to be represented (excluded)
    runs : int
        Number of training iterations to perform
    lifetime : int
        Number of steps after which to renew the dictionary

    Returns
    -------
    inst_dict : ndarray
        Dictionary containing the relative amplitudes of the harmonics
    """

    spect = np.asarray(spect)

    m = spect.shape[0]
    harscale = pursuit.calc_harscale(minfreq, maxfreq, m)
    dl = Learner(fsigma, tone_num, inst_num, har, m, minfreq, maxfreq,
                 lifetime, pexp, qexp)

    errnormsq = []
    signormsq = []

    for k in range(runs):
        print("training iteration {}".format(k))
        idx = np.random.randint(spect.shape[1])
        y = spect[:, idx]
        reconstruction = dl.learn(y)

        errnormsq.append(np.linalg.norm(reconstruction))
        signormsq.append(np.linalg.norm(y))

        if dl.cnt % lifetime == 0:
            inst_dict = dl.get_dict()
            print(inst_dict)

    return dl.get_dict()

def test_learn(fsigma, tone_num, inst_num,
               pexp, qexp, har, m, runs, test_samples, lifetime,
               inst_dict):
    """
    Evaluate the performance of the dictionary learning algorithm via
    artificial spectra.

    Parameters
    ----------
    fsigma : float
        Width of the Gaussians in the log-frequency spectrogram
    tone_num : int
        Maximum number of simultaneous tones for each instrument
    inst_num : int
       Number of instruments in the dictionaries
    pexp : float
        Exponent for the addition of sinusoids
    qexp : float
        Exponent to be applied on the spectrum
    har : int
        Number of harmonics
    m : int
        Height of the log-frequency spectrogram
    runs : int
        Number of training iterations to perform
    test_samples : int
        Number of test spectra to generate
    lifetime : int
        Number of steps after which to renew the dictionary
    inst_dict : array_like
        Dictionary containing the relative amplitudes of the harmonics

    Returns
    -------
    measures : ndarray
        Array containing, in that order, the SDR, SIR, SAR with the
        original dictionary and the SDR, SID, SAR with the trained dictionary
    """

    inst_dict = np.asarray(inst_dict)
    harscale = pursuit.calc_harscale(20, 20480, m)
    fixed_params = (inst_dict, harscale)
    insts = np.repeat(np.arange(inst_num), tone_num)
    print(insts)
    print(inst_dict)

    dl = Learner(fsigma, tone_num, inst_num * 2, har, m, 20, 20480,
                 lifetime, pexp, qexp)

    for k in range(runs):
        print("iteration {}".format(k))
        params = np.vstack((np.ones(tone_num * inst_num) * fsigma,
                            np.zeros(tone_num * inst_num)))
        peaks = pursuit.Peaks(np.random.rand(tone_num * inst_num),
                              np.random.rand(tone_num * inst_num) * 500,
                              params, insts)
        y = np.asarray(pursuit.inst_shift(peaks, fixed_params, pexp, m))
        dl.learn(y)

    test_spect = np.zeros((m, test_samples))
    test_inst_spects = []
    for i in range(inst_num):
        test_inst_spects.append(np.zeros((m, test_samples)))
    for k in range(test_samples):
        params = np.vstack((np.ones(tone_num * inst_num) * fsigma,
                            np.zeros(tone_num * inst_num)))
        peaks = pursuit.Peaks(np.random.rand(tone_num * inst_num),
                              np.random.rand(tone_num * inst_num) * 500,
                              params, insts)
        test_spect[:, k] = pursuit.inst_shift(peaks, fixed_params, pexp, m)
        for i in range(inst_num):
            idcs, = np.where(peaks.insts == i)
            test_inst_spects[i][:, k] = pursuit.inst_shift(
                peaks[idcs], fixed_params, pexp, m)

    inst_dict_learn = dl.get_dict()
    print(inst_dict_learn.shape)
    _, inst_spectrums_orig, _, _ = \
        synth_spect(test_spect, tone_num, inst_dict, fsigma, m,
                    pexp, qexp, 20, 20480)
    _, inst_spectrums_trained, _, _ = \
        synth_spect(test_spect, tone_num, inst_dict_learn, fsigma, m,
                    pexp, qexp, 20, 20480)

    test_spects = np.vstack([np.ravel(spect) for spect in test_inst_spects])
    orig_spects = np.vstack([np.ravel(spect) for spect in inst_spectrums_orig])
    trained_spects = np.vstack([np.ravel(spect)
                                for spect in inst_spectrums_trained])

    return np.hstack((performance.select_perm(
        *performance.measures(orig_spects, test_spects))[1].T,
                      performance.select_perm(
        *performance.measures(trained_spects, test_spects))[1].T))
    
def test_learn_multi(fsigma, tone_num, inst_num,
                     pexp, qexp, har, m, runs, test_samples, lifetime,
                     num_dicts):
    """
    Evaluate the performance of the dictionary learning algorithm via
    artificial spectra.

    Parameters
    ----------
    fsigma : float
        Width of the Gaussians in the log-frequency spectrogram
    tone_num : int
        Maximum number of simultaneous tones for each instrument
    inst_num : int
       Number of instruments in the dictionaries
    pexp : float
        Exponent for the addition of sinusoids
    qexp : float
        Exponent to be applied on the spectrum
    har : int
        Number of harmonics
    m : int
        Height of the log-frequency spectrogram
    runs : int
        Number of training iterations to perform
    test_samples : int
        Number of test spectra to generate
    lifetime : int
        Number of steps after which to renew the dictionary
    num_dicts : int
        Number of different dictionaries to generate and train

    Returns
    -------
    measures : ndarray
        Array containing, in the rows, the SDR, SIR, SAR with the
        original dictionary and the SDR, SID, SAR with the trained dictionary
    """

    measures = []
    for i in range(num_dicts):
        np.random.seed(i)
        inst_dict = gen_random_inst_dict(har, inst_num)
        measures.append(test_learn(fsigma, tone_num, inst_num, pexp,
                                   qexp, har, m, runs, test_samples,
                                   lifetime, inst_dict))

    measures = np.vstack(measures)
    print(measures)
    print("Averages:")
    print(np.mean(measures, axis=0))
    print("Standard deviations:")
    print(np.std(measures, axis=0, ddof=1))
    
    return measures

if __name__ == '__main__':
    pexp = 1
    qexp = 1/2

    np.savetxt('artificial.txt', test_learn_multi(6/np.pi, 1, 2, pexp, qexp,
                                                  25, 1024, 10000, 10000,
                                                  500, 10))
