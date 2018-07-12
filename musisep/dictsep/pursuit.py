#!python3

"""
Module for the sparse pursuit algorithm and its helper functions.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.optimize
import scipy.linalg
import scipy.fftpack

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import exptool

def calc_harscale(minfreq, maxfreq, numfreqs):
    """
    Calculate the scaling factor of the frequency axis for the log-frequency
    spectrogram.

    Parameters
    ----------
    minfreq : float
        Minimum frequency to be represented (included)
    maxfreq : float
        Maximum frequency to be represented (excluded)
    numfreqs : int
        Intended height of the spectrogram

    Returns
    -------
    harscale : float
        Scaling factor
    """

    return numfreqs / np.log(maxfreq / minfreq)

class Peaks:
    """
    Object to represent the parameters for the peaks in the spectrogram.

    Parameters
    ----------
    amps : array_like
        Amplitudes
    shifts : array_like
        Fundamental frequencies
    sigmas : array_like
        Standard deviations (frequency)
    spreads : array_like
        Inharmonicities
    insts : array_like
        Instrument numbers
    """

    def __init__(self, amps, shifts, sigmas, spreads, insts):
        self.amps = np.asarray(amps, dtype='double')
        self.shifts = np.asarray(shifts, dtype='double')
        self.sigmas = np.asarray(sigmas, dtype='double')
        self.spreads = np.asarray(spreads, dtype='double')
        self.insts = np.asarray(insts, dtype='int')

    @classmethod
    def empty(cls):
        """
        Construct an empty `Peaks` object.

        Returns
        -------
        Peaks
            A `Peaks` object with zero peaks
        """

        return cls(np.zeros(0), np.zeros(0), np.zeros(0),
                   np.zeros(0), np.zeros(0))

    @classmethod
    def from_array(cls, array, insts):
        """
        Construct a `Peaks` object from an array.

        Parameters
        ----------
        array : array_like
            Array that contains, in consecutive order, the amplitudes,
            the fundamental frequencies, the standard deviations, and
            the inharmoniticies
        insts : array_like
            Instrument numbers
        """

        array = np.asarray(array)
        insts = np.asarray(insts)

        return cls(*array.reshape((4, insts.size)), insts)

    def __len__(self):
        return self.insts.size

    def __getitem__(self, key):
        new = Peaks(self.amps[key], self.shifts[key], self.sigmas[key],
                    self.spreads[key], self.insts[key])
        return new

    def get_params(self):
        """
        Returns
        -------
        amps : array_like
            Amplitudes
        shifts : array_like
            Fundamental frequencies
        sigmas : array_like
            Standard deviations (frequency)
        spreads : array_like
            Inharmonicities
        insts : array_like
            Instrument numbers
        """

        return self.amps, self.shifts, self.sigmas, self.spreads, self.insts

    def get_array(self):
        """
        Returns
        -------
        array_like
            Array that contains, in consecutive order, the amplitudes,
            the fundamental frequencies, the standard deviations, and
            the inharmoniticies        
        """

        return np.concatenate((self.amps, self.shifts,
                               self.sigmas, self.spreads))

    def copy(self):
        """
        Returns
        -------
        Peaks
            Copy of the contained peak parameters
        """

        new = Peaks(self.amps.copy(), self.shifts.copy(), self.sigmas.copy(),
                    self.spreads.copy(), self.insts.copy())
        return new
    
    def merge(self, new):
        """
        Merge the `Peaks` object with another `Peaks` object contained in
        `new` by concatenating the parameters.

        Parameters
        ----------
        new : Peaks
            Object to merge with
        """

        self.amps = np.concatenate((self.amps, new.amps))
        self.shifts = np.concatenate((self.shifts, new.shifts))
        self.sigmas = np.concatenate((self.sigmas, new.sigmas))
        self.spreads = np.concatenate((self.spreads, new.spreads))
        self.insts = np.concatenate((self.insts, new.insts))

def inst_shift(peaks, inst_dict, harscale, pexp, m, n):
    """
    Synthesize the log-frequency spectrum.

    Parameters
    ----------
    peaks : Peaks
        Peak parameters
    inst_dict : array_like
        Dictionary containing the relative amplitudes of the harmonics
    harscale : float
        Scaling factor
    pexp : float
        Exponent for the addition of sinusoids
    m : int
        Height of the spectrogram
    n : int
        Number of instruments

    Returns
    -------
    ndarray
        Log-frequency spectrum
    """

    inst_dict = np.asarray(inst_dict)
    
    reconstruction = exptool.inst_shift(*peaks.get_params(),
                                        inst_dict, harscale, pexp, m, n) 
    return np.asarray(reconstruction) ** (1/pexp)

def inst_scale(peaks, inst_dict, pexp, m, n):
    """
    Synthesize the linear-frequency spectrum.

    Parameters
    ----------
    peaks : Peaks
        Peak parameters
    inst_dict : array_like
        Dictionary containing the relative amplitudes of the harmonics
    pexp : float
        Exponent for the addition of sinusoids
    m : int
        Height of the spectrogram
    n : int
        Number of instruments

    Returns
    -------
    ndarray
        Linear-frequency spectrum
    """

    reconstruction = exptool.inst_scale(*peaks.get_params(),
                                        inst_dict, pexp, m, n) 
    return np.asarray(reconstruction) ** (1/pexp)

def inst_shift_obj(peak_array, insts, inst_dict, harscale, pexp, qexp,
                   m, n, y):
    """
    Least-squares objective function for the log-frequency spectrum.

    Parameters
    ----------
    peak_array : array_like
        Peak parameters in array form
    insts : array_like
        Instrument numbers
    inst_dict : array_like
        Dictionary containing the relative amplitudes of the harmonics
    harscale : float
        Scaling factor
    pexp : float
        Exponent for the addition of sinusoids
    qexp : float
        Exponent to be applied on the spectrum
    m : int
        Height of the spectrogram
    n : int
        Number of instruments
    y : array_like
        Spectrum to compare with

    Returns
    -------
    obj : float
        Least-squares error
    """

    y = np.asarray(y)

    reconstruction = inst_shift(Peaks.from_array(peak_array, insts),
                                inst_dict, harscale, pexp, m, n)

    return np.sum(np.square(reconstruction**qexp - y**qexp)) / 2

def inst_shift_grad(peak_array, insts, inst_dict, harscale, pexp, qexp,
                    m, n, y):
    """
    Least-squares gradient function for the log-frequency spectrum
    w.r.t. the parameters.

    Parameters
    ----------
    peak_array : array_like
        Peak parameters in array form
    insts : array_like
        Instrument numbers
    inst_dict : array_like
        Dictionary containing the relative amplitudes of the harmonics
    harscale : float
        Scaling factor
    pexp : float
        Exponent for the addition of sinusoids
    qexp : float
        Exponent to be applied on the spectrum
    m : int
        Height of the spectrogram
    n : int
        Number of instruments
    y : array_like
        Spectrum to compare with

    Returns
    -------
    grad : ndarray
        Least-squares gradient
    """

    peaks = Peaks.from_array(peak_array, insts)
    reconstruction = inst_shift(peaks, inst_dict, harscale, pexp, m, n)
    reconstruction = np.asarray(reconstruction)
    expvec = ((reconstruction**qexp - y**qexp)
              * (reconstruction+1e-40) ** (qexp - pexp) * qexp)
    grad = exptool.inst_shift_grad(expvec, *peaks.get_params(),
                                   inst_dict, harscale, pexp-1, m, n)
    grad = np.asarray(grad)

    return grad

def inst_shift_dict_grad(peak_array, insts, inst_dict, harscale, pexp, qexp,
                         m, n, y):
    """
    Least-squares gradient function for the log-frequency spectrum
    w.r.t. the dictionary.

    Parameters
    ----------
    peak_array : array_like
        Peak parameters in array form
    insts : array_like
        Instrument numbers
    inst_dict : array_like
        Dictionary containing the relative amplitudes of the harmonics
    harscale : float
        Scaling factor
    pexp : float
        Exponent for the addition of sinusoids
    qexp : float
        Exponent to be applied on the spectrum
    m : int
        Height of the spectrogram
    n : int
        Number of instruments
    y : array_like
        Spectrum to compare with

    Returns
    -------
    grad : ndarray
        Least-squares gradient w.r.t. the dictionary
    """

    y = np.asarray(y)
    peaks = Peaks.from_array(peak_array, insts)
    reconstruction = inst_shift(peaks, inst_dict, harscale, pexp, m, n)
    expvec = ((reconstruction**qexp - y**qexp)
              * (reconstruction+1e-40) ** (qexp - pexp) * qexp)
    grad = exptool.inst_shift_dict_grad(expvec, *peaks.get_params(),
                                        inst_dict, harscale, pexp-1, m, n)
    grad = np.asarray(grad)

    return grad

def make_bounds(fsigma, length):
    """
    Compute sensible bounds for the peak parameters.

    Parameters
    ----------
    fsigma : float
        Standard deviation (frequency)
    length : int
        Number of instruments

    Returns
    -------
    bounds : list of tuple
        Bounds for the optimizer
    """    
    
    return ([(0, None)] * length
            + [(None, None)] * length
            + [(fsigma*0.8, fsigma*1.5)] * length
            + [(0, 2e-3)] * length)

def max_selector(y, prenum, n):
    """
    Callback selector to find peaks based on the local maxima which are
    dominant in a discrete interval, viewed from its midpoint.

    Parameters
    ----------
    y : array_like
        Spectrum
    prenum : int
        Number of peaks to consider
    n : int
        Length of the interval

    Returns
    -------
    amps : array_like
        Amplitudes
    shifts : array_like
        Frequencies
    insts : array_like
        Instrument numbers (always 0)
    """

    y = np.asarray(y)

    A = scipy.linalg.toeplitz(y[:n], y)
    A = np.roll(A, -(n-1)//2, axis=1)
    keys, = np.where(np.logical_and(np.amax(A, axis=0) == y,
                                    y > 0))
    idcs = np.argsort(-y[keys])[:prenum]

    return y[keys[idcs]], keys[idcs], np.zeros(idcs.size)

def fft_selector(y, prenum, baseshift, inst_spect, qexp):
    """
    Callback selector to find fundamental frequencies based on the correlation
    of the spectrum with the instrument spectra.

    Parameters
    ----------
    y : array_like
        Spectrum
    prenum : int
        Number of peaks to consider
    baseshift : int
        Length to add to the spectrum in order to avoid circular convolution
    inst_spect : array_like
        Spectra of the instruments, in the columns
    qexp : float
        Exponent to be applied on the spectrum

    Returns
    -------
    amps : array_like
        Amplitudes
    shifts : array_like
        Fundamental frequencies
    insts : array_like
        Instrument numbers
    """

    inst_spect = np.asarray(inst_spect)
    inst_spect_norms = np.linalg.norm(inst_spect, 2, axis=0)
    inst_spect = inst_spect / inst_spect_norms
    inst_spect_freq = scipy.fftpack.fft(inst_spect, axis=0)
    yaug = np.concatenate((np.zeros(baseshift), y))
    yfreq = scipy.fftpack.fft(yaug)
    corrs = scipy.fftpack.ifft(np.conj(inst_spect_freq)
                               * yfreq.reshape([-1, 1]), axis=0)
    corrs = np.real(corrs[:y.size, :])
    
    keys = np.argsort(-corrs, axis=None)[:prenum]
    keys, insts = np.unravel_index(keys, corrs.shape)

    amps = corrs[keys, insts] / inst_spect_norms[insts]

    idcs, = np.where(amps > 0)

    return amps[idcs], keys[idcs], insts[idcs]

def peak_pursuit(y, num, prenum, runs, inst_dict, fsigma, harscale,
                 selector, selector_args, pexp, qexp, beta=1, init=None):
    """
    Sparse pursuit algorithm for the identification of peaks in a spectrum.

    Parameters
    ----------
    y : array_like
        Spectrum
    num : int
        Maximum number of peaks
    prenum : int
        Number of new peaks to consider per step
    inst_dict : ndarray
        Dictionary containing the relative amplitudes of the harmonics
    fsigma : float
        Standard deviation (frequency)
    harscale : float
        Scaling factor
    selector : function
        Callback selector accepting `y` and `prenum` as arguments
    selector_args : sequence
        Extra arguments to pass to the selector
    pexp : float
        Exponent for the addition of sinusoids
    qexp : float
        Exponent to be applied on the spectrum
    beta : float
        Residual reduction factor
    init : Peaks
        Initial value for the peaks

    Returns
    -------
    peaks : Peaks
        Identified peaks
    reconstruction : ndarray
        Synthesized spectrum
    """

    y = np.asarray(y)
    inst_dict = np.asarray(inst_dict)

    m = y.size
    n = inst_dict.shape[1]

    if init is None:
        reconstruction = None
        peaks = Peaks.empty()
    else:
        peaks = init

    reconstruction = inst_shift(peaks, inst_dict, harscale, pexp,
                                m, len(peaks))
    r = y**qexp - reconstruction**qexp

    for i in range(runs):
        amps, keys, new_insts = selector(r, prenum, *selector_args)

        if keys.size == 0:
            print("break in iteration {} [empty selection]".format(i))
            break

        old_peaks = peaks.copy()

        if (amps < 0).any():
            print("amps: {}".format(amps))
        
        new_peaks = Peaks(amps ** (1/qexp), keys,
                          np.repeat(fsigma, keys.size),
                          np.zeros(keys.size), new_insts)
        peaks.merge(new_peaks)

        bounds = make_bounds(fsigma, len(peaks))

        res = scipy.optimize.fmin_l_bfgs_b(
            inst_shift_obj, peaks.get_array(),
            args=(peaks.insts, inst_dict, harscale, pexp, qexp, m,
                  len(peaks), y),
            bounds=bounds,
            fprime=inst_shift_grad,
            #factr=1e3,
            disp=0)

        peaks = Peaks.from_array(res[0], peaks.insts)

        reconstruction_new = inst_shift(peaks, inst_dict, harscale, pexp,
                                        m, len(peaks))

        if len(peaks) > num:
            idcs = np.argsort(-peaks.amps)[:num]
            peaks = peaks[idcs]

        bounds = make_bounds(fsigma, len(peaks))

        res = scipy.optimize.fmin_l_bfgs_b(
            inst_shift_obj, peaks.get_array(),
            args=(peaks.insts, inst_dict, harscale, pexp, qexp, m,
                  len(peaks), y),
            bounds=bounds,
            fprime=inst_shift_grad,
            #factr=1e1,
            disp=0)

        peaks = Peaks.from_array(res[0], peaks.insts)
        reconstruction_new = inst_shift(peaks, inst_dict, harscale, pexp, m,
                                        len(peaks))

        if (np.linalg.norm(reconstruction_new**qexp - y**qexp)
            < np.linalg.norm(reconstruction**qexp - y**qexp) * beta):
            reconstruction = reconstruction_new
            r = y**qexp - reconstruction_new**qexp
        else:
            peaks = old_peaks
            print("break in iteration {}".format(i))
            break

    return peaks, reconstruction

def gen_inst_spect(baseshift, fsigma, inst_dict, harscale, pexp, qexp, m):
    """
    Generate an instrument log-frequency spectrum.

    Parameters
    ----------
    baseshift : int
        Length to add to the spectrum in order to avoid circular convolution
    fsigma : float
        Standard deviation (frequency)
    inst_dict : ndarray
        Dictionary containing the relative amplitudes of the harmonics
    harscale : float
        Scaling factor
    pexp : float
        Exponent for the addition of sinusoids
    qexp : float
        Exponent to be applied on the spectrum
    m : int
        Height of the spectrogram

    Returns
    -------
    inst_spect : ndarray
        Spectra of the instruments, in the columns    
    """

    n = inst_dict.shape[1]
    inst_spect = np.zeros((m, n), order='F')
    for i in range(n):
        peaks = Peaks([1], [baseshift], [fsigma], [0], [i])
        inst_spect[:, i] = inst_shift(peaks, inst_dict, harscale,
                                      pexp, m, 1) ** qexp

    return inst_spect
