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
from ..audio import spect

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
    params : array_like
        Extra parameters (in the rows)
    insts : array_like
        Instrument numbers
    """

    def __init__(self, amps, shifts, params, insts):
        self.amps = np.asarray(amps, dtype='double')
        self.shifts = np.asarray(shifts, dtype='double')
        self.params = np.asarray(params, dtype='double')
        self.insts = np.asarray(insts, dtype='int')
        self.paramlen = len(self.params)

    @classmethod
    def empty(cls, params):
        """
        Construct an empty `Peaks` object.

        Returns
        -------
        Peaks
            A `Peaks` object with zero peaks
        """

        return cls(np.zeros(0), np.zeros(0), np.asarray(params),
                   np.zeros(0))

    @classmethod
    def from_array(cls, array, insts, paramlen):
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
        paramlen : int
            Number of extra parameters
        """

        array = np.asarray(array)
        insts = np.asarray(insts)

        array = array.reshape((paramlen + 2, insts.size))
        return cls(*array[:2], array[2:], insts)

    def __len__(self):
        return self.insts.size

    def __getitem__(self, key):
        new = Peaks(self.amps[key], self.shifts[key], self.params[:, key],
                    self.insts[key])
        return new

    def get_params(self):
        """
        Returns
        -------
        amps : ndarray
            Amplitudes
        shifts : ndarray
            Fundamental frequencies
        params : ndarray
            Extra parameters
        insts : ndarray
            Instrument numbers
        """

        return self.amps, self.shifts, self.params, self.insts

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
                               np.ravel(self.params)))

    def copy(self):
        """
        Returns
        -------
        Peaks
            Copy of the contained peak parameters
        """

        new = Peaks(self.amps.copy(), self.shifts.copy(), self.params.copy(),
                    self.insts.copy())
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
        self.params = np.concatenate((self.params, new.params), axis=1)
        self.insts = np.concatenate((self.insts, new.insts))

def inst_shift(peaks, fixed_params, pexp, m):
    """
    Synthesize the log-frequency spectrum.

    Parameters
    ----------
    peaks : Peaks
        Peak parameters
    fixed_params : sequence
        Extra fixed parameters for the synthesizer
    pexp : float
        Exponent for the addition of sinusoids
    m : int
        Height of the spectrogram

    Returns
    -------
    ndarray
        Log-frequency spectrum
    """

    inst_dict, harscale = fixed_params

    amps, shifts, params, insts = peaks.get_params()
    sigmas, spreads = params
    reconstruction = exptool.inst_shift(amps, shifts, sigmas, spreads, insts,
                                        inst_dict, harscale, pexp, m)
    reconstruction = np.maximum(0, reconstruction)

    return np.asarray(reconstruction) ** (1/pexp)

def inst_scale(peaks, inst_dict, pexp, m):
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

    Returns
    -------
    ndarray
        Linear-frequency spectrum
    """

    amps, shifts, params, insts = peaks.get_params()
    sigmas, spreads = params
    reconstruction = exptool.inst_scale(amps, shifts, sigmas, spreads, insts,
                                        inst_dict, pexp, m)
    reconstruction = np.maximum(0, reconstruction)

    return np.asarray(reconstruction) ** (1/pexp)

def inst_shift_obj(peak_array, insts, fixed_params, pexp, qexp,
                   m, y):
    """
    Least-squares objective function for the log-frequency spectrum.

    Parameters
    ----------
    peak_array : array_like
        Peak parameters in array form
    insts : array_like
        Instrument numbers
    fixed_params : sequence
        Extra fixed parameters for the synthesizer
    pexp : float
        Exponent for the addition of sinusoids
    qexp : float
        Exponent to be applied on the spectrum
    m : int
        Height of the spectrogram
    y : array_like
        Spectrum to compare with

    Returns
    -------
    obj : float
        Least-squares error
    """

    y = np.asarray(y)

    reconstruction = inst_shift(Peaks.from_array(peak_array, insts, 2),
                                fixed_params, pexp, m)
    
    return np.sum(np.square((reconstruction+1e-7)**qexp - (y+1e-7)**qexp)) / 2

def inst_shift_grad(peak_array, insts, fixed_params, pexp, qexp, m, y):
    """
    Least-squares gradient function for the log-frequency spectrum
    w.r.t. the parameters.

    Parameters
    ----------
    peak_array : array_like
        Peak parameters in array form
    insts : array_like
        Instrument numbers
    fixed_params : sequence
        Extra fixed parameters for the synthesizer
    pexp : float
        Exponent for the addition of sinusoids
    qexp : float
        Exponent to be applied on the spectrum
    m : int
        Height of the spectrogram
    y : array_like
        Spectrum to compare with

    Returns
    -------
    grad : ndarray
        Least-squares gradient
    """

    inst_dict, harscale = fixed_params

    peaks = Peaks.from_array(peak_array, insts, 2)
    reconstruction = inst_shift(peaks, fixed_params, pexp, m)
    amps, shifts, params, insts = peaks.get_params()
    sigmas, spreads = params
    expvec = (((reconstruction+1e-7)**qexp - (y+1e-7)**qexp)
              * (reconstruction+1e-7) ** (qexp - pexp) * qexp)
    grad = exptool.inst_shift_grad(expvec, amps, shifts, sigmas, spreads, insts,
                                   inst_dict, harscale, pexp-1, m)
    grad = np.asarray(grad)

    return grad

def inst_shift_dict_grad(peak_array, insts, fixed_params, pexp, qexp, m, y):
    """
    Least-squares gradient function for the log-frequency spectrum
    w.r.t. the dictionary.

    Parameters
    ----------
    peak_array : array_like
        Peak parameters in array form
    insts : array_like
        Instrument numbers
    fixed_params : sequence
        Extra fixed parameters for the synthesizer
    pexp : float
        Exponent for the addition of sinusoids
    qexp : float
        Exponent to be applied on the spectrum
    m : int
        Height of the spectrogram
    y : array_like
        Spectrum to compare with

    Returns
    -------
    grad : ndarray
        Least-squares gradient w.r.t. the dictionary
    """

    inst_dict, harscale = fixed_params

    y = np.asarray(y)
    peaks = Peaks.from_array(peak_array, insts, 2)
    m = y.size
    reconstruction = inst_shift(peaks, fixed_params, pexp, m)
    amps, shifts, params, insts = peaks.get_params()
    sigmas, spreads = params
    expvec = (((reconstruction+1e-7)**qexp - (y+1e-7)**qexp)
              * (reconstruction+1e-7) ** (qexp - pexp) * qexp)
    grad = exptool.inst_shift_dict_grad(expvec, amps, shifts, sigmas, spreads,
                                        insts, inst_dict, harscale, pexp-1, m)
    grad = np.asarray(grad)

    return grad

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

def peak_pursuit(y, nums, prenum, runs, n, inst_shift, inst_shift_obj,
                 inst_shift_grad, make_bounds, make_inits, fixed_params,
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
        Number of new peaks to consider per iteration
    runs : int
        Maximum number of training iterations
    n : int
        Number of patterns/instruments
    inst_shift : callable (peaks, fixed_params, pexp, m, n)
        Synthesizing function
    inst_shift_obj : callable (peak_array, insts, fixed_params, pexp, qexp, m, n, y)
        Synthesizing function objective
    inst_shift_grad : callable (peak_array, insts, fixed_params, pexp, qexp, m, n, y)
        Synthesizing function gradient
    make_bounds : lambda (length)
        Lambda that gives the bounds for `length` peaks
    make_inits : lambda (length)
        Lambda that gives the initial values `length` peaks
    fixed_params : sequence
        Extra fixed parameters for the synthesizer
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
    nums = np.asarray(nums)
    
    m = y.size

    if init is None:
        reconstruction = None
        peaks = Peaks.empty(make_inits(0))
    else:
        peaks = init

    reconstruction = inst_shift(peaks, fixed_params, pexp, m)

    r = y**qexp - reconstruction**qexp

    for i in range(runs):
        amps, keys, new_insts = selector(r, prenum, *selector_args)

        if keys.size == 0:
            print("break in iteration {} [empty selection]".format(i))
            break

        old_peaks = peaks.copy()

        new_peaks = Peaks(amps ** (1/qexp), keys,
                          make_inits(keys.size), new_insts)
        peaks.merge(new_peaks)

        bounds = make_bounds(len(peaks))

        res = scipy.optimize.fmin_l_bfgs_b(
            inst_shift_obj, peaks.get_array(),
            args=(peaks.insts, fixed_params, pexp, qexp, m, y),
            bounds=bounds,
            fprime=inst_shift_grad,
            #factr=1e3,
            disp=0)

        peaks = Peaks.from_array(res[0], peaks.insts, len(make_inits(0)))

        reconstruction_new = inst_shift(peaks, fixed_params, pexp, m)

        keep_peaks = Peaks.empty(make_inits(0))
        for j in range(n):
            inst_idcs, = np.where(peaks.insts == j)
            if inst_idcs.size > nums:
                idcs = np.argsort(-peaks.amps[inst_idcs])[:nums]
                keep_peaks.merge(peaks[inst_idcs[idcs]])
            else:
                keep_peaks.merge(peaks[inst_idcs])
        peaks = keep_peaks

        bounds = make_bounds(len(peaks))

        res = scipy.optimize.fmin_l_bfgs_b(
            inst_shift_obj, peaks.get_array(),
            args=(peaks.insts, fixed_params, pexp, qexp, m, y),
            bounds=bounds,
            fprime=inst_shift_grad,
            #factr=1e1,
            disp=0)

        peaks = Peaks.from_array(res[0], peaks.insts, len(make_inits(0)))
        reconstruction_new = inst_shift(peaks, fixed_params, pexp, m)

        if (np.linalg.norm(reconstruction_new**qexp - y**qexp)
            < np.linalg.norm(reconstruction**qexp - y**qexp) * beta):
            reconstruction = reconstruction_new
            r = y**qexp - reconstruction_new**qexp
        else:
            peaks = old_peaks
            print("break in iteration {}".format(i))
            break

    return peaks, reconstruction

def gen_inst_spect(baseshift, fsigma, fixed_params, pexp, qexp, m, n):
    """
    Generate an instrument log-frequency spectrum.

    Parameters
    ----------
    baseshift : int
        Length to add to the spectrum in order to avoid circular convolution
    fsigma : float
        Standard deviation (frequency)
    fixed_params : sequence
        Extra fixed parameters for the synthesizer
    pexp : float
        Exponent for the addition of sinusoids
    qexp : float
        Exponent to be applied on the spectrum
    m : int
        Height of the spectrogram
    n : int
        Number of patterns/instruments

    Returns
    -------
    inst_spect : ndarray
        Spectra of the instruments, in the columns    
    """

    inst_spect = np.zeros((m, n), order='F')
    for i in range(n):
        peaks = Peaks([1], [baseshift], [[fsigma], [0]], [i])
        inst_spect[:, i] = inst_shift(peaks, fixed_params,
                                      pexp, m) ** qexp

    return inst_spect

def test_pattern_gen(seed, scaling):
    """
    Generate the parameters for a random pattern.

    Parameters
    ----------
    seed : int
        Random seed
    scaling : float
        Scaling of the axis

    Returns
    -------
    amps : ndarray
        Amplitudes of the peaks
    shifts : ndarray
        Positions of the peaks on the axis
    sigmas : ndarray
        Standard deviations of the peaks
    """

    n = 5
    np.random.seed(seed)
    shifts = np.random.rand(2, n) * scaling
    sigmas = np.random.rand(2, n)/20 * scaling
    amps = np.random.rand(2, n)

    return amps, shifts, sigmas

def test_pattern_comp(x, amps, shifts, sigmas):
    """
    Evaluate a test pattern.

    Parameters
    ----------
    x : array_like
        Positions to evaluate the pattern
    amps : array_like
        Amplitudes of the peaks
    shifts : array_like
        Positions of the peaks on the axis
    sigmas : array_like
        Standard deviations of the peaks

    Returns
    -------
    y : ndarray
        Evaluation of the test pattern
    """

    x = np.asarray(x)
    amps = np.asarray(amps)
    shifts = np.asarray(shifts)
    sigmas = np.asarray(sigmas)

    n = amps.shape[1]
    y = np.zeros((amps.shape[0], x.size))

    for i in range(n):
        y += (spect.gauss(x - shifts[:, i:i+1], sigmas[:, i:i+1], False)
              * amps[:, i:i+1])

    return y

def test_pattern(peaks, fixed_params, pexp, m):
    """
    Evaluate a test pattern.

    Parameters
    ----------
    peaks : Peaks
        Continuous parameters for the peaks
    fixed_params : sequence
        Extra fixed parameters
    pexp : float
        (ignored)
    m : int
        (ignored)

    Returns
    -------
    y : ndarray
        Sampled test pattern
    """

    x, pat_amps, pat_shifts, pat_sigmas = fixed_params
    y = np.sum(
        test_pattern_comp(x,
                          peaks.amps[:, np.newaxis] * pat_amps[peaks.insts, :],
                          peaks.shifts[:, np.newaxis]
                          + pat_shifts[peaks.insts, :],
                          pat_sigmas[peaks.insts, :]), axis=0)

    return y

def test_pattern_obj(peak_array, insts, fixed_params, pexp, qexp, m, y):
    """
    Loss objective for a test pattern.

    Parameters
    ----------
    peak_array : array_like
        Array of all the continuous parameters
    insts : array_like
        Number of the instruments/patterns
    fixed_params : sequence
        Extra fixed parameters
    pexp : float
        (ignored)
    qexp : float
        (ignored)
    m : int
        (ignored)
    y : array_like
        Evaluation of the test pattern

    Returns
    -------
    obj : float
        Least-squares error
    """

    reconstruction = test_pattern(Peaks.from_array(peak_array, insts, 0),
                                  fixed_params, pexp, m)

    loss = np.sum(np.square(reconstruction - y)) / 2

    print("peak_array: {}".format(peak_array))
    print("loss: {}".format(loss))
    return loss

def test_pattern_grad_helper(x, r, amps, shifts,
                             pat_amps, pat_shifts, pat_sigmas):
    """
    Helper function for the computation of the gradient of the test pattern.

    Parameters
    ----------
    x : array_like
        Positions where pattern was evaluated
    r : array_like
        Residual of the pattern
    amps : array_like
        Amplitudes of the patterns
    shifts : array_like
        Shifts of the peaks
    pat_amps : array_like
        Amplitudes of the peaks for each pattern
    pat_shifts : array_like
        Positions of the peaks on the axis for each pattern
    pat_sigmas : array_like
        Standard deviations of the peaks for each pattern

    Returns
    -------
    grad : ndarray
        Gradient for the test pattern
    """

    x = np.asarray(x)
    r = np.asarray(r)
    amps = np.asarray(amps)
    shifts = np.asarray(shifts)
    pat_amps = np.asarray(pat_amps)
    pat_shifts = np.asarray(pat_shifts)
    pat_sigmas = np.asarray(pat_sigmas)
    
    n = amps.size
    m = pat_amps.shape[1]
    grad = np.zeros(2*n)

    for i in range(n):
        for j in range(m):
            grad[i] += r.dot(
                pat_amps[i,j] * spect.gauss(x - shifts[i, np.newaxis]
                                            - pat_shifts[i, j],
                                            pat_sigmas[i, j], False))
            grad[i+n] += r.dot(
                amps[i, np.newaxis] * pat_amps[i, j]
                * (x - shifts[i, np.newaxis] - pat_shifts[i,j])
                / pat_sigmas[i, j]**2 * spect.gauss(x - shifts[i, np.newaxis]
                                                    - pat_shifts[i, j],
                                                    pat_sigmas[i, j], False))

    return grad

def test_pattern_grad(peak_array, insts, fixed_params, pexp, qexp,
                      m, y):
    """
    Gradient for a test pattern.

    Parameters
    ----------
    peak_array : array_like
        Array of all the continuous parameters
    insts : array_like
        Number of the instruments/patterns
    fixed_params : sequence
        Extra fixed parameters
    pexp : float
        (ignored)
    qexp : float
        (ignored)
    m : int
        (ignored)
    y : array_like
        Evaluation of the test pattern

    Returns
    -------
    grad : ndarray
        Gradient for the test pattern
    """

    peak_array = np.asarray(peak_array)
    insts = np.asarray(insts)
    y = np.asarray(y)

    x, pat_amps, pat_shifts, pat_sigmas = fixed_params

    peaks = Peaks.from_array(peak_array, insts, 0)
    reconstruction = test_pattern(peaks, fixed_params, pexp, m)
    amps, shifts, params, insts = peaks.get_params()
    expvec = reconstruction - y
    grad = test_pattern_grad_helper(x, expvec, amps, shifts, pat_amps[insts, :],
                                    pat_shifts[insts, :], pat_sigmas[insts, :])

    return grad

def test_pursuit():
    "Testing the pursuit algorithm on a generic example."

    n = 20
    x = np.arange(n)
    x2 = np.linspace(0, n, 1000)

    make_bounds = lambda length : [(0, None)] * length + [(None, None)] * length
    make_inits = lambda length : np.zeros([0, length])

    pat_amps, pat_shifts, pat_sigmas = test_pattern_gen(3, 50 * n / 100)
    amps = np.random.rand(2)
    shifts = np.asarray([10.3*n/100, 40.8*n/100])
    print(shifts)
    baseshift = x.size
    y = np.sum(test_pattern_comp(x, amps[:,np.newaxis]*pat_amps,
                                 shifts[:,np.newaxis]+pat_shifts,
                                 pat_sigmas),
               axis=0)
    inst_spect = np.vstack((np.zeros((baseshift, 2)),
                            test_pattern_comp(x, pat_amps, pat_shifts,
                                              pat_sigmas).T))

    fixed_params = (x, pat_amps, pat_shifts, pat_sigmas)
    peaks, reconstruction = peak_pursuit(
        y, 1, 1, 4, 2, test_pattern, test_pattern_obj, test_pattern_grad,
        make_bounds, make_inits, fixed_params,
        fft_selector, (baseshift, inst_spect, 1), 1, 1, 0.9)

    inst_spect_hd = test_pattern_comp(x2, pat_amps, pat_shifts, pat_sigmas)
    inst_spect_hd_appl = test_pattern_comp(x2, pat_amps*amps[:, np.newaxis],
                                           pat_shifts+shifts[:, np.newaxis],
                                           pat_sigmas)

    np.savetxt('test_pursuit_sample.dat', np.vstack((x, y)).T)
    np.savetxt('test_pursuit_patterns.dat', np.vstack((x2, inst_spect_hd)).T)
    np.savetxt('test_pursuit_patterns_appl.dat',
               np.vstack((x2, inst_spect_hd_appl)).T)
    np.savetxt('test_pursuit_real.dat',
               np.vstack((x2, np.sum(inst_spect_hd_appl, axis=0))).T)
    
    print(10 * np.log10(np.sum(np.square(y))
                        / np.sum(np.square(reconstruction - y))))
    plt.plot(x, y, x, reconstruction)
    plt.savefig('test_pursuit.png')

if __name__ == '__main__':
    test_pursuit()
