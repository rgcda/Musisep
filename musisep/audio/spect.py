#!python3

"""
Module to generate spectograms, save them as images and resynthesize audio.
When invoked, a side-by-side comparison of the spectrograms from the
different methods is performed.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import scipy.fftpack as fftpack
import pyfftw
import pyfftw.interfaces.scipy_fftpack as fftpack
import scipy.misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from . import wav
from . import specttool
from ..dictsep import pursuit

def gauss(x, stdev, normalize=True):
    """
    Generate an l1-normalized Gaussian window/kernel with mean 0.

    Parameters
    ----------
    x : array_like
        Points to evaluate the Gaussian
    stdev : float
        Standard deviation
    normalize : bool
        Whether to l1-normalize the Gaussian

    Returns
    -------
    window : ndarray
        Gaussian window/kernel
    """

    x = np.asarray(x)

    window = np.exp(- np.square(x / stdev) / 2)
    if normalize:
        return window / np.sum(window)
    else:
        return window

def spectwrite(filename, spectrogram, color=True):
    """
    Save a spectrogram as an image.  The data is normalized to dynamic range
    of 100 dB, and a logarithmic Viridis color scale is used.

    Parameters
    ----------
    filename : string
        Name of the image file
    spectrogram : array_like
        Spectrogram
    color : bool
        Whether to make a color plot
    """

    spectrogram = np.asarray(spectrogram)
    print("spect max: {}".format(np.amax(spectrogram)))
    spectrogram = spectrogram / np.amax(spectrogram)

    logspect = np.log10((1e-5 + spectrogram) / (1e-5 + 1)) + 5
    maxspect = np.amax(logspect)
    scalespect = logspect / maxspect
    scalespect = scalespect[::-1, :]
    scalespect = np.maximum(scalespect, 0)
    scalespect = np.minimum(scalespect, 1)

    if color:
        plotspect = cm.viridis(scalespect)
        scipy.misc.imsave(filename, plotspect)
    else:
        scipy.misc.imsave(filename, 1 - scalespect)

def stripe(signal, spectheight, sigmas, sampdist):
    """
    Populate an array with time-shifted and windowed versions of an audio
    signal.  This serves as a precursor for FFT calculation.  The first
    spectrogram time frame coincides with the first sample in the signal.
    Out-of-bounds array entries are assumed as zero.

    Parameters
    ----------
    signal : array_like
        Audio signal
    spectheight : int
        Height of the linear-frequency spectrogram
    sigmas : float
        Number of standard deviations after which to cut the window
    sampdist : int
        Time intervals to sample the spectrogram

    Returns
    -------
    stripeplot : ndarray
        Populated array
    """

    signal = np.asarray(signal)

    pos = np.arange(0, signal.size, sampdist)
    stripeplot = pyfftw.zeros_aligned((spectheight*2, pos.size), order='F',
                                      dtype='complex128')

    window = gauss(np.arange(-spectheight, spectheight), spectheight / sigmas)
    
    for i in range(pos.size):
        p = pos[i]
        lo = max(0, spectheight - p)
        hi = min(2*spectheight, signal.size - p + spectheight)
        winsig = np.zeros(2*spectheight)
        winsig[lo:hi] = (signal[p-spectheight+lo:p-spectheight+hi]
                         * window[lo:hi])

        stripeplot[:,i] = winsig

    return stripeplot

def stft(signal, length, sigmas, sampdist):
    """
    Calculate the linear-frequency spectrogram of a given audio signal
    by calling `stripe` and computing the FFT along the first axis.

    Parameters
    ----------
    signal : array_like
        Audio signal
    spectheight : int
        Height of the linear-frequency spectrogram
    sigmas : float
        Number of standard deviations after which to cut the window
    sampdist : int
        Time intervals to sample the spectrogram

    Returns
    -------
    spectrogram : ndarray
        Complex-valued linear-frequency spectrogram
    """

    stripeplot = stripe(signal, length, sigmas, sampdist)
    fft_object = pyfftw.builders.fft(stripeplot, axis=0, threads=8,
                                     overwrite_input=False, avoid_copy=True)
    return fft_object()

def istft(spect, siglen, sigmas, sampdist):
    """
    Reconstruct an audio signal from a complex-valued linear-frequency
    spectrogram via orthogonal projection.  If a sample cannot be inferred
    from the spectrogram, it is set to zero.

    Parameters
    ----------
    spect : array_like
        Complex-valued linear-frequency spectrogram
    siglen : int
        Intended length of the audio signal
    sigmas : float
        Number of standard deviations after which to cut the window
    sampdist : int
        Time intervals to sample the spectrogram

    Returns
    -------
    signal : ndarray
        Reconstructed audio signal
    """

    spect = np.asarray(spect)
    spectheight = spect.shape[0]//2

    fft_object = pyfftw.builders.ifft(spect, axis=0, threads=8,
                                      overwrite_input=False, avoid_copy=True)
    stripeplot = np.real(fft_object())
    signal = np.zeros(siglen)
    window = gauss(np.arange(-spectheight, spectheight), spectheight / sigmas)

    return np.asarray(specttool.unstripe(stripeplot, window, siglen,
                                         spectheight, stripeplot.shape[1],
                                         sampdist))

def spectrogram(signal, spectheight, sigmas, sampdist):
    """
    Calculate the linear-frequency magnitude spectrogram via STFT.

    Parameters
    ----------
    signal : array_like
        Audio signal
    spectheight : int
        Height of the linear-frequency spectrogram
    sigmas : float
        Number of standard deviations after which to cut the window
    sampdist : int
        Time intervals to sample the spectrogram

    Returns
    -------
    spectrogram : ndarray
        Linear-frequency magnitude spectrogram
    """

    stripeplot = stripe(signal, spectheight, sigmas, sampdist)
    spect = np.abs(stft(signal, spectheight, sigmas, sampdist))

    return spect

def synth_audio(spect, siglen, sigmas, sampdist, iterations,
                guess=None, size=2000):
    """
    Reconstruct an audio signal from a linear-frequency magnitude
    spectrogram via the algorithm by Griffin and Lim.
    
    Parameters
    ----------
    spect : array_like
        Linear-frequency magnitude spectrogram
    siglen : int
        Intended length of the audio signal
    sigmas : float
        Number of standard deviations after which to cut the window
    sampdist : int
        Time intervals to sample the spectrogram
    iterations : int
        Number of Griffin-Lim iterations to perform
    guess : array_like
        Initial value for the audio signal
    size : int
        Batch size for the FFT

    Returns
    -------
    signal : ndarray
        Reconstructed audio signal
    """

    spect = np.asarray(spect)
    spectheight = spect.shape[0]
    
    origspect = spect
    spect = pyfftw.zeros_aligned((2*spectheight, spect.shape[1]),
                                 dtype='float64', order='F')
    spect[:spectheight, :] = origspect
    spect[2*spectheight-1:spectheight:-1, :] = origspect[1:, :]
    origspect = spect
    if guess is None:
        sigold = np.zeros(siglen)
    else:
        sigold = np.asarray(guess)

    throwaway = ((spectheight-1) // sampdist + 1) * 2
    
    for k in range(iterations):
        print("Griffin & Lim: iteration {}".format(k))

        signew = np.zeros(siglen)
        pos = np.arange(0, siglen, sampdist)
        lo = 0
        cutlo = 0
        for j in np.arange(0, pos.size, size):
            cutlo = pos[j]
            if j >= throwaway:
                lo = pos[j - throwaway]
            else:
                lo = pos[0]
            if pos.size > j + size + throwaway:
                hi = pos[j + size + throwaway]
            else:
                hi = siglen
            if pos.size > j + size:
                cuthi = pos[j + size]
            else:
                cuthi = hi
            spect = stft(sigold[lo:hi], spectheight, sigmas, sampdist)
            specttool.adapt_mag(
                spect, origspect[:, (lo//sampdist):((hi-1)//sampdist+1)],
                *spect.shape)
            frag = istft(spect, hi-lo, sigmas, sampdist)
            signew[cutlo:cuthi] = frag[cutlo-lo:cuthi-lo]
        sigold = signew

    return signew, spect[:spectheight, :]

def winlog_spect(spect, freqs, basefreq, sigmas):
    """
    Apply a logarithmic transform on the frequency axis of a linear-frequency
    magnitude spectrogram while preserving the width of the horizontal lines
    via Gaussian smoothing.  The attenuation of the higher frequency is
    counteracted by scaling.

    Parameters
    ----------
    spect : array_like
        Linear-frequency magnitude spectrogram
    freqs : array_like
        Frequencies to place the smoothing kernel
        (normalized to the sampling frequency)
    basefreq : float
        Frequency to assume as a minimum for smoothing
        (normalized to the sampling frequency)
    sigmas : float
        Number of standard deviations after which to cut the kernel

    Returns
    -------
    logspect : ndarray
        Log-frequency magnitude spectrogram
    """

    spect = np.asarray(spect)
    freqs = np.asarray(freqs) * (2 * spect.shape[0])
    basefreq = basefreq * (2 * spect.shape[0])
    
    logspect = np.zeros((freqs.size, spect.shape[1]))

    for i in range(freqs.size):
        freq = freqs[i]
        freqint = int(np.round(freq))
        freqdiff = freq - freqint
        stretch = freq / basefreq
        if stretch > 1 + 1e-12:
            sigma = sigmas/np.pi * np.sqrt(np.square(stretch) - 1)
            width = int(np.ceil(sigma * sigmas))
            lo = max(0, width - freqint)
            hi = min(2*width, spect.shape[0] - freqint + width)
            window = gauss(np.arange(lo-width, hi-width) - freqdiff, sigma,
                           False)
            logspectrow = np.sum(window.reshape(hi-lo, 1)
                                 * spect[freqint-width+lo:freqint-width+hi, :],
                                 axis=0)
        else:
            logspectrow = spect[freqint,:]
        logspect[i,:] = logspectrow

    return logspect

def logspect_freq(signal, spectheight, sigmas, sampdist, basefreq,
                  minfreq, maxfreq, numfreqs):
    """
    Compute the Mel-frequency spectrogram of an audio signal.

    Parameters
    ----------
    signal : array_like
        Audio signal
    spectheight : int
        Height of the linear-frequency spectrogram
    sigmas : float
        Number of standard deviations after which to cut the window/kernel
    sampdist : int
        Time intervals to sample the spectrogram
    basefreq : float
        Frequency to assume as a minimum for smoothing
        (normalized to the sampling frequency)
    minfreq : float
        Minimum frequency to be represented (included)
        (normalized to the sampling frequency)
    maxfreq : float
        Maximum frequency to be represented (excluded)
        (normalized to the sampling frequency)
    numfreqs : float
        Height of the log-frequency spectrogram

    Returns
    -------
    logspect : ndarray
        Log-frequency magnitude spectrogram
    """

    signal = np.asarray(signal)

    freqs = np.logspace(np.log10(minfreq), np.log10(maxfreq), numfreqs,
                        endpoint=False)

    spect = spectrogram(signal, spectheight, sigmas, sampdist)[:spectheight,:]
    logspect = winlog_spect(spect, freqs, basefreq, sigmas)
        
    return logspect

def logspect_tf(signal, spectheight, sigmas, sampdist, basefreq,
                minfreq, maxfreq, numfreqs):
    """
    Compute the time-smoothed CQT of an audio signal.

    Parameters
    ----------
    signal : array_like
        Audio signal
    spectheight : int
        Height of the linear-frequency spectrogram
    sigmas : float
        Number of standard deviations after which to cut the window/kernel
    sampdist : int
        Time intervals to sample the spectrogram
    basefreq : float
        Frequency to assume as a minimum for smoothing
        (normalized to the sampling frequency)
    minfreq : float
        Minimum frequency to be represented (included)
        (normalized to the sampling frequency)
    maxfreq : float
        Maximum frequency to be represented (excluded)
        (normalized to the sampling frequency)
    numfreqs : float
        Height of the log-frequency spectrogram

    Returns
    -------
    logspect : ndarray
        Log-frequency magnitude spectrogram
    """

    signal = np.asarray(signal)

    freqs = np.logspace(np.log10(minfreq), np.log10(maxfreq), numfreqs,
                        endpoint=False)
    pos = np.arange(0, signal.size)
    samppos = np.arange(0, signal.size, sampdist)
    spect = np.zeros((numfreqs, samppos.size))

    with tf.Graph().as_default():
        win_var = tf.placeholder(tf.float32, [None, 2])
        smooth_var = tf.placeholder(tf.float32, [None])
        row_var = tf.nn.conv2d(tf.reshape(tf.to_float(signal), [1, 1, -1, 1]),
                               tf.reshape(win_var, [1, -1, 1, 2]),
                               [1, 1, 1, 1], 'SAME')
        y_var = tf.nn.conv2d(tf.sqrt(tf.square(row_var[:,:,:,0:1])
                                     + tf.square(row_var[:,:,:,1:2])),
                             tf.reshape(smooth_var, [1, -1, 1, 1]),
                             [1, 1, sampdist, 1], 'SAME')
        sess = tf.Session()
    
    for i in range(freqs.size):
        freq = freqs[i]
        print("freq {}: {}".format(i, freq))
        
        stretch = basefreq / freq
        sigma = spectheight * stretch
        sigmalen = int(np.ceil(sigma))
        window = gauss(np.arange(-sigmalen, sigmalen+1), sigma / sigmas)
        phase = np.exp(-1j*2*np.pi * freq * np.arange(-sigmalen, sigmalen+1))
        window = window * phase

        if stretch >= 1 - 1e-6:
            convwindow = [1]
        else:
            sigmaadd = spectheight * np.sqrt(1 - np.square(stretch))
            sigmaaddlen = int(np.ceil(sigmaadd))
            convwindow = gauss(np.arange(-sigmaaddlen, sigmaaddlen+1),
                               sigmaadd / sigmas)
        y_val, = sess.run([y_var],
                          feed_dict={
                              win_var: np.vstack((np.real(window),
                                                  np.imag(window))).T,
                              smooth_var: convwindow})
        spect[i,:] = np.ravel(y_val)

    return spect

def logspect_pursuit(signal, spectheight, sigmas, sampdist, basefreq,
                     minfreq, maxfreq, numfreqs, fsigma):
    """
    Compute the log-frequency frequency via sparse pursuit.

    Parameters
    ----------
    signal : array_like
        Audio signal
    spectheight : int
        Height of the linear-frequency spectrogram
    sigmas : float
        Number of standard deviations after which to cut the window/kernel
    sampdist : int
        Time intervals to sample the spectrogram
    basefreq : float
        Frequency to assume as a minimum for smoothing
        (normalized to the sampling frequency)
    minfreq : float
        Minimum frequency to be represented (included)
        (normalized to the sampling frequency)
    maxfreq : float
        Maximum frequency to be represented (excluded)
        (normalized to the sampling frequency)
    numfreqs : float
        Height of the log-frequency spectrogram
    fsigma : float
        Standard deviation (frequency)

    Returns
    -------
    logspect : ndarray
        Log-frequency magnitude spectrogram
    """

    signal = np.asarray(signal)

    minfreq = minfreq * (2 * spectheight)
    maxfreq = maxfreq * (2 * spectheight)

    spect = spectrogram(signal, spectheight, sigmas, sampdist)[:spectheight, :]
    logspect = np.zeros((numfreqs, spect.shape[1]))
    linspect = np.zeros(spect.shape)

    print("timeslots: {}".format(spect.shape[1]))
    
    inst_dict = np.asarray([[1.]])
    harshifts = np.asarray([0.])

    init = None

    if basefreq is None:
        stretch = 1
    else:
        basefreq = basefreq * (2 * spectheight)
        stretch = numfreqs / np.log(maxfreq/minfreq) / basefreq
    print("stretch: {}".format(stretch))

    pexp = 1
    qexp = 1
    
    for i in range(spect.shape[1]):
        print("Timeslot {}".format(i))
        y = spect[:,i]
        peaks, reconstruction = pursuit.peak_pursuit(y, 1000, 1000, 20, inst_dict,
                                                     fsigma, harshifts,
                                                     pursuit.max_selector,
                                                     (5,), pexp, qexp)
        logshifts = np.ones(len(peaks)) * (-np.inf)
        idcs, = np.where(peaks.shifts > 0)
        logshifts[idcs] = (numfreqs / np.log(maxfreq / minfreq)
                           * np.log(peaks.shifts[idcs] / minfreq))
        peaks.shifts = logshifts
        peaks.sigmas = peaks.sigmas * stretch
        logspect[:, i] = pursuit.inst_shift(peaks, inst_dict, harshifts, pexp,
                                            numfreqs, len(peaks))
        linspect[:, i] = reconstruction

    return logspect, linspect

if __name__ == '__main__':
    n = 200000
    signal = np.zeros(n)
    signal[100000] = 2000
    signal += (np.sin(2*np.pi * 1000/48000 * np.arange(0, n)) +
               np.sin(2*np.pi * 2000/48000 * np.arange(0, n)) +
               np.sin(2*np.pi * 4000/48000 * np.arange(0, n)) +
               np.sin(2*np.pi * 8000/48000 * np.arange(0, n)) +
               np.sin(2*np.pi * 16000/48000 * np.arange(0, n)))

    spectheight = 1024*6

    spect = spectrogram(signal, spectheight, 6, 128)
    spectwrite('spect_lin.png', spect)

    spect = logspect_freq(signal, spectheight, 6, 128, 640/48000,
                          640/48000, 20480/48000, 1024)
    spectwrite('spect_freq.png', spect)

    spect = logspect_tf(signal, spectheight, 6, 128, 640/48000,
                        640/48000, 20480/48000, 1024)
    spectwrite('spect_tf.png', spect)
