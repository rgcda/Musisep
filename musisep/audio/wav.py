#!python3

"""
Module to handle WAV audio data.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.io.wavfile as wav

def unify(in_data):
    """
    Convert the input data to a double-type array with values in [-1,1].
    Input type must be `double`, `float32`, `int32`, `int16`, or `uint8`.

    Parameters
    ----------
    in_data : array_like
        Data to be unified

    Returns
    -------
    out_data : ndarray
        Unified data.
    """

    data = np.asarray(in_data)

    if data.dtype == np.dtype('double'):
        return data
    elif data.dtype == np.dtype('float32'):
        return data.astype('double')
    elif data.dtype == np.dtype('int32'):
        return data.astype('double') / 2147483648
    elif data.dtype == np.dtype('int16'):
        return data.astype('double') / 32768
    elif data.dtype == np.dtype('uint8'):
        return (data.astype('double') - 128) / 128
    else:
        raise NotImplementedError("invalid audio type: {}".format(data.dtype))

def read(filename):
    """Read WAV audio data from a file.  If the data has multiple channels,
    they will be averaged.

    Parameters
    ----------
    filename : string
        Name of the WAV file.

    Returns
    -------
    data : ndarray
        Audio data as `double` array with values in [-1,1].
    samprate : int
        Sampling rate of the WAV file.
    """

    samprate, rawdata = wav.read(filename)
    data = unify(rawdata)

    if len(data.shape) == 2:
        data = np.mean(data, axis=1)

    return data, samprate

def read_stereo(filename):
    """
    Read WAV audio data from a file.  If the data has multiple channels, they
    will be returned as rows of the output array.

    Parameters
    ----------
    filename : string
        Name of the WAV file.

    Returns
    -------
    data : ndarray
        Audio data as `double` array with values in [-1,1].
    samprate : int
        Sampling rate of the WAV file.
    """

    samprate, rawdata = wav.read(filename)
    data = unify(rawdata)

    return data.T

def write(filename, signal, samprate):
    """
    Write WAV audio data to a file.  The data type must be supported by
    scipy.io.wavfile.

    Parameters
    ----------
    filename : string
        Name of the WAV file.
    signal : array_like
        Audio data to write.
    samprate : int
        Intended sampling rate of the WAV file.

    Returns
    -------
    maxval : scalar
        Number by which was divided during normalization.
    """

    maxval = np.amax(np.abs(signal))
    print("maxval for {}: {}".format(filename, maxval))
    signal = signal / maxval
    wav.write(filename, samprate, signal)

    return maxval
