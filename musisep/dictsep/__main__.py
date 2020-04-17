#!python3

"""
Wrapper for the dictionary learning algorithm.  When invoked, the audio
sources in the supplied audio file are separated.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import sys
import os.path
import pickle
import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import matplotlib.cm as cm

from ..audio import spect
from ..audio import wav
from ..audio import performance
from . import dictlearn

def correct_signal_length(signal, length):
    """
    Right-pad or right-crop the signal such that it fits the desired length.

    Arguments
    ---------
    signal : ndarray
        Signal to be adjusted
    length : int
        Desired length of the signal

    Returns
    -------
    ndarray
        Adjusted signal
    """

    if signal.size > length:
        return signal[:length]
    elif signal.size < length:
        return np.concatenate([signal, np.zeros(length - signal.size)])
    else:
        return signal

def main(mixed_soundfile, orig_soundfiles, out_name, out_name_run_suffix="",
         inst_num=2, tone_num=1, pexp=1, qexp=0.5, har=25, sigmas=6,
         sampdist=256, spectheight=6*1024, logspectheight=1024, minfreq=20,
         maxfreq=20480, runs=10000, lifetime=500, num_dicts=10, mask=True,
         color=False, plot_range=None, spect_method="pursuit",
         supply_dicts=None):
    """
    Wrapper function for the dictionary learning algorithm.

    Parameters
    ----------
    mixed_soundfile : string
        Name of the mixed input file
    orig_soundfiles : list of string or NoneType
        Names of the files with the isolated instrument tracks or None
    out_name : string
        Prefix for the file names
    out_name_suffix : string
        Extra label for the output files
    inst_num : int
        Number of instruments
    tone_num : int
        Maximum number of simultaneous tones for each instrument
    pexp : float
        Exponent for the addition of sinusoids
    qexp : float
        Exponent to be applied on the spectrum
    har : int
        Number of harmonics
    sigmas : float
        Number of standard deviations after which to cut the window/kernel
    sampdist : int
        Time intervals to sample the spectrogram
    spectheight : int
        Height of the linear-frequency spectrogram
    logspectheight : int
        Height of the log-frequency spectrogram
    minfreq : float
        Minimum frequency in Hz to be represented (included)
    maxfreq : float
        Maximum frequency in Hz to be represented (excluded)
    runs : int
        Number of training iterations to perform
    lifetime : int
        Number of steps after which to renew the dictionary
    num_dicts : int
        Number of different dictionaries to generate and train
    mask : bool
        Whether to apply spectral masking
    color : bool or string
        Whether color should be used, or specification of the color scheme
    plot_range : slice or NoneType
        Part of the spectrogram to plot
    spect_method : string
        If set to `"mel"`, a mel spectrogram is used for separation.
        Otherwise, the log-frequency spectrogram is generated via
        sparse pursuit.
    supply_dicts : NoneType or list of array_like
        Is specified, use the given dictionaries rather than computing
        new ones

    Returns
    -------
    inst_dicts : list of ndarray
        Dictionaries that were used for the separation
    """

    signal, samprate = wav.read(mixed_soundfile)

    plotlen = signal.size

    orig_spectrum = spect.spectrogram(
        signal, spectheight, sigmas, sampdist)[:spectheight, :]

    if plot_range is not None:
        spect.spectwrite('output/{}-orig.png'.format(out_name),
                         orig_spectrum[:spectheight, plot_range],
                         color)

    if orig_soundfiles is None:
        orig_signals = None
    else:
        orig_signals = np.asarray(
            [correct_signal_length(wav.read(f)[0], signal.size)
             for f in orig_soundfiles])
        orig_spectrums = [spect.spectrogram(
            os, spectheight, sigmas, sampdist)[:spectheight, :]
            for os in orig_signals]

    fsigma = sigmas/np.pi

    if (os.path.exists('output/{}-lin.npy'.format(out_name))
        and os.path.exists('output/{}-log.npy'.format(out_name))
        and os.path.exists('output/{}-stretch.npy'.format(out_name))):
        linspect = np.load('output/{}-lin.npy'.format(out_name))
        logspect = np.load('output/{}-log.npy'.format(out_name))
        stretch = np.load('output/{}-stretch.npy'.format(out_name))
    elif spect_method == "mel":
        stretch = (logspectheight / np.log(maxfreq/minfreq)
                   / (minfreq / samprate * 2 * spectheight))
        print("stretch: {}".format(stretch))
        logspect, linspect = spect.logspect_mel(signal, spectheight,
                                                sigmas, sampdist,
                                                minfreq/samprate,
                                                minfreq/samprate,
                                                maxfreq/samprate,
                                                logspectheight)
        logspect = np.sqrt(logspect)
        linspect = np.sqrt(linspect)
        np.save('output/{}-lin.npy'.format(out_name), linspect)
        np.save('output/{}-log.npy'.format(out_name), logspect)
        np.save('output/{}-stretch.npy'.format(out_name), stretch)
    else:
        logspect, linspect = spect.logspect_pursuit(signal, spectheight,
                                                    sigmas, sampdist, None,
                                                    minfreq/samprate,
                                                    maxfreq/samprate,
                                                    logspectheight, fsigma)
        stretch = 1
        np.save('output/{}-lin.npy'.format(out_name), linspect)
        np.save('output/{}-log.npy'.format(out_name), logspect)
        np.save('output/{}-stretch.npy'.format(out_name), stretch)

    if plot_range is not None:
        spect.spectwrite('output/{}-lin.png'.format(out_name),
                         linspect[:, plot_range], color)
        spect.spectwrite('output/{}-log.png'.format(out_name),
                         logspect[:, plot_range], color)

    audio_measures = []
    inst_dicts = []

    for r in range(0, num_dicts):
        print("seed: {}".format(r))
        out_name_run = out_name + out_name_run_suffix + '-{}'.format(r)
        np.random.seed(r)

        if supply_dicts is not None:
            inst_dict = np.asarray(supply_dicts[r])
        elif os.path.exists('output/{}-dict.npy'.format(out_name_run)):
            inst_dict = np.load('output/{}-dict.npy'.format(out_name_run))
        else:
            inst_dict = dictlearn.learn_spect_dict(
                logspect, fsigma*stretch, tone_num, inst_num * 2, pexp, qexp,
                har, minfreq, maxfreq, runs, lifetime)
            np.save('output/{}-dict.npy'.format(out_name_run), inst_dict)

        print(inst_dict)
        inst_dicts.append(inst_dict)

        if os.path.exists('output/{}-spect.pkl'.format(out_name_run)):
            [dict_spectrum, inst_spectrums,
             dict_spectrum_lin, inst_spectrums_lin] = \
                pickle.load(open('output/{}-spect.pkl'.format(out_name_run),
                                 'rb'))
        else:
            (dict_spectrum, inst_spectrums,
             dict_spectrum_lin, inst_spectrums_lin) = \
                dictlearn.synth_spect(
                    logspect, tone_num, inst_dict, fsigma*stretch,
                    spectheight, pexp, qexp,
                    minfreq/samprate, maxfreq/samprate, stretch)
            pickle.dump([dict_spectrum, inst_spectrums,
                         dict_spectrum_lin, inst_spectrums_lin],
                        open('output/{}-spect.pkl'.format(out_name_run), 'wb'))

        if mask:
            inst_spectrums_lin, mask_spect = dictlearn.mask_spectrums(
                inst_spectrums_lin, orig_spectrum)
            dict_spectrum_lin = dict_spectrum_lin * mask_spect
            mask_str = "mask"
        else:
            mask_str = "nomask"

        if plot_range is not None:
            spect.spectwrite('output/{}-synth.png'
                             .format(out_name_run),
                             dict_spectrum[:, plot_range], color)
            spect.spectwrite('output/{}-synth-lin-{}.png'
                             .format(out_name_run, mask_str),
                             dict_spectrum_lin[:, plot_range], color)
            for i in range(len(inst_spectrums)):
                spect.spectwrite(
                    'output/{}-synth{}.png'
                    .format(out_name_run, i),
                    inst_spectrums[i][:, plot_range], color)
                spect.spectwrite(
                    'output/{}-synth{}-lin-{}.png'
                    .format(out_name_run, i, mask_str),
                    inst_spectrums_lin[i][:, plot_range], color)

        siglen = signal.size
        synth_signals = np.zeros((inst_num, siglen))
        audio, _ = spect.synth_audio(dict_spectrum_lin, siglen,
                                     sigmas, sampdist, 1, signal)
        wav.write('output/{}-synth-{}.wav'.format(out_name_run, mask_str),
                  audio, samprate)
        for i in range(len(inst_spectrums_lin)):
            audio, _ = spect.synth_audio(inst_spectrums_lin[i],
                                         siglen, sigmas, sampdist, 1,
                                         signal)
            synth_signals[i, :] = audio
            wav.write('output/{}-synth{}-{}.wav'
                      .format(out_name_run, i, mask_str),
                      audio, samprate)

        if orig_signals is not None:
            perm, perf = performance.select_perm(*performance.measures(
                synth_signals, orig_signals))
            audio_measures.append(perf)
            print("Permutation:")
            print(perm)
            print("Performance:")
            print(perf)

    if orig_signals is not None:
        audio_measures = np.asarray(audio_measures)
        print("Global measures mean:")
        print(np.mean(audio_measures, axis=0))
        print("Global measures stdev:")
        print(np.std(audio_measures, axis=0, ddof=1))
        bestidx = np.argmax(np.sum(audio_measures, axis=2)[:, 0])
        print("Global measures best index: {}".format(bestidx))
        print("Global measures best:")
        print(audio_measures[bestidx, :, :])

        np.savetxt('output/{}{}-{}-measures.dat'
                   .format(out_name, out_name_run_suffix, mask_str),
                   np.reshape(audio_measures, [num_dicts, 3 * inst_num]))

    return inst_dicts

def separate_mozart_recorder_violin():
    "Separation of recorder and violin on the piece by Mozart"

    main(mixed_soundfile='input/mozart/mix.wav',
         orig_soundfiles=['input/mozart/recorder.wav',
                          'input/mozart/violin.wav'],
         out_name='mozart/mozart',
         runs=100000,
         mask=True,
         plot_range=slice(0, 1580))
    main(mixed_soundfile='input/mozart/mix.wav',
         orig_soundfiles=['input/mozart/recorder.wav',
                          'input/mozart/violin.wav'],
         out_name='mozart/mozart',
         runs=100000,
         mask=False,
         plot_range=slice(0, 1580))

def separate_mozart_recorder_violin_mel():
    "Separation of recorder and violin on the piece by Mozart"

    main(mixed_soundfile='input/mozart/mix.wav',
         orig_soundfiles=['input/mozart/recorder.wav',
                          'input/mozart/violin.wav'],
         out_name='mozart_mel/mozart',
         minfreq=200,
         runs=100000,
         mask=True,
         plot_range=slice(0, 1580),
         spect_method="mel")
    main(mixed_soundfile='input/mozart/mix.wav',
         orig_soundfiles=['input/mozart/recorder.wav',
                          'input/mozart/violin.wav'],
         out_name='mozart_mel/mozart',
         minfreq=200,
         runs=100000,
         mask=False,
         plot_range=slice(0, 1580),
         spect_method="mel")

def separate_mozart_clarinet_piano():
    "Separation of clarinet and piano on the piece by Mozart"

    main(mixed_soundfile='input/mozart-cl/mix-cl-piano.wav',
         orig_soundfiles=['input/mozart-cl/clarinet-high.wav',
                          'input/mozart-cl/piano-low.wav'],
         out_name='mozart-cl/mozart',
         runs=100000)

def separate_frere_jacques():
    """
    Separation of Bb tin whistle and viola and generalization to
    C tin whistle and violin, then vice versa.
    """

    inst_dicts = main(mixed_soundfile='input/fj/bb.wav',
                      orig_soundfiles=['input/fj/bb-tw.wav',
                                       'input/fj/bb-viola.wav'],
                      out_name='fj/bb',
                      runs=100000)
    main(mixed_soundfile='input/fj/c.wav',
         orig_soundfiles=['input/fj/c-tw.wav',
                          'input/fj/c-violin.wav'],
         out_name='fj/c',
         out_name_run_suffix='-gen',
         runs=100000,
         supply_dicts=inst_dicts)

    inst_dicts = main(mixed_soundfile='input/fj/c.wav',
                      orig_soundfiles=['input/fj/c-tw.wav',
                                       'input/fj/c-violin.wav'],
                      out_name='fj/c',
                      runs=100000)
    main(mixed_soundfile='input/fj/bb.wav',
         orig_soundfiles=['input/fj/bb-tw.wav',
                          'input/fj/bb-viola.wav'],
         out_name='fj/bb',
         out_name_run_suffix='-gen',
         supply_dicts=inst_dicts,
         runs=100000)

def separate_jaiswal(number):
    """
    Separation of the data by Jaiswal et al.

    Parameters
    ----------
    number : int
        Number of the sample to be considered.
    """

    main(mixed_soundfile='input/jaiswal/test{}.wav'.format(number),
         orig_soundfiles=['input/jaiswal/test{}-01.wav'.format(number),
                          'input/jaiswal/test{}-02.wav'.format(number)],
         out_name='jaiswal/jaiswal{}'.format(number))

def separate_duan():
    """
    Separation of the data by Duan et al.

    Parameters
    ----------
    number : int
        Number of the sample to be considered.
    """

    main(mixed_soundfile='input/duan/Euphonium_Oboe.wav',
         orig_soundfiles=['input/duan/Oboe.wav',
                          'input/duan/Euphonium.wav'],
         out_name='duan/eo')

    main(mixed_soundfile='input/duan/dyrcj_wqyn.wav',
         orig_soundfiles=['input/duan/dyrcj_piccolo.wav',
                          'input/duan/wqyn_organ.wav'],
         out_name='duan/po')

    main(mixed_soundfile='input/duan/dyrcj_wqyn_fywz.wav',
         orig_soundfiles=['input/duan/dyrcj_piccolo.wav',
                          'input/duan/wqyn_organ.wav',
                          'input/duan/fywz_oboe.wav'],
         out_name='duan/poo',
         inst_num=3)

if __name__ == '__main__':
    separate_mozart_recorder_violin()
    separate_mozart_recorder_violin_mel()
    separate_mozart_clarinet_piano()
    separate_frere_jacques()

    # The number of the sample is given via command line.
    # Unfortunately, we cannot distribute the data.
    #separate_jaiswal(int(sys.argv[1]))

    # Get the data from:
    # https://sites.google.com/site/mperesult/musicseparationresults
    # Upsample to 44100 Hz.
    #separate_duan()
