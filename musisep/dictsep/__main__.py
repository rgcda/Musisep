#!python3

"""
Wrapper for the dictionary learning algorithm.  When invoked, the audio
sources in the supplied audio file are separated.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import os.path
import pickle

from ..audio import spect
from ..audio import wav
from ..audio import performance
from . import dictlearn

def main(mixed_soundfile, orig_soundfiles, inst_num, tone_num, pexp, qexp,
         har, sigmas, sampdist, spectheight, logspectheight,
         minfreq, maxfreq, out_name, runs, lifetime, num_dicts, mask,
         plot_range):
    """
    Wrapper function for the dictionary learning algorithm.

    Parameters
    ----------
    mixed_soundfile : string
        Name of the mixed input file
    orig_soundfiles : list of string or NoneType
        Names of the files with the isolated instrument tracks or None
    inst_num : int
        Number of instruments
    tone_num : int
        Maximum number of simultaneous tones
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
    out_name : string
        Prefix for the file names
    runs : int
        Number of training iterations to perform
    lifetime : int
        Number of steps after which to renew the dictionary
    num_dicts : int
        Number of different dictionaries to generate and train
    mask : bool
        Whether to apply spectral masking
    plot_range : slice or NoneType
        part of the spectrogram to plot
    """

    signal, samprate = wav.read(mixed_soundfile)

    orig_spectrum = spect.spectrogram(
        signal, spectheight, sigmas, sampdist)[:spectheight, :]
    if plot_range is not None:
        spect.spectwrite('output/{}-orig.png'.format(out_name),
                            orig_spectrum[:spectheight, plot_range])

    if orig_soundfiles is None:
        orig_signals = None
    else:
        orig_signals = np.asarray([wav.read(f)[0] for f in orig_soundfiles])
        orig_spectrums = [spect.spectrogram(
            os, spectheight, sigmas, sampdist)[:spectheight, :]
            for os in orig_signals]

    fsigma = sigmas/np.pi

    if os.path.exists('output/{}.npy'.format(out_name)):
        logspect = np.load('output/{}.npy'.format(out_name))
    else:
        logspect, linspect = spect.logspect_pursuit(signal, spectheight,
                                                    sigmas, sampdist, None,
                                                    minfreq/samprate,
                                                    maxfreq/samprate,
                                                    logspectheight, fsigma)
        np.save('output/{}.npy'.format(out_name), logspect)

        if plot_range is not None:
            spect.spectwrite('output/{}-log.png'.format(out_name),
                             logspect[:, plot_range])
            spect.spectwrite('output/{}-lin.png'.format(out_name),
                             linspect[:, plot_range])

    audio_measures = []
    spectrum_measures = []

    for r in range(num_dicts):
        print("seed: {}".format(r))
        out_name = 'mozart-{}'.format(r)
        np.random.seed(r)
        
        if os.path.exists('output/{}-dict.npy'.format(out_name)):
            inst_dict = np.load('output/{}-dict.npy'.format(out_name))
        else:
            inst_dict = dictlearn.learn_spect_dict(
                logspect, fsigma, tone_num, inst_num*2, pexp, qexp,
                har, logspectheight, minfreq, maxfreq, runs, lifetime)
            np.save('output/{}-dict.npy'.format(out_name), inst_dict)
    
        print(inst_dict)

        if os.path.exists('output/{}-spect.pkl'.format(out_name)):
            [dict_spectrum, inst_spectrums,
             dict_spectrum_lin, inst_spectrums_lin] = \
                pickle.load(open('output/{}-spect.pkl'.format(out_name), 'rb'))
        else:
            (dict_spectrum, inst_spectrums,
             dict_spectrum_lin, inst_spectrums_lin) = \
                dictlearn.synth_spect(
                    logspect, tone_num, inst_dict, fsigma,
                    spectheight, pexp, qexp,
                    minfreq/samprate, maxfreq/samprate)
            pickle.dump([dict_spectrum, inst_spectrums,
                         dict_spectrum_lin, inst_spectrums_lin],
                        open('output/{}-spect.pkl'.format(out_name), 'wb'))
    
        inst_spectrums_lin, mask_spect = dictlearn.mask_spectrums(
            inst_spectrums_lin, orig_spectrum)
        dict_spectrum_lin = dict_spectrum_lin * mask_spect

        if plot_range is not None:
            spect.spectwrite('output/{}-synth.png'.format(out_name),
                             dict_spectrum[:, plot_range])
            spect.spectwrite('output/{}-synth-lin.png'.format(out_name),
                             dict_spectrum_lin[:, plot_range])
            for i in range(len(inst_spectrums)):
                spect.spectwrite(
                    'output/{}-synth{}.png'.format(out_name, i),
                    inst_spectrums[i][:, plot_range])
                spect.spectwrite(
                    'output/{}-synth{}-lin.png'.format(out_name, i),
                    inst_spectrums_lin[i][:, plot_range])

        siglen = signal.size
        synth_signals = np.zeros((inst_num, siglen))
        audio, _ = spect.synth_audio(dict_spectrum_lin, siglen,
                                     sigmas, sampdist, 1, signal)
        wav.write('output/{}-synth.wav'.format(out_name), audio, samprate)
        for i in range(len(inst_spectrums_lin)):
            audio, _ = spect.synth_audio(inst_spectrums_lin[i],
                                         siglen, sigmas, sampdist, 1,
                                         signal)
            synth_signals[i, :] = audio
            wav.write('output/{}-synth{}.wav'.format(out_name, i),
                      audio, samprate)

        if orig_signals is not None:
            _, ap = performance.select_perm(*performance.measures(
                synth_signals, orig_signals))
            audio_measures.append(ap)
            print(ap)

            _, sp = performance.select_perm(*performance.measures(
                np.vstack([np.ravel(spect) for spect in inst_spectrums_lin]),
                np.vstack([np.ravel(spect) for spect in orig_spectrums])))
            spectrum_measures.append(sp)
            print(sp)

    if orig_signals is not None:
        print("Global measures:")
        print(np.mean(np.asarray(audio_measures), axis=0))
        print(np.mean(np.asarray(spectrum_measures), axis=0))

if __name__ == '__main__':
    main(mixed_soundfile='input/mozart/mix.wav',
         orig_soundfiles=['input/mozart/recorder.wav',
                          'input/mozart/violin.wav'],
         inst_num=2,
         tone_num=2,
         pexp=2,
         qexp=1/2,
         har=25,
         sigmas=6,
         sampdist=256,
         spectheight=6*1024,
         logspectheight=1024,
         minfreq=20,
         maxfreq=20480,
         out_name='mozart',
         runs=100000,
         lifetime=500,
         num_dicts=10,
         mask=True,
         plot_range=slice(0, 1580))
