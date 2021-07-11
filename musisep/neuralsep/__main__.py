#!python3

"""
Wrapper for the policy gradient separation algorithm.  When invoked, the
audio sources in the supplied audio file are separated.
"""

import numpy as np
import tensorflow as tf

from . import trainsep

def separate_train(seed_np, seed_tf, name, mixed_soundfile, orig_soundfiles,
                   loss_coeffs=(0, 10, 1, 10), har_num=25, num_guesses=(3, 3),
                   spl=0.9, batch_size=12, batch_size_pred=100,
                   virt_batch_mul=1,
                   stepsize_net=1e-3, stepsize_dict=1e-4, tau=0.01,
                   max_iter=100000, eval_interval=2500,
                   sampdist=128, sub_factor=4, sigmas_an=6,
                   load_dir=None, plot_color=False,
                   save_points=(70000,), init_dict=None):
    """
    Separate a music recording into the contribution of the
    individual instruments.

    Parameters
    ----------
    seed_np : int
        Random seed for NumPy
    seed_tf : int
        Random seed for Tensorflow
    name : string
        Name of the training run.  Used for files names and logging.
    mixed_soundfile : string
        Name of the sound file containing the mixture
    orig_soundfiles : sequence of string
        Name of the sound files containing the individual instrument tracks
    loss_coeffs : sequence of float
        Weights of the dictionary prediction loss, the sparse loss,
        the regularization loss, and the direct prediction loss
    har_num : int
        Number of harmonics to identify
    num_guesses : sequence of int
        Number of samples per tone
    spl : float
        Discount factor for the sparsity
    batch_size : int
        Batch size for training
    batch_size_pred : int
        Batch size for prediction
    virt_batch_mul : int
        Virtual batch multiplier
    stepsize_net : float
        Learning rate for training the neural network
    stepsize_dict : float
        Learning rate for training the dictionary
    tau : float
        Exponent to control exploration
    max_iter : int
        Total number of training iterations
    eval_interval : int
        Interval at which to evaluate the entire spectrogram
    sampdist : int
        Time interval of the spectrogram
    sub_factor : int
        Factor by which to subsample the spectrogram for resynthesis
    sigmas_an : float
       Number of standard deviations at which the analysis window is cut
    load_dir : string
       Path from where to preload the model and the dictionary
    plot_color : string or NoneType
       Whether to make a color plot
    save_points: sequence of int
       Iterations at which to save the output
    init_dict : tensor of float
       Dictionary with the shape [instruments, harmonics]
    """

    np.random.seed(seed_np)
    tf.random.set_seed(seed_tf)

    fullname = name + '-s{}.{}'.format(seed_np, seed_tf)
    trainer = trainsep.Trainer(
        fullname, mixed_soundfile, orig_soundfiles, loss_coeffs,
        har_num, num_guesses, spl, batch_size, batch_size_pred,
        virt_batch_mul,
        stepsize_net, stepsize_dict, tau,
        sampdist, sub_factor, sigmas_an, plot_color, save_points,
        init_dict)
    if load_dir is not None:
        trainer.load(tf.saved_model.load(load_dir))
    trainer.train_loop(max_iter, eval_interval)

def separate_mozart(seed):
    """
    Separate the piece by Mozart for recorder and violin.

    Parameters
    ----------
    seed : int
        Random seed for both NumPy and Tensorflow
    """

    separate_train(seed, seed, 'mozart/mozart',
                   'input/mozart/mix.wav',
                   ['input/mozart/recorder.wav',
                    'input/mozart/violin.wav'],
                   plot_color=False)

def separate_mozart_cl(seed):
    """
    Separate the piece by Mozart for clarinet and piano.

    Parameters
    ----------
    seed : int
        Random seed for both NumPy and Tensorflow
    """

    separate_train(seed, seed, 'mozart-cl/mozart',
                   'input/mozart-cl/mix-cl-piano.wav',
                   ['input/mozart-cl/clarinet-high.wav',
                    'input/mozart-cl/piano-low.wav'],
                   plot_color=False)

def separate_mozart_piano(seed):
    """
    Separate the piece by Mozart for clarinet and piano.

    Parameters
    ----------
    seed : int
        Random seed for both NumPy and Tensorflow
    """

    separate_train(seed, seed, 'mozart-cl/mozart-piano',
                   'input/mozart-cl/piano-low.wav',
                   ['input/mozart-cl/piano-low.wav'],
                   plot_color=False,
                   num_guesses=(3,))

def separate_duan_synth2(seed):
    """
    Separate the piece from Duan et al. with piccolo and organ.

    Parameters
    ----------
    seed : int
        Random seed for both NumPy and Tensorflow
    """

    separate_train(seed, seed, 'duan/synth2',
                   'input/duan22050/dyrcj_wqyn.wav',
                   ['input/duan22050/dyrcj_piccolo.wav',
                    'input/duan22050/wqyn_organ.wav'],
                   sampdist=32
    )

def separate_duan_synth3(seed):
    """
    Separate the piece from Duan et al. with piccolo, organ, and oboe.

    Parameters
    ----------
    seed : int
        Random seed for both NumPy and Tensorflow
    """
    separate_train(seed, seed, 'duan/synth3',
                   'input/duan22050/dyrcj_wqyn_fywz.wav',
                   ['input/duan22050/dyrcj_piccolo.wav',
                    'input/duan22050/wqyn_organ.wav',
                    'input/duan22050/fywz_oboe.wav'],
                   num_guesses=(3, 3, 3),
                   sampdist=32,
                   max_iter=300000,
                   batch_size=4, batch_size_pred=50,
                   virt_batch_mul=3,
                   save_points=[70000*3])

def separate_duan_acous(seed):
    """
    Separate the piece from Duan et al. with euphonium and oboe.

    Parameters
    ----------
    seed : int
        Random seed for both NumPy and Tensorflow
    """

    separate_train(seed, seed, 'duan/acous',
                   'input/duan22050/Euphonium_Oboe.wav',
                   ['input/duan22050/Oboe.wav',
                    'input/duan22050/Euphonium.wav'],
                   sampdist=4,
                   plot_color=False,
                   tau=0.01
    )

def separate_urmp_03(seed):
    """
    Separate the piece from URMP with flute and clarinet.

    Parameters
    ----------
    seed : int
        Random seed for both NumPy and Tensorflow
    """

    separate_train(seed, seed, 'urmp/03',
                   'input/URMP/AuMix_03_Dance_fl_cl.wav',
                   ['input/URMP/AuSep_1_fl_03_Dance.wav',
                    'input/URMP/AuSep_2_cl_03_Dance.wav'])

def separate_urmp_03_fl(seed):
    """
    Represent the flute track from URMP (oracle).

    Parameters
    ----------
    seed : int
        Random seed for both NumPy and Tensorflow
    """

    separate_train(seed, seed, 'urmp/03-fl',
                   'input/URMP/AuSep_1_fl_03_Dance.wav',
                   ['input/URMP/AuSep_1_fl_03_Dance.wav'],
                   num_guesses=(3,))

def separate_urmp_03_cl(seed):
    """
    Represent the clarinet track from URMP (oracle).

    Parameters
    ----------
    seed : int
        Random seed for both NumPy and Tensorflow
    """

    separate_train(seed, seed, 'urmp/03-cl',
                   'input/URMP/AuSep_2_cl_03_Dance.wav',
                   ['input/URMP/AuSep_2_cl_03_Dance.wav'],
                   num_guesses=(3,))

def separate_urmp_03_oracle(seed):
    """
    Separate the piece from URMP with flute and clarinet with a fixed
    oracle dictionary.

    Parameters
    ----------
    seed : int
        Random seed for both NumPy and Tensorflow
    """

    separate_train(seed, seed, 'urmp/03-oracle',
                   'input/URMP/AuMix_03_Dance_fl_cl.wav',
                   ['input/URMP/AuSep_1_fl_03_Dance.wav',
                    'input/URMP/AuSep_2_cl_03_Dance.wav'],
                   stepsize_dict=0,
                   init_dict=[[9.9999094e-01, 2.8566617e-01, 1.3464449e-01,
                               6.2039699e-02, 3.1025456e-02, 1.4180289e-02,
                               6.4844196e-03, 4.3262830e-03, 3.6610016e-03,
                               2.7565733e-03, 1.6078090e-03, 1.5638472e-03,
                               1.2531099e-03, 8.7739876e-04, 8.1821729e-04,
                               8.1135111e-04, 1.0761940e-03, 1.0867843e-03,
                               9.5700449e-04, 7.1245024e-04, 6.6909962e-04,
                               9.3294185e-04, 1.2074142e-03, 1.2356122e-03,
                               1.1139465e-03],
                              [0.9999976 , 0.06495468, 0.5615128 , 0.08051097,
                               0.21722846, 0.07488991, 0.08725404, 0.04041256,
                               0.03574327, 0.02376241, 0.01887124, 0.01387317,
                               0.01072307, 0.00816069, 0.0074843 , 0.00694616,
                               0.00535791, 0.00544416, 0.00442878, 0.00364856,
                               0.00310717, 0.00326273, 0.00261571, 0.00225215,
                               0.00219441]])

def separate_urmp_03_init(seed):
    """
    Separate the piece from URMP with flute and clarinet with an initial
    oracle dictionary.

    Parameters
    ----------
    seed : int
        Random seed for both NumPy and Tensorflow
    """

    separate_train(seed, seed, 'urmp/03-init',
                   'input/URMP/AuMix_03_Dance_fl_cl.wav',
                   ['input/URMP/AuSep_1_fl_03_Dance.wav',
                    'input/URMP/AuSep_2_cl_03_Dance.wav'],
                   init_dict=[[9.9999094e-01, 2.8566617e-01, 1.3464449e-01,
                               6.2039699e-02, 3.1025456e-02, 1.4180289e-02,
                               6.4844196e-03, 4.3262830e-03, 3.6610016e-03,
                               2.7565733e-03, 1.6078090e-03, 1.5638472e-03,
                               1.2531099e-03, 8.7739876e-04, 8.1821729e-04,
                               8.1135111e-04, 1.0761940e-03, 1.0867843e-03,
                               9.5700449e-04, 7.1245024e-04, 6.6909962e-04,
                               9.3294185e-04, 1.2074142e-03, 1.2356122e-03,
                               1.1139465e-03],
                              [0.9999976 , 0.06495468, 0.5615128 , 0.08051097,
                               0.21722846, 0.07488991, 0.08725404, 0.04041256,
                               0.03574327, 0.02376241, 0.01887124, 0.01387317,
                               0.01072307, 0.00816069, 0.0074843 , 0.00694616,
                               0.00535791, 0.00544416, 0.00442878, 0.00364856,
                               0.00310717, 0.00326273, 0.00261571, 0.00225215,
                               0.00219441]])

def separate_urmp_09(seed):
    """
    Separate the piece from URMP with trumpet and violin.

    Parameters
    ----------
    seed : int
        Random seed for both NumPy and Tensorflow
    """

    separate_train(seed, seed, 'urmp/09',
                   'input/URMP/AuMix_09_Jesus_tpt_vn.wav',
                   ['input/URMP/AuSep_1_tpt_09_Jesus.wav',
                    'input/URMP/AuSep_2_vn_09_Jesus.wav'],
                   plot_color=None)

def separate_urmp_10(seed):
    """
    Separate the piece from URMP with trumpet and saxophone.

    Parameters
    ----------
    seed : int
        Random seed for both NumPy and Tensorflow
    """

    separate_train(seed, seed, 'urmp/10',
                   'input/URMP/AuMix_10_March_tpt_sax.wav',
                   ['input/URMP/AuSep_1_tpt_10_March.wav',
                    'input/URMP/AuSep_2_sax_10_March.wav'])

def separate_urmp_11(seed):
    """
    Separate the piece from URMP with oboe and violoncello.

    Parameters
    ----------
    seed : int
        Random seed for both NumPy and Tensorflow
    """

    separate_train(seed, seed, 'urmp/11',
                   'input/URMP/AuMix_11_Maria_ob_vc.wav',
                   ['input/URMP/AuSep_1_ob_11_Maria.wav',
                    'input/URMP/AuSep_2_vc_11_Maria.wav'])

def separate_urmp_11_ob(seed):
    """
    Represent the flute track from URMP (oracle).

    Parameters
    ----------
    seed : int
        Random seed for both NumPy and Tensorflow
    """

    separate_train(seed, seed, 'urmp/11-ob',
                   'input/URMP/AuSep_1_ob_11_Maria.wav',
                   ['input/URMP/AuSep_1_ob_11_Maria.wav'],
                   num_guesses=(3,),)

def separate_urmp_11_vc(seed):
    """
    Represent the violoncello track from URMP (oracle).

    Parameters
    ----------
    seed : int
        Random seed for both NumPy and Tensorflow
    """

    separate_train(seed, seed, 'urmp/11-vc',
                   'input/URMP/AuSep_2_vc_11_Maria.wav',
                   ['input/URMP/AuSep_2_vc_11_Maria.wav'],
                   num_guesses=(3,),)

def separate_urmp_11_oracle(seed):
    """
    Separate the piece from URMP with oboe and violoncello with a fixed
    oracle dictionary.

    Parameters
    ----------
    seed : int
        Random seed for both NumPy and Tensorflow
    """

    separate_train(seed, seed, 'urmp/11-oracle',
                   'input/URMP/AuMix_11_Maria_ob_vc.wav',
                   ['input/URMP/AuSep_1_ob_11_Maria.wav',
                    'input/URMP/AuSep_2_vc_11_Maria.wav'],
                   stepsize_dict=0,
                   init_dict=[[0.71527076, 0.15165104, 0.22934589, 0.07952369,
                               0.15811892, 1.0000482 , 0.05900749, 0.10863784,
                               0.3203387 , 0.07528263, 0.0558194 , 0.39714912,
                               0.06068192, 0.06662366, 0.18988474, 0.06253739,
                               0.02324584, 0.2287349 , 0.01589424, 0.02267853,
                               0.03540009, 0.02727244, 0.0439751 , 0.13116299,
                               0.01700303],
                              [9.6702337e-01, 1.0000093e+00, 2.4351594e-01,
                               8.5118622e-02, 4.3554008e-02, 1.6179238e-02,
                               4.7902358e-03, 2.9573606e-03, 1.7883319e-03,
                               1.6545599e-03, 1.4326506e-03, 1.2560592e-03,
                               1.0190095e-03, 9.4876578e-04, 9.1352483e-04,
                               9.6369127e-04, 9.7806891e-04, 8.9357258e-04,
                               6.8492559e-04, 4.8300021e-04, 3.8419233e-04,
                               2.7534369e-04, 1.9588925e-04, 3.3240172e-04,
                               1.1243264e-03]])

def separate_urmp_11_init(seed):
    """
    Separate the piece from URMP with oboe and violoncello with an initial
    oracle dictionary.

    Parameters
    ----------
    seed : int
        Random seed for both NumPy and Tensorflow
    """

    separate_train(seed, seed, 'urmp/11-init',
                   'input/URMP/AuMix_11_Maria_ob_vc.wav',
                   ['input/URMP/AuSep_1_ob_11_Maria.wav',
                    'input/URMP/AuSep_2_vc_11_Maria.wav'],
                   init_dict=[[0.71527076, 0.15165104, 0.22934589, 0.07952369,
                               0.15811892, 1.0000482 , 0.05900749, 0.10863784,
                               0.3203387 , 0.07528263, 0.0558194 , 0.39714912,
                               0.06068192, 0.06662366, 0.18988474, 0.06253739,
                               0.02324584, 0.2287349 , 0.01589424, 0.02267853,
                               0.03540009, 0.02727244, 0.0439751 , 0.13116299,
                               0.01700303],
                              [9.6702337e-01, 1.0000093e+00, 2.4351594e-01,
                               8.5118622e-02, 4.3554008e-02, 1.6179238e-02,
                               4.7902358e-03, 2.9573606e-03, 1.7883319e-03,
                               1.6545599e-03, 1.4326506e-03, 1.2560592e-03,
                               1.0190095e-03, 9.4876578e-04, 9.1352483e-04,
                               9.6369127e-04, 9.7806891e-04, 8.9357258e-04,
                               6.8492559e-04, 4.8300021e-04, 3.8419233e-04,
                               2.7534369e-04, 1.9588925e-04, 3.3240172e-04,
                               1.1243264e-03]])

if __name__ == '__main__':
    separate_mozart(0)
    #separate_mozart_cl(11)
    #separate_mozart_piano(13)
    #separate_duan_synth2(15)
    #separate_duan_synth3(11)
    #separate_duan_acous(15)
    #separate_urmp_03(15)
    #separate_urmp_03_fl(11)
    #separate_urmp_03_cl(10)
    #separate_urmp_03_oracle(13)
    #separate_urmp_03_init(13)
    #separate_urmp_09(15)
    #separate_urmp_10(15)
    #separate_urmp_11(15)
    #separate_urmp_11_ob(11)
    #separate_urmp_11_vc(11)
    #separate_urmp_11_oracle(13)
    #separate_urmp_11_init(13)
