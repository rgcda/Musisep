#!python3

"""
All the training mechanisms for blind separation via neural networks.
"""

import itertools
import types
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..audio import spect
from ..audio import wav
from ..audio import performance
from ..dictsep.__main__ import correct_signal_length
from .adamax import AdamaxDict

@tf.function
def gauss(x, mean, stdev):
    """
    Evaluate the Gaussian function.

    Parameters
    ----------
    x : tensor of float
        Points of evaluation
    mean : tensor of float
        Mean value(s)
    stdev : tensor of float
        Standard deviation(s)

    Returns
    -------
    spect : tensor of float
        Values of the Gaussian
    """

    return tf.exp(- (x - mean)**2 / (2 * stdev**2))

@tf.function
def inst_scale_raw(params, har_num, spectheight, sigmas_an):
    """
    Evaluate the linear-frequency spectra for the harmonics of tones,
    disregarding the amplitudes.

    Parameters
    ----------
    params : tensor of float
       Continous parameters for the tones, stacked along axis 2
    har_num : int
       Number of harmonics to evaluate
    spectheight : int
       Size of the output spectrum
    sigmas_an : float
       Number of standard deviations at which the analysis window is cut

    Returns
    -------
    spect : tensor of float
        Spectra of the harmonics
    """

    x = tf.range(spectheight, dtype=tf.float32)
    x = tf.reshape(x, [1, 1, -1, 1])
    amps, scales, sigmas, spreads = \
        trans_params(*tf.unstack(params, axis=2), sigmas_an)
    har_range = tf.reshape(tf.range(1, har_num + 1, dtype=tf.float32),
                           [1, 1, 1, -1])
    scales = tf.expand_dims(tf.expand_dims(scales, axis=2), axis=3)
    sigmas = tf.expand_dims(tf.expand_dims(sigmas, axis=2), axis=3)
    spreads = tf.expand_dims(tf.expand_dims(spreads, axis=2), axis=3)
    means = scales * har_range * tf.sqrt(1 + spreads * har_range**2)

    return gauss(x, means, sigmas)

@tf.function
def inst_scale(params, insts, inst_dict, har_coeffs, spectheight, sigmas_an):
    """
    Evaluate the linear-frequency spectra for tones.

    Parameters
    ----------
    params : tensor of float
       Continous parameters for the tones, stacked along axis 2
    insts : int
       Indices of the instruments playing the tones
    inst_dict : tensor of float
       Dictionary with the shape [instruments, harmonics]
    har_coeffs : tensor of float
       Relative complexe amplitudes of the harmonics
    spectheight : int
       Size of the output spectrum
    sigmas_an : float
       Number of standard deviations at which the analysis window is cut

    Returns
    -------
    spect : tensor of float
        Spectra of the tones
    """

    inst_num, har_num = tf.unstack(tf.shape(inst_dict))
    amps, scales, sigmas, spreads = tf.unstack(params, axis=2)
    amps = tf.math.abs(amps)
    tone_factors = tf.gather(inst_dict, insts)
    tone_factors = tf.einsum('bt,bth->bth', amps, tone_factors)
    tone_factors = tf.einsum('bth,bthc->bthc', tone_factors, har_coeffs)

    har_spects = inst_scale_raw(params, har_num, spectheight, sigmas_an)
    tone_spects = tf.einsum('bthc,btsh->btcs', tone_factors, har_spects)

    inst_mask = tf.where(tf.expand_dims(insts, 2)
                         == tf.reshape(tf.range(inst_num), [1, 1, -1]),
                         1., 0.)
    inst_spects = tf.einsum('bti,btcs->bics', inst_mask, tone_spects)

    return inst_spects

@tf.function
def inst_scale_post(amps, insts, inst_dict, inst_num, har_coeffs, har_spect):
    """
    Combine the spectra of individual harmonics into tone spectra.

    Parameters
    ----------
    amps : tensor of float
       Amplitudes of the tones
    insts : int
       Indices of the instruments playing the tones
    inst_dict : tensor of float
       Dictionary with the shape [instruments, harmonics]
    inst_num : int
       Number of instruments available
    har_coeffs : tensor of float
       Relative complexe amplitudes of the harmonics
    har_spect : tensor of float
       Spectra of the individual harmonics (without amplitudes)

    Returns
    -------
    spect : tensor of float
        Spectra of the tones
    """

    if inst_dict is not None:
        har_coeffs = tf.einsum('bghc,bgh->bghc', har_coeffs,
                               tf.gather(inst_dict, insts))
    if amps is not None:
        har_coeffs = tf.einsum('bghc,bg->bghc', har_coeffs,
                               tf.math.abs(amps))

    tone_spects = tf.einsum('bghc,bgsh->bgcs', har_coeffs, har_spect)

    # for many tones, scatter_nd might be cheaper
    inst_mask = tf.where(tf.expand_dims(insts, 2)
                         == tf.reshape(tf.range(inst_num), [1, 1, -1]),
                         1., 0.)
    inst_spects = tf.einsum('bgi,bgcs->bgics', inst_mask, tone_spects)

    return inst_spects

@tf.function
def trans_params(amps, scales, sigmas, spreads, sigmas_an):
    """
    Apply transformations on instrument parameters to ensure their validity.

    Parameters
    ----------
    amps : tensor of float
        Amplitudes of the tones
    scales : tensor of float
        Natural fundamental frequencies of the tones
    sigmas : tensor of float
        Widths of the Gaussians
    spreads : tensor of float
        Inharmonicities of the tones

    Returns
    -------
    amps : tensor of float
        Amplitudes of the tones
    scales : tensor of float
        Natural fundamental frequencies of the tones
    sigmas : tensor of float
        Widths of the Gaussians
    spreads : tensor of float
        Inharmonicities of the tones
    """
    sigmas = (tf.math.softplus(tf.math.maximum(-3., sigmas))
              * sigmas_an / np.pi / np.log(2))
    spreads = tf.math.abs(spreads)

    return amps, scales, sigmas, spreads

class SpectLoss:
    """
    Container object for the spectra and losses for the individual
    tones of the instruments.

    Parameters
    ----------
    batch_size : int
        Batch size of all the data structures
    num_guesses_prod : int
        Total number of all the samples per spectrum
    inst_num : int
        Number of instruments in the sample
    spectheight : int
       Size of the input/output spectrum
    """

    def __init__(self, batch_size, num_guesses_prod, inst_num, spectheight):
        self.spects = types.SimpleNamespace()
        self.losses = types.SimpleNamespace()

        self.spects.insts = tf.zeros(
            [batch_size, num_guesses_prod, inst_num, 2, spectheight])
        self.spects.insts_sparse = tf.zeros(
            [batch_size, num_guesses_prod, inst_num, 2, spectheight])
        self.spects.insts_dir = tf.zeros(
            [batch_size, num_guesses_prod, inst_num, 2, spectheight])
        self.spects.mix = None
        self.spects.mix_sparse = None
        self.spects.mix_dir = None

    def add_tone(self, params, har_coeffs, on_factors, insts, har_spect,
                 inst_dict, orig_spect):
        """
        Add the results of a new tone to the object

        Parameters
        ----------
        params : tensor of float
            Instrument parameters for the tone
        har_coeffs : tensor of float
            Relative complexe amplitudes of the harmonics
        on_factors : tensor of int
            Binary indicator if a tone contributes to the sparse prediction
        insts : tensor of int
            Indices of the instruments playing the tones
        har_spect : tensor of float
            Spectra of the individual harmonics (without amplitudes)
        inst_dict : tensor of float
            Dictionary with the shape [instruments, harmonics]
        orig_spect : tensor of float
            Original input sampled spectrum
        """

        loss_fun = lifted_l2_cmplx

        orig_spect = tf.expand_dims(orig_spect, 1)
        phase_coeffs = (complex_arg(har_coeffs, axis=3))
        phase_coeffs_sparse = tf.einsum('bg,bghc->bghc',
                                        on_factors, phase_coeffs)

        amps, scales, sigmas, spreads = tf.unstack(params, axis=2)
        inst_num = tf.shape(inst_dict)[0]

        self.spects.insts += inst_scale_post(
            amps, insts, inst_dict, inst_num, phase_coeffs, har_spect)
        self.spects.insts_sparse += inst_scale_post(
            amps, insts, inst_dict, inst_num, phase_coeffs_sparse, har_spect)
        self.spects.insts_dir += inst_scale_post(
            None, insts, None, inst_num, har_coeffs, har_spect)

        self.spects.mix = mix_inst_spects(
            self.spects.insts, axis=2)
        self.spects.mix_sparse = mix_inst_spects(
            self.spects.insts_sparse, axis=2)
        self.spects.mix_dir = mix_inst_spects(
            self.spects.insts_dir, axis=2)

        self.losses.mix = tf.reduce_mean(
            lifted_l2_abs(orig_spect, self.spects.mix, axis=2),
            axis=2)
        self.losses.mix_sparse = tf.reduce_mean(
            lifted_l2_abs(orig_spect, self.spects.mix_sparse, axis=2),
            axis=2)
        self.losses.mix_dir = tf.reduce_mean(
            loss_fun(orig_spect, self.spects.mix_dir, axis=2),
            axis=2)
        self.losses.inst = tf.reduce_mean(
            loss_fun(self.spects.insts_dir, self.spects.insts, axis=3),
            axis=[2, 3])

def comp_total_loss(losses, on_factors, spl, loss_coeffs):
    """
    Compute the linear combination of losses

    Parameters
    ----------
    losses : SimpleNamespace
        The individual loss values
    on_factors : tensor of int
        Binary indicator if a tone contributes to the sparse prediction
    spl : float
        Discount factor for the sparsity
    loss_coeffs : array_like
        Linear weights for the loss terms

    Returns
    -------
    total_loss : tensor of float
        Computed total loss
    discounts : tensor of float
        Sparsity discount factors
    """

    discounts = spl ** tf.reduce_sum(1 - on_factors, axis=2)

    return (losses.mix * loss_coeffs[0]
            + losses.mix_sparse * discounts * loss_coeffs[1]
            + losses.inst * loss_coeffs[2]
            + losses.mix_dir * loss_coeffs[3],
            discounts)

@tf.function
def add_linspace(x):
    """
    Add a linear range layer to a CNN tensor

    Parameters
    ----------
    x : tensor of float
        Layer to add the linear range to

    Returns
    -------
    model_in : tensor of float
        Input layer with the linear range added
    """

    shape = tf.shape(x)

    model_in = tf.concat(
        [x,
         tf.broadcast_to(
             tf.reshape(tf.linspace(1e-2, 0, shape[2]), [1, 1, shape[2], 1]),
             [shape[0], 1, shape[2], 1])],
        axis=1)

    return model_in

def unet(x, inst_num, spectheight):
    """
    Create the U-Net as a Keras model.

    Parameters
    ----------
    inst_num : int
        Number of instruments expected in the sample
    spectheight : int
        Size of the input/output spectrum

    Returns
    -------
    y : tensor of float
        Network output
    """

    y = x
    w = 80
    nonlin = tf.nn.relu

    # Keras, y u no Conv1Transpose?
    y = tf.keras.layers.Lambda(lambda w: tf.expand_dims(w, 3))(y)
    layers = [y]

    reductions = [4, 4, 4, 4, 4, 3, 2]
    channels = [80, 160, 240, 320, 400, 480, 560]

    for i in range(len(reductions)):
        if i == 0:
            y = tf.keras.layers.Lambda(
                add_linspace,
                lambda s: (s[0], s[1] + 1, s[2], s[3]))(y)
        y = tf.keras.layers.Conv2D(channels[i], (5, 1),
                                   data_format='channels_first',
                                   padding='same',
                                   strides=(reductions[i], 1),
                                   use_bias=True)(y)
        y = tf.keras.layers.Activation(nonlin)(y)
        layers.append(y)

    for i in reversed(range(len(reductions))):
        y = tf.keras.layers.Conv2DTranspose(channels[i], (5, 1),
                                            data_format='channels_first',
                                            padding='same',
                                            strides=(reductions[i], 1),
                                            use_bias=True)(y)
        y = tf.keras.layers.Activation(nonlin)(y)
        y = tf.keras.layers.concatenate([y, layers[i]], axis=1)

    y = tf.keras.layers.Conv2D(w, (3, 1),
                               data_format='channels_first',
                               padding='same',
                               use_bias=True)(y)
    y = tf.keras.layers.Activation(tf.nn.relu)(y)
    y = tf.keras.layers.Conv2D(w, (1, 1),
                               data_format='channels_first',
                               padding='same',
                               use_bias=True)(y)
    y = tf.keras.layers.Activation(tf.nn.relu)(y)

    out_dim = 9
    y = tf.keras.layers.Conv2D(inst_num * out_dim, (1, 1),
                               data_format='channels_first',
                               padding='same',
                               use_bias=False)(y)
    y = tf.keras.layers.Reshape((inst_num, out_dim, spectheight))(y)

    return y

@tf.function
def lift_spect(x, shift=1e-7, qexp=0.5):
    """
    Lift a positive-valued spectrum via a concave power function.

    Parameters
    ----------
    x : tensor of float
        Spectrum
    shift : float
        Additive constant to keep the transform differentiable
    qexp : float
        Exponent of the power function

    Returns
    -------
    The lifted spectrum
    """

    return (x + shift)**qexp - shift**qexp

@tf.function
def lift_spect_sign(x, shift=1e-7, qexp=0.5):
    """
    Lift a real-valued spectrum via a concave power function.

    Parameters
    ----------
    x : tensor of float
        Spectrum
    shift : float
        Additive constant to keep the transform differentiable
    qexp : float
        Exponent of the power function

    Returns
    -------
    The lifted spectrum
    """

    return ((tf.math.abs(x) + shift)**qexp - shift**qexp) * tf.math.sign(x)

@tf.function
def lift_cmplx(x, axis, shift=1e-7, qexp=0.5):
    """
    Lift a complex-valued spectrum via a concave power function.

    Parameters
    ----------
    x : tensor of float
        Spectrum
    shift : float
        Additive constant to keep the transform differentiable
    qexp : float
        Exponent of the power function

    Returns
    -------
    The lifted spectrum
    """

    spect_abs = lift_spect(complex_abs(x, axis), shift, qexp)
    spect_arg = complex_arg(x, axis)

    return spect_abs * spect_arg

@tf.function
def lifted_l2_cmplx(x, y, axis):
    """
    Radially symmetric lifted l2 distance between two spectra.

    Parameters
    ----------
    x : tensor of float
        First spectrum
    y : tensor of float
        Second spectrum
    axis : int
        Complex axis

    Returns
    -------
    l2 loss
    """

    return tf.reduce_sum((lift_cmplx(x, axis)
                          - lift_cmplx(y, axis))**2,
                         axis=axis)

@tf.function
def lifted_l2_abs(x, y, axis):
    """
    Radially symmetric lifted l2 distance between two spectra.

    Parameters
    ----------
    x : tensor of float
        First spectrum
    y : tensor of float
        Second spectrum
    axis : int
        Complex axis

    Returns
    -------
    l2 loss
    """

    return tf.reduce_sum((lift_spect(complex_abs(x, axis))
                          - lift_spect(complex_abs(y, axis)))**2,
                         axis=axis)

@tf.function
def mix_inst_spects(inst_spects, axis):
    """
    Combine the spectra for multiple instruments, dropping the summation axis.

    Parameters
    ----------
    inst_spects : tensor of float
        Spectra for the individual instruments
    axis : int
        Summation axis

    Returns
    -------
    Mixture spectrum
    """

    return tf.reduce_sum(inst_spects, axis=axis)

@tf.function
def complex_abs(spect, axis):
    """
    Compute the absolute value of a complex tensor.

    Parameters
    ----------
    spect : tensor of float
        Real-valued tensor with a complex axis
    axis : int
        Complex axis

    Returns
    -------
    Absolute value of the input tensor (with complex axis shrunken to size 1)
    """

    tf.debugging.assert_equal(tf.shape(spect)[axis], 2)
    real_abs = tf.math.abs(tf.gather(spect, [0], axis=axis))
    imag_abs = tf.math.abs(tf.gather(spect, [1], axis=axis))

    min_abs = tf.math.minimum(real_abs, imag_abs)
    max_abs = tf.math.maximum(real_abs, imag_abs)

    retval = max_abs * tf.sqrt(1 + (min_abs / tf.maximum(max_abs, 1e-20))**2)

    return retval

@tf.function
def complex_arg(spect, axis, bias=1e-20):
    """
    Compute the argument of a complex tensor.

    Parameters
    ----------
    spect : tensor of float
        Real-valued tensor with a complex axis
    axis : int
        Complex axis
    bias : float
        Offset to avoid division by 0

    Returns
    -------
    Tensor normalized to an absolute value of 1
    """

    real_part = tf.gather(spect, [0], axis=axis)
    imag_part = tf.gather(spect, [1], axis=axis)

    spect_sp = tf.concat([real_part, imag_part], axis=axis)
    spect_sp_abs = complex_abs(spect_sp, axis)

    return spect_sp / (spect_sp_abs + bias)

def plot_spectrum(filename, spectheight, *spects):
    """
    Plot real-valued spectra to a file

    Parameters
    ----------
    filename : string
        Name of the file to save the figure to
    spectheight : int
        Size of the output spectra
    spects : sequence of array_like of float
        Spectra to plot
    """

    for sp in spects:
        plt.plot(np.arange(spectheight), sp)
    plt.savefig(filename)
    plt.clf()

@tf.function
def sample_multi(pdf):
    """
    Sample batch-wise from a categorical distribution.

    Parameters
    ----------
    pdf : tensor of float
        Log probabilies of [batch, insts, scales]

    Returns
    -------
    insts : tensor of int
        Indices of the sampled instruments
    scales : tensor of int
        Discrete sampled frequencies
    """

    pdf_shape = tf.shape(pdf)
    pdf = tf.reshape(pdf, [pdf_shape[0], pdf_shape[1] * pdf_shape[2]])
    samples = tf.random.categorical(pdf, 1, dtype=tf.int32)
    samples = tf.stop_gradient(tf.squeeze(samples, axis=1))
    insts = samples // pdf_shape[2]
    scales = samples % pdf_shape[2]

    return insts, scales

@tf.function
def sample_multi_max(pdf):
    """
    Pick the mode batch-wise from a categorical distribution.

    Parameters
    ----------
    pdf : tensor of float
        Log probabilies of [batch, insts, scales]

    Returns
    -------
    insts : tensor of int
        Indices of the selected instruments
    scales : tensor of int
        Discrete selected frequencies
    """

    pdf_shape = tf.shape(pdf)
    pdf = tf.reshape(pdf, [pdf_shape[0], pdf_shape[1] * pdf_shape[2]])
    samples = tf.math.argmax(pdf, axis=1, output_type=tf.int32)
    insts = samples // pdf_shape[2]
    scales = samples % pdf_shape[2]

    return insts, scales

@tf.function
def norm_pdf(pdf):
    """
    Normalize a categorical distribution batch-wise via softmax.

    Parameters
    ----------
    pdf : tensor of float
        Log probabilies of [batch, insts, scales]

    Returns
    -------
    Normalized log probabilities
    """

    pdf_shape = tf.shape(pdf)
    pdf = tf.reshape(pdf, [pdf_shape[0], pdf_shape[1] * pdf_shape[2]])
    pdf = tf.nn.log_softmax(pdf, axis=1)
    pdf = tf.reshape(pdf, pdf_shape)

    return pdf

@tf.function
def gamma_probs(spreads, spreads_a, spreads_b):
    """
    Evaluate the gamma distribution

    Parameters
    ----------
    spreads : tensor of float
        Values where to evaluate
    spreads_a : tensor of float
        "alpha" parameter of the distribution
    spreads_b : tensor of float
        "beta" parameter of the distribution

    Returns
    -------
    Log probabilities
    """

    g = tfp.distributions.Gamma(spreads_a, spreads_b, allow_nan_stats=False)

    return g.log_prob(spreads)

@tf.function
def lsq_stock(har_spects, samp):
    """
    Solve a regularized least-squares system.

    Parameters
    ----------
    har_spects : tensor of float
        Spectra of the individual harmonics (without amplitudes)
    samp : tensor of float
        Direct prediction

    Returns
    -------
    Phase values for the harmonics
    """

    retval = tf.linalg.lstsq(har_spects, samp, 1e-3)

    return retval

class ParamsDict:
    """
    Container object for the model parameters.

    Parameters
    ----------
    batch_size : int
        Batch size of all the data structures
    num_guesses_prod : int
        Total number of all the samples per spectrum
    """

    _keys = ('probs', 'probs_samp', 'g_probs',
             'on_probs', 'on_probs_samp', 'on_factors')
    _special_keys = ('inst_params', 'insts', 'reg', 'entropy')

    def __init__(self, batch_size, num_guesses_prod):
        for key in self._keys:
            setattr(self, key, tf.zeros([batch_size, num_guesses_prod, 0]))
        self.inst_params = tf.zeros([batch_size, num_guesses_prod, 0, 4])
        self.insts = tf.zeros([batch_size, num_guesses_prod, 0],
                              dtype=tf.int32)
        self.reg = tf.zeros([0])
        self.entropy = tf.zeros([batch_size, 0])

    def _append(self, new):
        """
        Add parameters from a new tone to the object.

        Parameters
        ----------
        new : object
            Object with tone parameters
        """
        for key in self._keys + ('inst_params', 'insts'):
            setattr(self, key, tf.concat(
                [getattr(self, key),
                 tf.expand_dims(getattr(new, key), 2)],
                axis=2))
        self.entropy = tf.concat(
            [self.entropy, tf.expand_dims(new.entropy, 1)], axis=1)

    def _sum(self):
        """
        Sum up all the parameters from the collected tones.

        Returns
        -------
        params_sum : SimpleNamespace
            Sum of the parameters
        """

        params_sum = types.SimpleNamespace()
        for key in self._keys + ('inst_params', 'insts'):
            setattr(params_sum, key, tf.reduce_sum(
                getattr(self, key), axis=2))
        for key in ('reg', 'entropy'):
            setattr(params_sum, key, tf.reduce_sum(
                getattr(self, key), axis=0))

        return params_sum

class SpectvisDict:
    """
    Container object to visualize spectra.

    Parameters
    ----------
    spectheight : int
        Size of the spectrum
    """

    _keys = ('pdf', 'on_probs', 'amps', 'sigmas', 'offsets',
             'spreads_a', 'spreads_b')

    def __init__(self, spectheight):
        for key in self._keys:
            setattr(self, key, tf.zeros([0, spectheight]))

    def _append(self, new):
        """
        Add spectra from a new tone to the object.

        Parameters
        ----------
        new : object
            Object with tone parameters
        """

        for key in self._keys:
            setattr(self, key, tf.concat(
                [getattr(self, key), getattr(new, key)[0, :, :]],
                axis=0))

class Trainer:
    """
    Object containing all the data necessary for the training.

    Parameters
    ----------
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
    sampdist : int
        Time interval of the spectrogram
    sub_factor : int
        Factor by which to subsample the spectrogram for resynthesis
    sigmas_an : float
       Number of standard deviations at which the analysis window is cut
    plot_color : string or NoneType
       Whether to make a color plot
    save_points: sequence of int
       Iterations at which to save the output
    init_dict : tensor of float
       Dictionary with the shape [instruments, harmonics]
    """

    def __init__(self, name, mixed_soundfile, orig_soundfiles, loss_coeffs,
                 har_num, num_guesses, spl, batch_size, batch_size_pred,
                 virt_batch_mul, stepsize_net, stepsize_dict, tau,
                 sampdist, sub_factor, sigmas_an, plot_color, save_points,
                 init_dict):
        self.loss_coeffs = loss_coeffs
        self.har_num = har_num
        self.num_guesses = num_guesses
        self.spl = spl
        self.batch_size = batch_size
        self.batch_size_pred = batch_size_pred
        self.virt_batch_mul = virt_batch_mul
        self.tau = tau
        self.plot_color = plot_color
        self.save_points = save_points

        self.inst_num = len(self.num_guesses)
        self.spectheight = 6144
        self.sampdist = sampdist
        self.sigmas_an = sigmas_an
        self.name = name
        self.log_dir = 'logs-nn/' + name

        self.param_scales = tf.Variable(
            np.broadcast_to(np.log([10000, 1, 3, 1, 30, 30, 1, 1, 1000]),
                            [self.inst_num, 9]),
            dtype=tf.float32, name="param_scales")

        self.inst_model = self.make_model(self.inst_num * 4 + 6)
        self.variables_cnn = self.inst_model.trainable_weights
        self.variables_out = None
        self.variables_trans = None
        self.gradients_model_cnn = None
        self.gradients_model_out = None
        self.gradient_dict = None
        self.gradient_scales = None

        if init_dict is None:
            self.inst_dict = tf.Variable(
                np.asarray(
                    np.maximum(
                        1e-3,
                        np.expand_dims(0.5 / np.arange(1, self.inst_num + 1),
                                       axis=1) ** np.arange(0, self.har_num)),
                    dtype=np.float32),
                name="inst_dict")
        else:
            self.inst_dict = tf.Variable(
                np.asarray(init_dict, dtype=np.float32),
                name="inst_dict")

        print(self.inst_dict)

        self.save_module = tf.Module()
        self.save_module.inst_dict = self.inst_dict
        self.save_module.inst_model = self.inst_model
        self.save_module.param_scales = self.param_scales

        self.file_writer = tf.summary.create_file_writer(
            self.log_dir, flush_millis=30000)

        self.signal, self.samprate = wav.read(mixed_soundfile)
        print("samprate", self.samprate)
        self.sigmas_an *= (48000 / self.samprate)
        self.orig_signals = np.stack(
            [correct_signal_length(wav.read(os)[0], self.signal.size)
             for os in orig_soundfiles], axis=0)

        self.sign_mask = np.tile([1, -1], self.spectheight // 2)
        spectrogram = spect.stft(
            self.signal, self.spectheight, self.sigmas_an, self.sampdist
        )[:self.spectheight, :].T
        spectrogram *= self.sign_mask
        self.spectlen = spectrogram.shape[0]
        self.sub_factor = sub_factor
        self.spectlen_sub = (self.spectlen - 1) // self.sub_factor + 1
        self.spect_data = tf.data.Dataset.from_tensor_slices(
            tf.convert_to_tensor(
                np.stack([np.real(spectrogram), np.imag(spectrogram)],
                         axis=1),
                dtype=tf.float32))
        self.spect_data_sub = self.spect_data.shard(self.sub_factor, 0)
        print("spectlen", self.spectlen)
        self.random_slice = iter(
            self.spect_data.shuffle(self.spectlen)
            .repeat().batch(self.batch_size, drop_remainder=True))
        self.inline_slices = self.spect_data_sub.padded_batch(
            self.batch_size_pred)

        self.eval_writer = tf.summary.create_file_writer(
            self.log_dir + '-eval', flush_millis=30000)

        self.measures_table = []

        self.optimizer = tf.keras.optimizers.Adamax(
            stepsize_net, 0.99, 0.999)
        self.dict_optimizer = AdamaxDict(
            stepsize_dict, 0.99, 0.999, axis=1)

    def load(self, module):
        """
        Load dictionary, output scalings, and the model from a module.

        module : tf.Module
            Module with the saved data
        """

        self.inst_dict = module.inst_dict
        self.param_scales = module.param_scales
        self.inst_model = module.inst_model
        self.variables_cnn = self.inst_model.trainable_variables

    def make_model(self, input_sizes):
        """
        Construct a Keras model for the parameter prediction

        Parameters
        ----------
        input_sizes : int
            Total number of input channels for the network

        Returns
        -------
        The neural network as a Keras model
        """

        spect_in = tf.keras.Input(shape=(input_sizes, self.spectheight))
        pdf_out = unet(spect_in, self.inst_num, self.spectheight)
        model_cnn = tf.keras.Model(inputs=spect_in, outputs=pdf_out)

        return model_cnn

    def separate_inst(self, model, model_in, training, batch_factor,
                      fan_factor, num_guesses, inst_mask, tau, batch_size):
        """
        Identify the parameters for one tone in a given spectrum

        Parameters
        ----------
        model : tf.Keras.Model
            Definition of the neural network
        model_in : tensor of float
            Input channels that the neural network receives
        training : bool
            Whether to perform training
        batch_factor : int
            Product of the number of samples for previous tones
        fan_factor : int
            Product of the number of samples for current and future tones
        num_guesses : int
            Number of samples for the current tone
        inst_mask : tensor of int
            1 for instruments that have already played a tone, 0 otherwise
        tau : float
            Exponent to control exploration
        batch_size : int
            Batch size for training

        Returns
        -------
        params_tone : SimpleNamespace
            All parameters relating to an identified tone
        params_spect : SimpleNamespace
            Unsampled parameters in the dimensionality of the spectrum
        har_spects_raw : tensor of float
            Spectra of the harmonics
        """

        model_in_shape = tf.shape(model_in)
        model_in = tf.reshape(model_in, [batch_size * batch_factor,
                                         model_in_shape[2], self.spectheight])
        model_out = model(model_in, training)
        model_out = tf.einsum('bips,ip->bips', model_out,
                              tf.math.exp(self.param_scales))
        out_p = types.SimpleNamespace()
        (out_p.pdf, out_p.amps, out_p.sigmas, out_p.offsets,
         out_p.spreads_a, out_p.spreads_b, out_p.dir_r, out_p.dir_i,
         out_p.on_probs) = tf.unstack(model_out, axis=2)

        num_guesses_prod = np.prod(num_guesses, dtype=np.int32)
        shape = [batch_size, batch_factor * fan_factor]

        pdf_raw = tf.reshape(norm_pdf(out_p.pdf),
                             [batch_size, batch_factor,
                              self.inst_num, self.spectheight])
        entropy = tf.reduce_mean(pdf_raw, axis=[2,3])
        entropy = - fan_factor * tf.reduce_sum(entropy, axis=1)

        inst_mask_shape = tf.shape(inst_mask)
        inst_mask = tf.reshape(inst_mask, [batch_size * batch_factor,
                                           inst_mask_shape[2], 1])
        pdf_masked = tf.where(inst_mask == 0, out_p.pdf, - np.infty)
        pdf_norm = norm_pdf(pdf_masked)

        if tau is not None:
            tau_range = tf.exp(tf.linspace(0., tf.math.log(tau),
                                           num_guesses[0]))
            tau_range = tf.keras.backend.repeat_elements(
                tau_range, rep=np.prod(num_guesses[1:], dtype=np.int32),
                axis=0)
            tau_range = tf.reshape(tau_range, [1, num_guesses_prod])
            tau_range = tf.tile(tau_range,
                                [batch_size * batch_factor
                                 * fan_factor // num_guesses_prod, 1])
            tau_range = tf.reshape(tau_range,
                                   [batch_size * batch_factor, fan_factor])

            pdf_samp = tf.einsum('bis,bg->bgis', pdf_norm, tau_range)
            pdf_samp = tf.reshape(pdf_samp,
                                  [batch_size * batch_factor * fan_factor,
                                   self.inst_num, self.spectheight])
            pdf_samp = norm_pdf(pdf_samp)

        params_spect = types.SimpleNamespace()
        params_spect.pdf = pdf_norm
        params_spect.on_probs = out_p.on_probs + 500
        params_spect.amps = out_p.amps
        params_spect.sigmas = out_p.sigmas
        params_spect.offsets = tf.math.tanh(out_p.offsets) * 5
        params_spect.spreads_a = tf.math.exp(
            tf.clip_by_value(out_p.spreads_a, -30, 30))
        params_spect.spreads_b = tf.math.exp(
            tf.clip_by_value(out_p.spreads_b, -30, 30)) * 1000

        if tau is None:
            insts, scales = sample_multi_max(pdf_norm)
        else:
            insts, scales = sample_multi(pdf_samp)

        insts = tf.reshape(insts, [batch_size * batch_factor, fan_factor])
        scales = tf.reshape(scales, [batch_size * batch_factor, fan_factor])
        idcs = tf.stack([insts, scales], axis=2)
        probs = tf.reshape(tf.gather_nd(pdf_norm, idcs, batch_dims=1), shape)

        if tau is None:
            probs_samp = tf.zeros(shape)
        else:
            pdf_samp = tf.reshape(pdf_samp,
                                  [batch_size * batch_factor, fan_factor,
                                   self.inst_num, self.spectheight])
            probs_samp = tf.reshape(
                tf.gather_nd(pdf_samp,
                             tf.stack([tf.broadcast_to(
                                 tf.range(fan_factor),
                                 [batch_size * batch_factor, fan_factor]),
                                       insts, scales], axis=2),
                             batch_dims=1), shape)
        amps = tf.reshape(
            tf.gather_nd(params_spect.amps, idcs, batch_dims=1), shape)
        sigmas = tf.reshape(
            tf.gather_nd(params_spect.sigmas, idcs, batch_dims=1), shape)
        offsets = tf.reshape(
            tf.gather_nd(params_spect.offsets, idcs, batch_dims=1), shape)

        on_probs = tf.gather_nd(params_spect.on_probs, idcs, batch_dims=1)
        if tau is not None:
            on_probs_samp = tf.einsum('bg,bg->bg', on_probs, tau_range)
            on_probs_samp = tf.reshape(on_probs_samp, shape)
        else:
            on_probs_samp = tf.zeros(shape)
        on_probs = tf.reshape(on_probs, shape)

        if tau is not None:
            on_factors = tf.stop_gradient(tf.math.sign(tf.nn.relu(
                tf.math.sigmoid(on_probs_samp)
                - tf.random.uniform(tf.shape(on_probs)))))
        else:
            on_factors = tf.stop_gradient(tf.math.sign(tf.nn.relu(
                tf.math.sigmoid(on_probs) - 0.5)))

        on_probs = (
            tf.math.log_sigmoid(on_probs) * on_factors
            + tf.math.log_sigmoid(-on_probs) * (1 - on_factors))
        on_probs_samp = (
            tf.math.log_sigmoid(on_probs_samp) * on_factors
            + tf.math.log_sigmoid(- on_probs_samp) * (1 - on_factors))

        spreads_a = tf.reshape(
            tf.gather_nd(params_spect.spreads_a, idcs, batch_dims=1), shape)
        spreads_b = tf.reshape(
            tf.gather_nd(params_spect.spreads_b, idcs, batch_dims=1), shape)
        if training:
            spreads = tf.stop_gradient(
                tf.random.gamma([], spreads_a, spreads_b))
        else:
            spreads = tf.math.maximum(spreads_a - 1, 0) / spreads_b
        g_probs = gamma_probs(spreads, spreads_a, spreads_b)
        scales = tf.reshape(tf.cast(scales, tf.float32), shape) + offsets
        inst_params = tf.stack([amps, scales, sigmas, spreads], axis=2)

        dir_r = tf.reshape(tf.gather(out_p.dir_r, insts, batch_dims=1),
                           [batch_size * batch_factor * fan_factor,
                            self.spectheight])
        dir_i = tf.reshape(tf.gather(out_p.dir_i, insts, batch_dims=1),
                           [batch_size * batch_factor * fan_factor,
                            self.spectheight])
        dir_c = tf.stack([dir_r, dir_i], axis=2)

        insts = tf.reshape(insts, shape)

        har_spects_raw = inst_scale_raw(
            inst_params, self.har_num, self.spectheight, self.sigmas_an)
        har_spects = tf.reshape(har_spects_raw,
                                [batch_size * batch_factor * fan_factor,
                                 self.spectheight, self.har_num])

        coeffs = tf.reshape(lsq_stock(har_spects, dir_c),
                            [batch_size, batch_factor * fan_factor,
                             self.har_num, 1, 2])
        har_coeffs, = tf.unstack(coeffs, axis=3)

        params_tone = types.SimpleNamespace(
            probs=probs, probs_samp=probs_samp, g_probs=g_probs,
            on_probs=on_probs, on_probs_samp=on_probs_samp,
            on_factors=on_factors, inst_params=inst_params,
            insts=insts, har_coeffs=har_coeffs, entropy=entropy)

        return params_tone, params_spect, har_spects_raw

    def separate(self, mix_spect_in, tau, batch_size, predict=False):
        """
        Identify all the parameters for the tones in the spectrum.

        Parameters
        ----------
        mix_spect_in : tensor of float
            Mixture spectrogram to perform the separation on
        tau : float
            Exponent to control exploration
        batch_size : int
            Batch size for training
        predict : bool
            Whether to go into prediction mode instead of training

        Results
        -------
        spects : SimpleNamespace
            Spectra related to the mixture
        losses : SimpleNamespace
            Losses related to the mixture
        params_d : ParamsDict
            Parameters related to the mixture
        spectvis_d : SpectvisDict
            Visualization spectra related to the mixture
        """

        if predict:
            num_guesses = [1] * self.inst_num
        else:
            num_guesses = self.num_guesses

        num_guesses_prod = np.product(num_guesses, dtype=np.int32)

        params_d = ParamsDict(batch_size, num_guesses_prod)
        spectvis_d = SpectvisDict(self.spectheight)

        constant_in = tf.reshape(mix_spect_in,
                                 [batch_size, 1, 2, self.spectheight])
        constant_in_abs = complex_abs(constant_in, axis=2)
        inst_spects_flat = np.zeros([batch_size, 1, self.inst_num * 2,
                                     self.spectheight])
        model_in = tf.concat([constant_in, constant_in_abs, inst_spects_flat,
                              constant_in, constant_in_abs, inst_spects_flat],
                             axis=2)

        inst_mask = tf.zeros([batch_size, 1, self.inst_num], dtype=tf.int32)

        spectloss = SpectLoss(batch_size, num_guesses_prod,
                              self.inst_num, self.spectheight)

        for i in range(self.inst_num):
            if i == 0:
                batch_factor = 1
                fan_factor = num_guesses_prod
            else:
                batch_factor = num_guesses_prod
                fan_factor = 1

            params_tone, spectvis_tone, har_spect = \
                self.separate_inst(self.inst_model, model_in, not predict,
                                   batch_factor, fan_factor, num_guesses[i:],
                                   inst_mask, tau, batch_size)

            spectvis_d._append(spectvis_tone)
            params_d._append(params_tone)

            spectloss.add_tone(params_tone.inst_params, params_tone.har_coeffs,
                               params_tone.on_factors, params_tone.insts,
                               har_spect, self.inst_dict, mix_spect_in)
            spects = spectloss.spects
            losses = spectloss.losses

            idcs = tf.stack(
                [tf.tile(tf.reshape(tf.range(batch_size), [batch_size, 1, 1]),
                         [1, batch_factor * fan_factor, i+1]),
                 tf.tile(tf.reshape(tf.range(batch_factor * fan_factor,
                                             dtype=tf.int32),
                                    [1, batch_factor * fan_factor, 1]),
                         [batch_size, 1, i+1]),
                 params_d.insts],
                axis=3)
            inst_mask = tf.scatter_nd(
                idcs,
                tf.broadcast_to(1, [batch_size,
                                    batch_factor * fan_factor, i+1]),
                [batch_size, batch_factor * fan_factor, self.inst_num])

            if i+1 < self.inst_num:
                constant_in = (
                    tf.reshape(mix_spect_in,
                               [batch_size, 1, 2, self.spectheight])
                    - tf.reshape(spects.mix,
                                 [batch_size, batch_factor * fan_factor,
                                  2, self.spectheight]))
                constant_in_abs = complex_abs(constant_in, axis=2)
                inst_spects_flat = tf.reshape(
                    spects.insts,
                    [batch_size, batch_factor * fan_factor,
                     self.inst_num * 2, self.spectheight])
                constant_in_dir = (
                    tf.reshape(mix_spect_in,
                               [batch_size, 1, 2, self.spectheight])
                    - tf.reshape(spects.mix_dir,
                                 [batch_size, batch_factor * fan_factor,
                                  2, self.spectheight]))
                constant_in_abs_dir = complex_abs(constant_in_dir, axis=2)
                inst_spects_flat_dir = tf.reshape(
                    spects.insts_dir,
                    [batch_size, batch_factor * fan_factor,
                     self.inst_num * 2, self.spectheight])
                model_in = tf.concat(
                    [constant_in, constant_in_abs, inst_spects_flat,
                     constant_in_dir, constant_in_abs_dir,
                     inst_spects_flat_dir],
                    axis=2)
                spectvis_d.constant_in = constant_in

        return spects, losses, params_d, spectvis_d

    def add_gradients_model_cnn(self, gradients_model_cnn):
        """
        Add gradients related to the CNN model

        Parameters
        ----------
        gradients_model_cnn : sequence of tensor of float
            Gradients with respect to the model
        """

        if self.gradients_model_cnn is None:
            self.gradients_model_cnn = gradients_model_cnn
        else:
            self.gradients_model_cnn = [ g1 + g2 for g1, g2
                                         in zip(self.gradients_model_cnn,
                                                gradients_model_cnn) ]

    def add_gradient_dict(self, gradient_dict):
        """
        Add gradients related to the dictionary.

        Parameters
        ----------
        gradient_dict : tensor of float
            Gradient with respect to the dictionary
        """

        if self.gradient_dict is None:
            self.gradient_dict = gradient_dict
        else:
            self.gradient_dict += gradient_dict

    def add_gradient_scales(self, gradient_scales):
        """
        Add gradients related to the output scalings.

        Parameters
        ----------
        gradient_dict : tensor of float
            Gradient with respect to the output scalings
        """

        if self.gradient_scales is None:
            self.gradient_scales = gradient_scales
        else:
            self.gradient_scales += gradient_scales

    def apply_gradients(self):
        """
        Apply all gradients to the optimization algorithm.
        """

        variables_list = [self.param_scales] + self.variables_cnn
        gradient_list = [self.gradient_scales] + self.gradients_model_cnn
        self.optimizer.apply_gradients(
            zip(gradient_list, variables_list))
        self.gradients_model_cnn = None
        self.gradient_scales = None

        self.dict_optimizer.apply_gradients(
            [[self.gradient_dict, self.inst_dict]])
        self.gradient_dict = None

    def predict_mix_spect(self, mix_spect_in):
        """
        Predict the separation of a mixture spectrum and compute the losses.

        Parameters
        ----------
        mix_spect_in : tensor of float
            Mixture spectrogram to perform the separation on

        Returns
        -------
        spects : SimpleNamespace
            Spectra related to the mixture
        losses : SimpleNamespace
            Losses related to the mixture
        """

        # add a minimum amount of noise
        mix_spect_in += tf.random.normal(tf.shape(mix_spect_in), stddev=1e-10)

        spects, losses, params, spectvis = \
            self.separate(mix_spect_in, None, self.batch_size_pred,
                          predict=True)
        params_sum = params._sum()
        total_loss, _ = comp_total_loss(losses, params.on_factors,
                                        self.spl, self.loss_coeffs)

        return spects, losses

    def train_mix_spect(self, mix_spect_in, writer, k):
        """
        Train the separation of a mixture spectrum and compute the losses.

        Parameters
        ----------
        mix_spect_in : tensor of float
            Mixture spectrogram to perform the separation on
        writer : SummaryWriter
            Writer object to capture the summarized variables
        k : int
            Iteration number

        Returns
        -------
        spects : SimpleNamespace
            Spectra related to the mixture
        losses : SimpleNamespace
            Losses related to the mixture
        params : ParamsDict
            Parameters related to the mixture
        spectvis : SpectvisDict
            Visualization spectra related to the mixture
        """

        # add a minimum amount of noise
        mix_spect_in += tf.random.normal(tf.shape(mix_spect_in), stddev=1e-10)

        with tf.GradientTape(persistent=True) as self.tape:
            spects, losses, params, spectvis = \
                self.separate(mix_spect_in, self.tau, self.batch_size)

            params_sum = params._sum()
            probs_factor = tf.stop_gradient(tf.nn.softmax(
                params_sum.probs + params_sum.on_probs
                - params_sum.probs_samp - params_sum.on_probs_samp))
            probs_factor_avg = tf.stop_gradient(
                tf.nn.softmax(tf.zeros_like(params_sum.probs)))
            probs_corr = probs_factor
            mix_spect_loss = losses.mix
            mix_spect_loss_dir = losses.mix_dir
            mix_spect_loss_sparse = losses.mix_sparse
            loss_inst = losses.inst
            total_loss, _ = comp_total_loss(losses, params.on_factors,
                                            self.spl, self.loss_coeffs)
            baseline = tf.reduce_sum(probs_factor_avg * total_loss,
                                     axis=1, keepdims=True)

            pol_obj = (((params_sum.probs_samp
                           + params_sum.on_probs_samp)
                          * tf.stop_gradient(total_loss - baseline)
                          + (params_sum.g_probs
                             * tf.stop_gradient(total_loss - baseline)))
                          #)
                         * probs_factor_avg) / 10
            total_loss_obj = probs_factor_avg * total_loss
            total_loss_obj_w = probs_factor * total_loss

            obj = total_loss_obj + pol_obj
            dict_obj = total_loss_obj_w

        mix_spect_loss_w = mix_spect_loss * probs_factor
        mix_spect_loss_dir_w = mix_spect_loss_dir * probs_factor
        mix_spect_loss_sparse_w = mix_spect_loss_sparse * probs_factor
        loss_inst_w = loss_inst * probs_factor

        neff = 1 / tf.reduce_sum(probs_factor**2, axis=1)
        variance = (tf.reduce_sum(probs_factor * (total_loss - baseline)**2,
                                  axis=1)
                    / (1 - 1 / neff))

        gradients_model_cnn, gradient_scales_loss = \
            self.tape.gradient(obj, [self.variables_cnn, self.param_scales],
                          unconnected_gradients=tf.UnconnectedGradients.ZERO)
        self.add_gradients_model_cnn(gradients_model_cnn)
        self.add_gradient_scales(gradient_scales_loss)

        gradient_dict = self.tape.gradient(
            dict_obj, self.inst_dict,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)

        grad_norm = np.sum([tf.reduce_sum(g**2) for g in
                            self.gradients_model_cnn])
        grad_scales_norm = tf.reduce_sum(gradient_scales_loss**2)
        grad_dict_norm = tf.reduce_sum(tf.convert_to_tensor(gradient_dict)**2)

        self.add_gradient_dict(gradient_dict)

        with writer.as_default():
            tf.summary.scalar('probs_corr_norm',
                              tf.reduce_sum(probs_corr**2),
                              step=k)
            tf.summary.scalar('mix_spect_loss',
                              tf.reduce_sum(
                                  tf.reduce_mean(mix_spect_loss_w,
                                                 axis=0)),
                              step=k)
            tf.summary.scalar('mix_spect_loss_dir',
                              tf.reduce_sum(
                                  tf.reduce_mean(mix_spect_loss_dir_w,
                                                 axis=0)),
                              step=k)
            tf.summary.scalar('mix_spect_loss_sparse',
                              tf.reduce_sum(
                                  tf.reduce_mean(mix_spect_loss_sparse_w,
                                                 axis=0)),
                              step=k)
            tf.summary.scalar('inst_loss',
                              tf.reduce_sum(
                                  tf.reduce_mean(loss_inst_w,
                                                 axis=0)),
                              step=k)
            tf.summary.scalar('total_loss',
                              tf.reduce_sum(
                                  tf.reduce_mean(total_loss_obj,
                                                 axis=0)),
                              step=k)
            tf.summary.scalar('on_factors',
                              tf.reduce_sum(
                                  tf.reduce_mean(params_sum.on_factors,
                                                 axis=0)),
                              step=k)
            tf.summary.scalar('on_probs',
                              - tf.reduce_sum(
                                  tf.reduce_mean(params_sum.on_probs,
                                                 axis=0)),
                              step=k)
            tf.summary.scalar('probs',
                              - tf.reduce_sum(
                                  tf.reduce_mean(params_sum.probs,
                                                 axis=0)),
                              step=k)
            tf.summary.scalar('g_probs',
                              - tf.reduce_sum(
                                  tf.reduce_mean(params_sum.g_probs,
                                                 axis=0)),
                              step=k)
            tf.summary.scalar('grad_norm',
                              grad_norm, step=k)
            tf.summary.scalar('grad_dict_norm',
                              grad_dict_norm, step=k)
            tf.summary.scalar('neff',
                              tf.reduce_mean(neff), step=k)
            tf.summary.scalar('stdev',
                              tf.reduce_mean(tf.sqrt(variance)), step=k)
            tf.summary.scalar('entropy',
                              tf.reduce_mean(params_sum.entropy), step=k)

        return spects, losses, params, spectvis

    def train_dict_norm(self, writer, k):
        """
        Train the dictionary such that the largest entry for each instrument
        gets to 1.

        Parameters
        ----------
        writer : SummaryWriter
            Writer object to capture the summarized variables
        k : int
            Iteration number

        Returns
        -------
        inst_dict_norm : tensor of float
            Dictionary norm loss
        """

        with tf.GradientTape() as tape:
            inst_dict_norm = tf.reduce_mean(tf.math.log(tf.reduce_max(
                self.inst_dict, axis=1)) ** 2)

        gradient_dict = tape.gradient(inst_dict_norm, self.inst_dict)
        grad_dict_norm = tf.reduce_sum(gradient_dict**2)

        with writer.as_default():
            tf.summary.scalar('inst_dict_norm',
                              tf.reduce_sum(inst_dict_norm), step=k)
            tf.summary.scalar('grad_dict_scale_norm',
                              tf.reduce_sum(grad_dict_norm), step=k)

        self.add_gradient_dict(gradient_dict)

        return inst_dict_norm

    def predict_loop(self, k, write):
        """
        Predict and resynthesize the entire spectrogram.


        Parameters
        ----------
        k : int
            Iteration number
        write : bool
            Whether to save the output to files
        """

        inst_spects_full_dir = np.zeros(
            [self.spectlen_sub, self.inst_num, 2, self.spectheight])
        inst_spects_full = np.zeros(
            [self.spectlen_sub, self.inst_num, 2, self.spectheight])
        count = 0

        mix_spect_loss = 0
        mix_spect_loss_dir = 0
        mix_spect_loss_sparse = 0
        loss_inst = 0

        cnt = 0
        for mix_spect_in in iter(self.inline_slices):
            print("{} out of {}".
                  format(cnt, self.spectlen_sub // self.batch_size_pred))
            cnt += 1

            retcount = tf.shape(mix_spect_in)[0]

            mix_spect_in = tf.concat(
                [mix_spect_in, tf.zeros([self.batch_size_pred - retcount,
                                         2, self.spectheight])],
                axis=0)
            spects, losses = \
                self.predict_mix_spect(mix_spect_in)
            inst_spects_dir = (self.sign_mask * spects.insts_dir).numpy()
            inst_spects_dir = np.squeeze(inst_spects_dir, axis=1)
            inst_spects = (self.sign_mask * spects.insts).numpy()
            inst_spects = np.squeeze(inst_spects, axis=1)
            newcount = count + inst_spects_dir.shape[0]

            inst_spects_full_dir[count:newcount, :, :, :] = \
                inst_spects_dir[:retcount, :, :, :]
            inst_spects_full[count:newcount, :, :, :] = \
                inst_spects[:retcount, :, :, :]
            count = newcount

            mix_spect_loss += losses.mix
            mix_spect_loss_dir += losses.mix_dir
            mix_spect_loss_sparse += losses.mix_sparse
            loss_inst += losses.inst

        corr_factor = 1 / self.spectlen_sub
        with self.eval_writer.as_default():
            tf.summary.scalar('mix_spect_loss',
                              tf.reduce_sum(mix_spect_loss) * corr_factor,
                              step=k)
            tf.summary.scalar('mix_spect_loss_dir',
                              tf.reduce_sum(mix_spect_loss_dir) * corr_factor,
                              step=k)
            tf.summary.scalar('mix_spect_loss_sparse',
                              tf.reduce_sum(mix_spect_loss_sparse)
                              * corr_factor,
                              step=k)
            tf.summary.scalar('inst_loss',
                              tf.reduce_sum(loss_inst) * corr_factor,
                              step=k)

        synth_signals = np.zeros((self.inst_num, self.signal.size))
        synth_signals_dir = np.zeros((self.inst_num, self.signal.size))

        for i in range(self.inst_num):
            inst_spect = (inst_spects_full[:, i, 0, :]
                          + 1j * inst_spects_full[:, i, 1, :])
            audio = spect.project_audio(
                inst_spect.T, self.signal.size, self.sigmas_an,
                self.sampdist * self.sub_factor)
            synth_signals[i, :] = audio
            if write and False:
                wav.write('out-nn/' + self.name + '-{}-{}.wav'.format(i, k),
                          audio, self.samprate)
            inst_spect = None

            inst_spect_dir = (inst_spects_full_dir[:, i, 0, :]
                              + 1j * inst_spects_full_dir[:, i, 1, :])
            audio = spect.project_audio(
                inst_spect_dir.T, self.signal.size, self.sigmas_an,
                self.sampdist * self.sub_factor)
            synth_signals_dir[i, :] = audio
            if write:
                wav.write('out-nn/' + self.name + '-dir-{}-{}.wav'
                          .format(i, k),
                          audio, self.samprate)

            if write and self.plot_color is not None:
                spect.spectwrite('out-nn/' + self.name + '-dir-{}-{}.png'
                                 .format(i, k),
                                 np.abs(inst_spect_dir.T), self.plot_color)
            inst_spect_dir = None


        perm, perf = performance.select_perm(*performance.measures(
            synth_signals, self.orig_signals))
        print("model")
        print(perf)
        with self.eval_writer.as_default():
            tf.summary.scalar('SDR', np.mean(perf[0, :]), step=k)
            tf.summary.scalar('SIR', np.mean(perf[1, :]), step=k)
            tf.summary.scalar('SAR', np.mean(perf[2, :]), step=k)

        perm_dir, perf_dir = performance.select_perm(*performance.measures(
            synth_signals_dir, self.orig_signals))
        print("dir")
        print(perf_dir)
        with self.eval_writer.as_default():
            tf.summary.scalar('SDR_dir', np.mean(perf_dir[0, :]), step=k)
            tf.summary.scalar('SIR_dir', np.mean(perf_dir[1, :]), step=k)
            tf.summary.scalar('SAR_dir', np.mean(perf_dir[2, :]), step=k)

        self.measures_table.append(
            np.ravel(np.concatenate([perf, perf_dir], axis=1)))

        np.savetxt('out-nn/' + self.name + '-measures.dat',
                   np.vstack([np.arange(len(self.measures_table)),
                              np.asarray(self.measures_table).T]).T)

        mix_spect = np.sum(inst_spects_full[:, :, 0, :]
                           + 1j * inst_spects_full[:, :, 1, :], axis=1)
        audio = spect.project_audio(
            mix_spect.T, self.signal.size, self.sigmas_an,
            self.sampdist * self.sub_factor)
        if write and False:
            wav.write('out-nn/' + self.name + '-{}.wav'.format(k),
                      audio, self.samprate)
        mix_spect = None

        mix_spect_dir = np.sum(inst_spects_full_dir[:, :, 0, :]
                               + 1j * inst_spects_full_dir[:, :, 1, :], axis=1)
        audio = spect.project_audio(
            mix_spect_dir.T, self.signal.size, self.sigmas_an,
            self.sampdist * self.sub_factor)
        if write:
            wav.write('out-nn/' + self.name + '-dir-{}.wav'.format(k),
                      audio, self.samprate)

        if write and self.plot_color is not None:
            spect.spectwrite('out-nn/' + self.name + '-dir-{}.png'.format(k),
                             np.abs(mix_spect_dir.T), self.plot_color)
        mix_spect_dir = None

    def train_loop(self, max_iter, eval_interval, interval=50):
        """
        Train the neural network.
        Predict and resynthesize the entire spectrogram.

        Parameters
        ----------
        max_iter : int
            Total number of training iterations
        eval_interval : int
            Interval at which to evaluate the entire spectrogram
        interval : bool
            Interval at which to output debug information
        """

        for k in itertools.count():
            print("iteration: {}".format(k), end="\r")

            if k % eval_interval == 0:
                self.predict_loop(k, k in self.save_points)

            if k in self.save_points:
                tf.saved_model.save(self.save_module,
                                    'out-nn/{}-{}'.format(self.name, k))

            if k >= max_iter:
                break

            mix_spect_in = next(self.random_slice)
            mix_spect_in_abs = complex_abs(mix_spect_in, axis=1)

            spects, losses, params, spectvis = \
                self.train_mix_spect(mix_spect_in, self.file_writer, k)

            inst_dict_norm = self.train_dict_norm(self.file_writer, k)

            if k % self.virt_batch_mul == 0:
                self.apply_gradients()

            if k % interval == 0 and False:
                plot_spectrum('out-nn/' + self.name + '-pdf.pdf'.format(),
                              self.spectheight,
                              lift_spect_sign(mix_spect_in_abs[0, 0, :]),
                              *tf.unstack(spectvis.pdf, axis=0))
                plot_spectrum('out-nn/' + self.name + '-amps.pdf'.format(),
                              self.spectheight,
                              lift_spect_sign(mix_spect_in_abs[0, 0, :]),
                              *[lift_spect_sign(s)
                                for s in tf.unstack(spectvis.amps, axis=0)])
                plot_spectrum('out-nn/' + self.name + '-probs.pdf'.format(),
                              self.spectheight,
                              lift_spect_sign(mix_spect_in_abs[0, 0, :]),
                              *tf.unstack(spectvis.on_probs, axis=0))
                plot_spectrum('out-nn/' + self.name + '-sigmas.pdf'.format(),
                              self.spectheight,
                              lift_spect_sign(mix_spect_in_abs[0, 0, :]),
                              *tf.unstack(spectvis.sigmas, axis=0))
                plot_spectrum('out-nn/' + self.name + '-offsets.pdf'.format(),
                              self.spectheight,
                              lift_spect_sign(mix_spect_in_abs[0, 0, :]),
                              *tf.unstack(spectvis.offsets, axis=0))
                plot_spectrum('out-nn/' + self.name
                              + '-spreads_a.pdf'.format(),
                              self.spectheight,
                              *tf.unstack(tf.math.log(spectvis.spreads_a)
                                          / np.log(10), axis=0))
                plot_spectrum('out-nn/' + self.name
                              + '-spreads_b.pdf'.format(),
                              self.spectheight,
                              *tf.unstack(tf.math.log(spectvis.spreads_b)
                                          / np.log(10), axis=0))
                plot_spectrum('out-nn/' + self.name + '-spect.pdf'.format(),
                              self.spectheight,
                              *[lift_spect_sign(s) for s in
                                tf.reshape(mix_spect_in[0, :, :],
                                           [-1, self.spectheight])],
                              *[lift_spect_sign(s) for s in
                                tf.unstack(
                                    tf.reshape(spects.insts[0, 0, :, :, :],
                                               [-1, self.spectheight]),
                                    axis=0)])
                plot_spectrum('out-nn/' + self.name + '-resin.pdf'.format(),
                              self.spectheight,
                              *[lift_spect_sign(s) for s in
                                tf.unstack(
                                    tf.reshape(
                                        spectvis.constant_in[0, 0, :, :],
                                        [-1, self.spectheight]),
                                    axis=0)])
                plot_spectrum('out-nn/' + self.name + '-dir.pdf'.format(),
                              self.spectheight,
                              *[lift_spect_sign(s) for s in
                                tf.reshape(mix_spect_in[0, :, :],
                                           [-1, self.spectheight])],
                              *[lift_spect_sign(s) for s in
                                tf.unstack(
                                    tf.reshape(
                                        spects.insts_dir[0, 0, :, :, :],
                                        [-1, self.spectheight]),
                                    axis=0)])
                plot_spectrum('out-nn/' + self.name + '-sp.pdf'.format(),
                              self.spectheight,
                              *[lift_spect_sign(s) for s in
                                tf.reshape(mix_spect_in[0, :, :],
                                           [-1, self.spectheight])],
                              *[lift_spect_sign(s) for s in
                                tf.unstack(
                                    tf.reshape(
                                        spects.insts_sparse[0, 0, :, :, :],
                                        [-1, self.spectheight]), axis=0)])

                print("param_scales", tf.math.exp(self.param_scales))
                print("inst_dict", self.inst_dict)
