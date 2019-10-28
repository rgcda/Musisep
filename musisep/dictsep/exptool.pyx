#!python3
#cython: boundscheck=True, language_level=3

"""
Back-end module for the generation of spectrograms and their gradients.
"""

import numpy as np
cimport numpy
cimport libc.math

from cython.parallel import prange

cpdef double[:] inst_shift(double[:] amps, double[:] shifts, double[:] sigmas,
                           double[:] spreads, long[:] insts,
                           double[:,:] inst_dict, double harscale,
                           double pexp,
                           int m):
    cdef double[:] expvec = np.zeros(m, dtype='double')
    cdef double amp, shift, sigma, spread, hcorr, hshift, hamp, entry
    cdef int lo, hi, i, j, h, inst
    
    for j in range(amps.size):
        amp = amps[j]
        sigma = sigmas[j]
        shift = shifts[j]
        spread = spreads[j]
        inst = insts[j]

        if not libc.math.isfinite(shift):
            continue

        for h in range(0, inst_dict.shape[0]):
            hcorr = libc.math.log(1 + spread * (h+1)**2) / 2
            hshift = shift + (libc.math.log(h+1) + hcorr) * harscale
            hamp = amp * inst_dict[h, inst]

            lo = <int?> libc.math.floor(hshift - 6 * sigma)
            if lo < 0:
                lo = 0
            hi = <int?> libc.math.ceil(hshift + 6 * sigma)
            if hi >= m:
                hi = m

            for i in range(lo, hi):
                entry = hamp * libc.math.exp(- ((i - hshift) / sigma)**2 / 2)
                expvec[i] += entry ** pexp

    return expvec

cpdef double[:] inst_shift_grad(double[:] expvec,
                                double[:] amps, double[:] shifts,
                                double[:] sigmas,
                                double[:] spreads, long[:] insts,
                                double[:,:] inst_dict, double harscale,
                                double pexp,
                                int m):
    cdef int n = amps.size
    cdef double[:] expgrad = np.zeros(4 * n, dtype='double')
    cdef double amp, shift, sigma, spread
    cdef double diff_amp, diff_shift, diff_sigma, diff_spread    
    cdef double hcorr, hshift, hentry, hamp, expval, factor, dcorr
    cdef int lo, hi, i, j, s, h, inst

    for j in prange(n, nogil=True):
        amp = amps[j]
        sigma = sigmas[j]
        shift = shifts[j]
        spread = spreads[j]
        inst = insts[j]

        if not libc.math.isfinite(shift):
            continue

        for h in range(0, inst_dict.shape[0]):
            hcorr = libc.math.log(1 + spread * (h+1)**2) / 2
            hshift = shift + (libc.math.log(h+1) + hcorr) * harscale
            hentry = inst_dict[h, inst]
            hamp = amp * hentry

            lo = <int?> libc.math.floor(hshift - 6 * sigma)
            if lo < 0:
                lo = 0
            hi = <int?> libc.math.ceil(hshift + 6 * sigma)
            if hi >= m:
                hi = m

            for i in range(lo, hi):
                expval = libc.math.exp(- ((i - hshift) / sigma)**2 / 2)
                factor = (hamp * expval + 1e-40*0) ** pexp
                dcorr = (h+1)**2 / (1 + spread * (h+1)**2) * harscale / 2
                diff_amp = hentry * expval
                diff_shift = hamp * expval * (i - hshift) / sigma**2
                diff_sigma = hamp * expval * (i - hshift)**2 / sigma**3
                diff_spread = hamp * expval * (i - hshift) / sigma**2 * dcorr

                expgrad[j + 0*n] += factor * diff_amp * expvec[i]
                expgrad[j + 1*n] += factor * diff_shift * expvec[i]
                expgrad[j + 2*n] += factor * diff_sigma * expvec[i]
                expgrad[j + 3*n] += factor * diff_spread * expvec[i]

    return expgrad

cpdef double[:, :] inst_shift_dict_grad(double[:] expvec,
                                        double[:] amps, double[:] shifts,
                                        double[:] sigmas,
                                        double[:] spreads, long[:] insts,
                                        double[:, :] inst_dict,
                                        double harscale,
                                        double pexp,
                                        int m):
    cdef int n = amps.size
    cdef double[:, :] dictgrad = np.zeros((inst_dict.shape[0],
                                           inst_dict.shape[1]),
                                             dtype='double')
    cdef double amp, shift, sigma, spread
    cdef double hcorr, hshift, hentry, hamp, expval, factor
    cdef int lo, hi, i, j, s, h, inst

    for j in range(n):
        amp = amps[j]
        sigma = sigmas[j]
        shift = shifts[j]
        spread = spreads[j]
        inst = insts[j]

        if not libc.math.isfinite(shift):
            continue

        for h in prange(0, inst_dict.shape[0], nogil=True):
            hcorr = libc.math.log(1 + spread * (h+1)**2) / 2
            hshift = shift + (libc.math.log(h+1) + hcorr) * harscale
            hentry = inst_dict[h, inst]
            hamp = amp * hentry

            lo = <int?> libc.math.floor(hshift - 6 * sigma)
            if lo < 0:
                lo = 0
            hi = <int?> libc.math.ceil(hshift + 6 * sigma)
            if hi >= m:
                hi = m

            for i in range(lo, hi):
                expval = libc.math.exp(- ((i - hshift) / sigma)**2 / 2)
                factor = (hamp * expval + 1e-40*0) ** pexp
                dictgrad[h, inst] += factor * amp * expval * expvec[i]

    return dictgrad

cpdef double[:] inst_scale(double[:] amps, double[:] shifts, double[:] sigmas,
                           double[:] spreads, long[:] insts,
                           double[:,:] inst_dict,
                           double pexp,
                           int m):
    cdef int n = amps.size
    cdef double[:] expvec = np.zeros(m, dtype='double')
    cdef double amp, shift, sigma, spread, hcorr, hshift, hamp, entry
    cdef int lo, hi, i, j, h, inst
    
    for j in range(n):
        amp = amps[j]
        sigma = sigmas[j]
        shift = shifts[j]
        spread = spreads[j]
        inst = insts[j]

        if not libc.math.isfinite(shift):
            continue

        for h in range(0, inst_dict.shape[0]):
            hcorr = libc.math.sqrt(1 + spread * (h+1)**2)
            hshift = shift * (h+1) * hcorr
            hamp = amp * inst_dict[h, inst]

            lo = <int?> libc.math.floor(hshift - 6 * sigma)
            if lo < 0:
                lo = 0
            hi = <int?> libc.math.ceil(hshift + 6 * sigma)
            if hi >= m:
                hi = m

            for i in range(lo, hi):
                entry = hamp * libc.math.exp(- ((i - hshift) / sigma)**2 / 2)
                expvec[i] += entry ** pexp

    return expvec

cpdef double[:] inst_scale_grad(double[:] expvec,
                                double[:] amps, double[:] shifts,
                                double[:] sigmas,
                                double[:] spreads, long[:] insts,
                                double[:,:] inst_dict,
                                double pexp,
                                int m):
    cdef int n = amps.size
    cdef double[:] expgrad = np.zeros(4 * n, dtype='double')
    cdef double amp, shift, sigma, spread
    cdef double diff_amp, diff_shift, diff_sigma, diff_spread    
    cdef double hsqrt, hcoeff, hshift, hentry, hamp, expval, factor
    cdef int lo, hi, i, j, s, h, inst

    for j in prange(n, nogil=True):
        amp = amps[j]
        sigma = sigmas[j]
        shift = shifts[j]
        spread = spreads[j]
        inst = insts[j]

        if not libc.math.isfinite(shift):
            continue

        for h in range(0, inst_dict.shape[0]):
            hshift = shift * (h+1)
            hentry = inst_dict[h, inst]
            hamp = amp * hentry

            lo = <int?> libc.math.floor(hshift - 6 * sigma)
            if lo < 0:
                lo = 0
            hi = <int?> libc.math.ceil(hshift + 6 * sigma)
            if hi >= m:
                hi = m

            for i in range(lo, hi):
                expval = libc.math.exp(- ((i - hshift) / sigma)**2 / 2)
                factor = (hamp * expval) ** pexp
                diff_amp = hentry * expval
                diff_shift = expval * hamp * (i - hshift) / sigma**2
                diff_sigma = hamp * expval * (i - hshift)**2 / sigma**3
                diff_spread = 0

                expgrad[j + 0*n] += factor * diff_amp * expvec[i]
                expgrad[j + 1*n] += factor * diff_shift * expvec[i]
                expgrad[j + 2*n] += factor * diff_sigma * expvec[i]
                expgrad[j + 3*n] += factor * diff_spread * expvec[i]

    return expgrad
