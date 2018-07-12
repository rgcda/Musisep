#!python3
#cython: boundscheck=True, language_level=3

"""
Back-end module for the Griffin-Lim algorithm.
"""

import numpy as np
cimport numpy
cimport libc.math

from cython.parallel import prange

cpdef double[:] unstripe(double[:, :] stripeplot, double[:] window,
      	                 int siglen, int length, int n, int sampdist):
    cdef double[:] signal = np.zeros(siglen)
    cdef double iprod, winsum
    cdef int i, j, k, offset, p
    
    for j in prange(siglen, nogil=True):
        iprod = 0
        winsum = 0
        
        offset = j % sampdist
        p = j // sampdist

        k = p
        i = offset + length

        while i >= 0 and k < n:
            iprod = iprod + window[i] * stripeplot[i, k]
            winsum = winsum + window[i] ** 2
            i = i - sampdist
            k = k + 1
        
        k = p - 1
        i = offset + length + sampdist
            
        while i < length*2 and k >= 0:
            if k < n:
                iprod = iprod + window[i] * stripeplot[i, k]
                winsum = winsum + window[i] ** 2
            i = i + sampdist
            k = k - 1

        if winsum > 0:
            signal[j] = iprod / winsum

    return signal

cpdef void adapt_mag(double complex[:, :] spect, double[:, :] magspect,
                     int m, int n):
    cdef int i, j
    cdef double complex entry

    for j in prange(n, nogil=True):
        for i in range(m):
            entry = spect[i, j]
            spect[i, j] = (entry + 1e-40) * magspect[i, j] \
                / (libc.math.hypot(entry.real, entry.imag) + 1e-40)
