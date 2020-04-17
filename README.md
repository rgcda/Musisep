# Musisep

This software package is designed to separate the contributions of
different musical instruments in an audio recording.  The algorithms
are documented in:

* Sören Schulze, Emily J. King:  Sparse Pursuit and Dictionary
  Learning for Blind Source Separation in Polyphonic Music Recordings.
  2019.  <https://arxiv.org/abs/1806.00273>.

Please cite this paper if you use the software in any academic
context.  Further description is available on the [website of the
mathematics department of Colorado State
University](https://www.math.colostate.edu/~king/software.html#Musisep).
There is also the [precompiled API
documentation](https://www.math.colostate.edu/~king/software/Musisep-API/).

## System Requirements

The code is written for Python 3.5; it will *not* work with any
earlier version of Python.  It was tested with NumPy 1.16.1,
SciPy 0.18.1, Cython 0.25.2, PyFFTW 0.10.4, Matplotlib 2.0.0, and
Tensorflow 1.12.0.  The latter is only necessary for the narrowband
method, which is not used in the test example, so its import may be
commented out.  CUDA support is recommended but not required.

With the default example, at least 16 GB of RAM are required.

## Installation

In order to compile the Cython code, Makefiles are provided in
`musisep/audio` and `musisep/dictsep`.  To compile the Sphinx
documentation, run `make html` in the `docs` directory.  In summary,
you can run the following commands from the main directory:
```
make -C musisep/audio
make -C musisep/dictsep
make -C docs html
```

The input data can be obtained from:
<https://www.math.colostate.edu/~king/software/Musisep-data.zip>.  It
should be unzipped to the main directory.

## Running the test case

From the main directory, invoke:
```
python3 -m musisep.dictsep
```

## License

Copyright (C) 2018-2020 Sören Schulze

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
