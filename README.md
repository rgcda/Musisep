# Musisep

This software package is designed to separate the contributions of
different musical instruments in an audio recording.  The algorithms
are documented in:

* Sören Schulze, Emily J. King:  Sparse Pursuit and Dictionary
  Learning for Blind Source Separation in Polyphonic Music Recordings.
  EURASIP J. Audio Speech Music Process. 2021.
  doi:10.1186/s13636-020-00190-4.
* Sören Schulze, Johannes Leuschner, Emily J. King:  Training a
  Deep Neural Network via Policy Gradients for Blind Source Separation
  in Polyphonic Music Recordings. 2021.

Please cite at least on of these papers if you use the software in any
academic context.  Further description is available on the [website of
the mathematics department of Colorado State
University](https://www.math.colostate.edu/~king/software.html#Musisep).
There is also the [precompiled API
documentation](https://www.math.colostate.edu/~king/software/Musisep-API/).

## System Requirements

The code was tested with Python 3.7.7, NumPy 1.19.1,
SciPy 1.5.0, Cython 0.29.21, PyFFTW 0.11.1, Matplotlib 3.2.2,
Tensorflow 2.2.0, and Tensorflow-probability 0.9.0.  The sparse pursuit
method does not require Tensorflow in its default settings, so its import
may be commented out.

Both methods require at least 16 GB of RAM and the neural network was
trained on an NVIDIA GeForce GTX 1080 Ti.

## Installation

In order to compile the Cython code (needed for the sparse pursuit method),
Makefiles are provided in `musisep/audio` and `musisep/dictsep`.
To compile the Sphinx documentation, run `make html` in the `docs` directory.
In summary, you can run the following commands from the main directory:
```
make -C musisep/audio
make -C musisep/dictsep
make -C docs html
```

The input data can be obtained from:
<https://www.math.colostate.edu/~king/software/Musisep-data.zip>.  It
should be unzipped to the main directory.

## Running the test case

For the sparse pursuit method, invoke:
```
python3 -m musisep.dictsep
```
from the main dictionary.  The neural network training is run via:
```
python3 -m musisep.neuralsep
```
However, the file `musisep/neuralsep/__main__.py` must be adjusted
for the different samples.

## License

Copyright (C) 2018-2021 Sören Schulze

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
