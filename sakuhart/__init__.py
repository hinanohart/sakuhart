"""sakuhart -- a photomosaic renderer for images and video.

Every cell of the output is a different photograph, and a video stays steady
between frames instead of boiling. See the README, or run `sakuhart --demo`.

The thread settings below have to run before OpenCV and the BLAS behind numpy
are loaded, and importing any submodule runs this file first -- which is why
they live here and not in the CLI. Exactly one component is allowed to own the
cores: nested parallelism (BLAS threads inside a parallel loop) is slower than
either alone, and OpenBLAS is not bit-reproducible across thread counts."""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2  # noqa: E402

cv2.setNumThreads(1)

__version__ = "0.1.0"
