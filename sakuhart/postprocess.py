"""Postprocessing: the "make-up" pass that gives vivid its look.

Two kinds of work happen here and they are kept strictly apart:

* Pixel-independent stages (vibrance, saturation, contrast, harmony) depend
  only on a pixel's own colour plus a few frozen frame statistics, so they are
  baked into a 3D lookup table once per keyframe and then applied by
  interpolation. That is what turns ~2.5 s of chain into a table lookup.
* Spatial stages (high-frequency lift, unsharp) look at neighbouring pixels
  and cannot be tabulated; they run every frame.

A still image never builds the table: baking 128**3 lattice points to colour
one frame of two million pixels is more arithmetic than colouring the pixels
directly, and the direct answer is the exact one. The table exists to be
reused across the frames between two cuts.

The statistics are captured at keyframes and held constant until the next one.
Recomputing them per frame makes the saturation pulse."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from .color import chroma_percentile, vibrance_oklch

NDArray = np.ndarray

# The other half of the vivid recipe: everything the make-up pass applies.
GAMMA = 0.94
SHADOW_LIFT = 9
SATURATION_COOL = 1.75
SATURATION_WARM = 1.45
VIBRANCE = 0.48
CONTRAST = 1.06
SHARPNESS = 0.28
COLOR_HARMONY = 0.07

# 256 is exact to 0.0002 of a level against the chain itself, where 128 is off
# by 0.80 on average and 17 at worst -- and it takes 13x as long to build. The
# table is only ever built for video, where that cost lands on every cut.
LUT_SIZE = 128
LUT_WORKERS = 8  # both LUT stages are per pixel, so they split across threads

_local = threading.local()  # scratch buffers must not be shared between threads
_pool: ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def workers() -> ThreadPoolExecutor:
    """The shared pool for the two per-pixel LUT stages, built on first use.

    One pool for the process: a fresh ThreadPoolExecutor per frame spends more
    time starting and joining threads than the stage saves."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(LUT_WORKERS, thread_name_prefix="lut")
        return _pool


def _by_slices(n: int, fn) -> list:
    """Run `fn(lo, hi)` on the shared pool over LUT_WORKERS contiguous,
    near-equal bands of `n` rows, and give the results back in order."""
    edges = np.linspace(0, n, LUT_WORKERS + 1).astype(int).tolist()
    return list(workers().map(lambda s: fn(*s), zip(edges[:-1], edges[1:])))


def _tone_lut() -> NDArray:
    """Gamma and shadow lift are both per-byte curves, so they fuse into one.

    The gamma result is truncated to uint8 before the lift, which reproduces
    the two-pass form exactly rather than merely closely."""
    v = ((np.arange(256) / 255.0) ** GAMMA * 255.0).astype(np.uint8).astype(np.float32)
    return np.clip(v + SHADOW_LIFT * (1.0 - v / 255.0), 0, 255).astype(np.uint8)


TONE_LUT = _tone_lut()


FILTER_REACH = 9  # rows the blurs reach: freq 7 + unsharp 2
DIRTY_MAX = 0.7  # dirty rectangles bigger than this share of the frame: full pass


@dataclass
class ToneStats:
    """Everything the tabulated stages need to know about the frame."""

    chroma_max: float
    mean_a: float
    mean_b: float


def capture_stats(bgr: NDArray) -> ToneStats:
    """Measure the frame statistics, at half resolution: they are a percentile
    and two means, and sampling every other pixel does not move them enough to
    see (measured SSIM 0.99975 end to end)."""
    small = bgr[::2, ::2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    return ToneStats(
        chroma_max=chroma_percentile(small),
        mean_a=float(lab[:, :, 1].mean()),
        mean_b=float(lab[:, :, 2].mean()),
    )


# The tabulated half
def _pixel_chain(bgr: NDArray, stats: ToneStats) -> NDArray:
    """vibrance -> saturation -> contrast -> harmony. Every stage reads one
    pixel and the frozen frame statistics, which is what makes it tabulatable."""
    out = vibrance_oklch(bgr, VIBRANCE, stats.chroma_max)
    out = _boost_saturation(out)
    contrast = 128 + (out.astype(np.float32) - 128) * CONTRAST
    return _harmony(np.clip(contrast, 0, 255).astype(np.uint8), stats)


def _boost_saturation(bgr: NDArray) -> NDArray:
    """HSV saturation gain, warmer hues pushed less than cool ones."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv[:, :, 0]
    factor = np.full_like(h, SATURATION_COOL)
    factor[(h <= 25) | (h >= 170)] = SATURATION_WARM
    ramp = (h > 25) & (h < 40)
    factor[ramp] = (
        SATURATION_WARM + (SATURATION_COOL - SATURATION_WARM) * (h[ramp] - 25.0) / 15.0
    )
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _harmony(bgr: NDArray, stats: ToneStats) -> NDArray:
    """Pull a/b toward the frame's mean chroma. Frozen means, or it flickers."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 1] += (stats.mean_a - lab[:, :, 1]) * COLOR_HARMONY
    lab[:, :, 2] += (stats.mean_b - lab[:, :, 2]) * COLOR_HARMONY
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def build_luts(stats: ToneStats) -> NDArray:
    """Bake the pixel-independent chain into one table, for video.

    Every stage in the chain looks at one pixel and some frozen numbers, so the
    lattice splits into chunks that are computed independently and glued back
    together -- the same bytes as one call (verified by hash), on 8 threads.
    Measured at a keyframe: 2.36 s -> 0.50 s."""
    axis = np.linspace(0, 255, LUT_SIZE).astype(np.uint8)
    lattice = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    flat = lattice.reshape(LUT_SIZE * LUT_SIZE, LUT_SIZE, 3)
    parts = _by_slices(len(flat), lambda a, b: _pixel_chain(flat[a:b], stats))
    return np.concatenate(parts).reshape(lattice.shape)


def apply_lut(bgr: NDArray, lut: NDArray) -> NDArray:
    """Trilinear lookup of a (N,N,N,3) table indexed by B,G,R.

    Split by pixel across the same pool: map_coordinates releases the GIL, so
    the pieces really do run at once (measured 351 ms -> 96 ms at 1080p), and
    each worker writes only its own slice of the output."""
    flat = bgr.reshape(-1, 3)
    return np.concatenate(
        _by_slices(len(flat), lambda a, b: _lut_slice(flat[a:b], lut))
    ).reshape(bgr.shape)


def _lut_slice(src: NDArray, lut: NDArray) -> NDArray:
    """One worker's share. The scratch buffer is a local, never a shared one:
    a module-level buffer here is a data race waiting to be shipped."""
    coords = src.T.astype(np.float32) * ((LUT_SIZE - 1) / 255.0)
    buf = np.empty((3, coords.shape[1]), np.float32)
    for c in range(3):
        map_coordinates(lut[..., c], coords, order=1, output=buf[c], mode="nearest")
    return buf.T.astype(np.uint8)


# The spatial half
def frequency_enhance(bgr: NDArray) -> NDArray:
    """Mild high-pass lift. float32 throughout: float64 costs 3x for 0.0007%.

    Written in place, into two buffers rather than the seven a plain expression
    would ask for: a 1080p frame is 24 MB per float32 temporary, and freshly
    allocating them was two thirds of this stage (82 ms -> 29, same bits).
    The scratch is a thread-local, never a shared one: a module-level buffer
    here is a data race waiting to be shipped."""
    high = getattr(_local, "freq", None)
    if high is None or high.shape != bgr.shape:
        high = _local.freq = np.empty(bgr.shape, np.float32)
    np.copyto(high, bgr)
    low = cv2.GaussianBlur(high, (15, 15), 5)
    high -= low  # 128 + (low - 128) * 1.03 + (bgr - low) * 1.08
    high *= 1.08
    low -= 128
    low *= 1.03
    low += 128
    low += high
    return np.clip(low, 0, 255, out=low).astype(np.uint8)


def postprocess(
    raw: NDArray,
    stats: ToneStats,
    lut: NDArray | None = None,
    prev: NDArray | None = None,
    box: tuple[int, int, int, int] | None = None,
) -> NDArray:
    """Full chain. `raw` is the freshly pasted mosaic. `lut` comes from
    `build_luts(stats)` and is reused for every frame until the next keyframe;
    a still image passes None and the chain runs on the pixels directly.

    Give it the previous finished frame as `prev` and the rectangle of cells
    that were repainted and it does
    the dirty-rect pass instead: only the part of the frame those cells can
    reach is recomputed, and the rest is the previous frame's answer. That is
    the same bytes, not an approximation -- see `_splice_box` for why the
    reach is finite, and `tests/panel.py` for the frame-by-frame proof."""
    h, w = raw.shape[:2]
    out = cv2.LUT(raw, TONE_LUT)
    keep = _splice_box(h, w, box) if prev is not None else None
    if keep is None:
        return _colour(frequency_enhance(out), stats, lut)

    y0, y1, x0, x1 = keep
    cy0, cy1 = max(0, y0 - FILTER_REACH), min(h, y1 + FILTER_REACH)
    cx0, cx1 = max(0, x0 - FILTER_REACH), min(w, x1 + FILTER_REACH)
    inner = (slice(y0 - cy0, y1 - cy0), slice(x0 - cx0, x1 - cx0))
    sub = _colour(frequency_enhance(out[cy0:cy1, cx0:cx1]), stats, lut)
    final = prev.copy()  # a copy: the last one may still be in the queue
    final[y0:y1, x0:x1] = sub[inner]
    return final


def _colour(pre: NDArray, stats: ToneStats, lut: NDArray | None) -> NDArray:
    """The tabulated half, then the sharpening, over a ready pre-colour frame.

    Without a table the chain runs on the pixels, in bands of whole rows: a
    band that keeps the row length is what stops OpenCV's separable blurs from
    answering differently at the edges (see `_splice_box`)."""
    out = (
        apply_lut(pre, lut)
        if lut is not None
        else np.concatenate(
            _by_slices(len(pre), lambda a, b: _pixel_chain(pre[a:b], stats))
        )
    )
    return cv2.addWeighted(
        out, 1.0 + SHARPNESS, cv2.GaussianBlur(out, (5, 5), 0), -SHARPNESS, 0
    )


def _splice_box(
    h: int, w: int, box: tuple[int, int, int, int] | None
) -> tuple[int, int, int, int] | None:
    """Everything a change inside `box` can alter, or None for the whole frame.

    Rows only: the band is narrowed vertically and always spans the full width.
    A narrower array is not the same computation -- OpenCV's separable blurs
    walk a row in vector-sized blocks and finish the remainder one pixel at a
    time, so the last columns of a 1252-wide crop come out 1 to 3 levels away
    from the same columns of the 1890-wide frame, at identical borders.
    Measured; multiples of 16 happened to agree and nothing else did, which is
    not a property to build on. Row length never changes, so cutting rows is
    free of it.

    The vertical reach is the blurs, 7 + 2. Past `DIRTY_MAX` of the frame the
    bookkeeping stops paying for itself and the whole frame goes through."""
    if box is None:
        return None
    y0, y1 = max(0, box[0] - FILTER_REACH), min(h, box[1] + FILTER_REACH)
    if y1 - y0 > DIRTY_MAX * h:
        return None
    return y0, y1, 0, w
