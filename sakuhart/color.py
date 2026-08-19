"""Perceptual colour: Oklab conversions and the per-cell colour transfer.

The transfer chain is fixed to the "vivid" recipe: MKL optimal transport in
CIELAB followed by a partial histogram match. There is no method dispatch --
sakuhart ships one look.

References:
    Bjoern Ottosson, "A perceptual color space for image processing" (2020).
    Pitie et al., "The linear Monge-Kantorovitch colour mapping" (IET-CVMP 2007)."""

from __future__ import annotations

import cv2
import numpy as np

NDArray = np.ndarray
EPS = 1e-10

# sRGB -> linear depends only on the byte value, so the branchy power law
# collapses into a 256-entry table. Identical output, no pow() per pixel.
_S = np.arange(256, dtype=np.float64) / 255.0
SRGB_TO_LINEAR = np.where(_S <= 0.04045, _S / 12.92, ((_S + 0.055) / 1.055) ** 2.4)


def bgr_to_oklab(bgr_u8: NDArray) -> NDArray:
    """BGR uint8 -> Oklab float64. L in [0,1], a/b roughly in [-0.4, 0.4]."""
    lin = SRGB_TO_LINEAR[bgr_u8]
    b, g, r = lin[..., 0], lin[..., 1], lin[..., 2]
    l_ = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack(
        [
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        ],
        axis=-1,
    )


def oklab_to_bgr(oklab: NDArray) -> NDArray:
    """Oklab float64 -> BGR uint8, clipped to gamut."""
    lig, a, b = oklab[..., 0], oklab[..., 1], oklab[..., 2]
    l_ = (lig + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (lig - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (lig - 0.0894841775 * a - 1.2914855480 * b) ** 3
    lin = np.stack(
        [
            -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,  # B
            -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,  # G
            +4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,  # R
        ],
        axis=-1,
    )
    np.clip(lin, 0, 1, out=lin)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(srgb * 255, 0, 255).astype(np.uint8)


# Per-cell transfer: MKL optimal transport + partial histogram match
def _cov(x: NDArray) -> NDArray:
    """Covariance of an (n, 3) sample: np.cov's arithmetic without np.cov.

    Called twice per cell, where np.cov's argument handling costs three times
    its arithmetic: 76 ms against 24 per 700 cells, and the same numbers on
    every cell and tile in the demo (checked, 4,700 of them)."""
    c = x.T.copy()
    c -= c.mean(axis=1)[:, None]
    return (c @ c.T) * (1.0 / (len(x) - 1))


def _mkl_matrix(cov_src: NDArray, cov_dst: NDArray) -> NDArray:
    """Closed-form Gaussian optimal transport map between two covariances."""
    d_sq, u = np.linalg.eigh(cov_src)
    d = np.sqrt(np.maximum(d_sq, EPS))
    d_inv = np.diag(1.0 / d)
    c_sq, uc = np.linalg.eigh(np.diag(d) @ u.T @ cov_dst @ u @ np.diag(d))
    dc = np.diag(np.sqrt(np.maximum(c_sq, EPS)))
    return u @ d_inv @ uc @ dc @ uc.T @ d_inv @ u.T


def match_histograms(src: NDArray, ref: NDArray) -> NDArray:
    """Rewrite `src` so each channel's histogram becomes `ref`'s.

    Byte images only, and channel-independent: read off both cumulative
    distributions, then look up where each source level lands in the
    reference. scikit-image has this, but one function is a thin reason to
    make every user install it and the six packages behind it; this is bit
    for bit the same answer (checked on 500 tile/cell pairs)."""
    out = np.empty(src.shape, np.float64)
    for c in range(src.shape[-1]):
        level = src[..., c].reshape(-1)
        ref_counts = np.bincount(ref[..., c].reshape(-1))
        # Levels the reference never uses are dropped: an empty level is a flat
        # step in its distribution, and interpolating across one would invent a
        # colour that is not in the target cell.
        ref_levels = np.nonzero(ref_counts)[0]
        src_cdf = np.cumsum(np.bincount(level, minlength=256)) / level.size
        ref_cdf = np.cumsum(ref_counts[ref_levels]) / ref[..., c].size
        out[..., c] = np.interp(src_cdf, ref_cdf, ref_levels)[level].reshape(
            src.shape[:-1]
        )
    return out


def transfer_cell(
    tile_bgr: NDArray,
    cell_bgr: NDArray,
    cell_lab: NDArray,
    strength: float,
    hist_alpha: float,
) -> NDArray:
    """Recolor one tile toward one target cell. This is the vivid look.

    The cell arrives in both spaces because the frame was converted whole: one
    cvtColor for the frame beats one per cell."""
    src = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2LAB).astype(np.float64).reshape(-1, 3)
    dst = cell_lab.reshape(-1, 3).astype(np.float64)
    eye = np.eye(3) * 1e-6
    t = _mkl_matrix(_cov(src) + eye, _cov(dst) + eye)
    moved = (src - src.mean(axis=0)) @ t + dst.mean(axis=0)
    blended = src * (1.0 - strength) + moved * strength
    lab = np.clip(blended, 0, 255).astype(np.uint8).reshape(tile_bgr.shape)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    # Histogram matching is channel-independent, so it runs directly on BGR;
    # the reference implementation's BGR->RGB->BGR round trip is a no-op.
    matched = match_histograms(out, cell_bgr).astype(np.uint8)
    return cv2.addWeighted(out, 1.0 - hist_alpha, matched, hist_alpha, 0)


# Vibrance (Oklch). Statistics are injected, never measured here: freezing
# them per keyframe is what makes the postprocess chain LUT-able and stops
# the saturation from pulsing frame to frame.
def vibrance_oklch(bgr: NDArray, amount: float, chroma_max: float) -> NDArray:
    """Boost low-chroma pixels more than saturated ones, with a soft knee."""
    oklab = bgr_to_oklab(bgr)
    lig, a, b = oklab[..., 0], oklab[..., 1], oklab[..., 2]
    chroma = np.hypot(a, b)

    boost = 1.0 + amount * (1.0 - np.clip(chroma / chroma_max, 0, 1)) ** 2
    chroma_new = chroma * boost

    threshold = chroma_max * 0.85
    over = np.maximum(chroma_new - threshold, 0)
    headroom = chroma_max * 0.5
    chroma_new = np.where(
        chroma_new > threshold, threshold + over / (1.0 + over / headroom), chroma_new
    )
    # cos and sin of the hue are a/chroma and b/chroma, so the angle never has
    # to be taken: rescaling a and b together turns the chroma and keeps the hue.
    scale = np.divide(chroma_new, chroma, out=np.zeros_like(chroma), where=chroma > 0)
    return oklab_to_bgr(np.stack([lig, a * scale, b * scale], axis=-1))


def chroma_percentile(bgr: NDArray) -> float:
    """The one statistic vibrance needs. Frozen at keyframes, reused between."""
    oklab = bgr_to_oklab(bgr)
    chroma = np.hypot(oklab[..., 1], oklab[..., 2])
    safe = chroma[chroma > 0.001]
    return float(np.percentile(safe, 99.0) + EPS) if safe.size else 0.3
