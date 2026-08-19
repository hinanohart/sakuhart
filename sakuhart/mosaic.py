"""The mosaic core: features -> saliency -> cost -> assignment -> rerank -> twins -> paste.

One quality level ("vivid"), one code path. `render_frame` builds a still image
when `state` is None and a video frame when it is not."""

from __future__ import annotations

import hashlib
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from scipy.ndimage import correlate1d
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from .color import SRGB_TO_LINEAR, bgr_to_oklab, transfer_cell

NDArray = np.ndarray

# The vivid recipe for placement and paste. These constants ARE the product;
# there is no preset system. The make-up constants live in postprocess.py.
MKL_STRENGTH = 0.50  # LAB optimal-transport strength
HIST_ALPHA = 0.30  # histogram match blended over the transported tile
BLEND_MIN = 0.05  # target-image bleed-through at low saliency
BLEND_MAX = 0.14  # ... and at high saliency
BLEND_DARK_FLOOR = 0.30  # dark cells need more bleed: tile stock runs out there
BLEND_DARK_LEVEL = 60  # mean BGR below which a cell counts as dark
OKLAB_COST_WEIGHT = 0.20
OKLAB_COST_NEAR = 100  # the Oklab term only refines the near candidates
RERANK_CANDIDATES = 25
RERANK_SHORTLIST = 4  # of those, the ones cheap NCC likes get the SSIM test
RERANK_MARGIN = 1.002  # hysteresis: a swap must beat the incumbent by this much

OUT_HEIGHT = 1080
MAX_WORK_PIXELS = 64_000_000  # the ceiling video.MAX_PIXELS puts on the input,
# applied again to the working frame: the height is fixed, so a wide enough
# source multiplies past the input guard on its way to the grid.


# Any cell size that divides the working height exactly. The width is trimmed
# to a whole number of cells anyway, so dividing the width is not a constraint
# -- which is what lets the same ladder serve a 9:16 phone clip. Snapping here
# removes the whole "build oversized, resize at the end" stage the reference
# implementation needs. Below 20px a tile stops reading as a photograph.
def cell_sizes(out_height: int) -> tuple[int, ...]:
    """Cell sizes that tile this height exactly."""
    return tuple(px for px in range(20, out_height + 1) if out_height % px == 0)


QUADRANTS, HIST_BINS, GRAD_BINS, LBP_BINS = 5, 12, 8, 10
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
AUGMENT_CHUNK = 512  # photos brightened at a time; caps the float32 scratch
DESCRIBE_WORKERS = 8  # tiles are described independently
DESCRIBE_PIXELS = 8_000_000  # tile pixels described at a time; caps it again
CACHE_VERSION = 1  # bump when the stored arrays stop meaning the same thing
CACHE_KEEP = 8  # photo sets remembered before the least recent is dropped


@dataclass
class Pool:
    """A loaded tile pool. Arrays, not lists: indexing a pool is a hot path."""

    tiles: NDArray  # (n, px, px, 3) uint8
    grays: NDArray  # (n, px, px)    uint8
    features: NDArray  # (n, 191)       float32
    oklab: NDArray  # (n, 3)         float64
    n_photos: int  # tiles before the x4 augmentation, for error messages

    # Derived, because the per-frame swap search would otherwise rebuild them
    # every single frame: measured 78 ms/frame at a 32,000 tile pool.
    features64: NDArray = field(init=False)
    feat_sq: NDArray = field(init=False)
    oklab_sq: NDArray = field(init=False)

    def __post_init__(self) -> None:
        self.features64 = self.features.astype(np.float64)
        self.feat_sq = (self.features64**2).sum(1)
        self.oklab_sq = (self.oklab**2).sum(1)

    def __len__(self) -> int:
        return len(self.tiles)


# Grid
def plan_geometry(
    width: int, height: int, requested_cells: int, out_height: int = OUT_HEIGHT
) -> tuple[int, int, int]:
    """Choose the working size and cell size for a source of any shape.

    The frame is scaled to `out_height` and its width trimmed to a whole number
    of cells, so the grid divides the frame exactly and the "build oversized,
    resize at the end" stage never has to exist.

    The height caps how fine the grid can be, because a cell is never smaller
    than 20px: at 1080 a 9:16 source tops out at 1,620 cells whatever
    --cells asks for. Raising the height is the only way past that, which is
    why it is an option rather than a constant."""
    sizes = cell_sizes(out_height)
    if not sizes:
        raise SystemExit(
            f"a {out_height}px tall frame has no whole cell size of 20px or "
            f"more, because a cell below that stops reading as a photograph."
        )
    if out_height % 2:
        raise SystemExit(
            f"--height {out_height} is odd, and H.264 will not encode an odd "
            f"side. Use an even height."
        )
    scaled_w = max(sizes[0], int(round(width * out_height / height)))
    px = snap_cell_size(requested_cells, scaled_w, out_height)
    out_w = max(px, scaled_w - scaled_w % px)
    # An odd cell size can leave an odd width -- 1917x1080 at --cells 3000 --
    # and libx264 refuses that outright ("width not divisible by 2"), so the
    # video dies after the pool has been built. One column is the whole cost.
    if out_w % 2:
        out_w = out_w - px if out_w > px else out_w + px
    # An extreme aspect ratio buys its width in full: a 100:1 strip asks for
    # 108000x1080, which the input-side guard never sees.
    if out_w * out_height > MAX_WORK_PIXELS:
        raise SystemExit(
            f"a {width}x{height} source becomes a {out_w}x{out_height} working "
            f"frame ({out_w * out_height / 1e6:.0f} megapixels), past the "
            f"{MAX_WORK_PIXELS // 10**6} megapixel guard. Crop it closer to "
            f"16:9, or lower --height."
        )
    return px, out_w, out_height


def snap_cell_size(requested_cells: int, width: int, out_height: int) -> int:
    """Nearest cell size that gives `requested_cells` cells at THIS frame shape.

    Neither size has a default, on purpose. Sizing off a hardcoded 1920x1080 makes
    --cells mean a different number at every other aspect ratio: 0.17x on a
    9:16 phone clip, 30x on a 100:1 strip -- the second of which asks for a
    pool of thousands of tiles nobody meant to request."""
    counts = {px: (width // px) * (out_height // px) for px in cell_sizes(out_height)}
    best = min(counts, key=lambda px: (abs(counts[px] - requested_cells), px))
    return best


def fit_to_grid(img: NDArray, out_w: int, out_h: int) -> NDArray:
    """Scale to the working height, then centre-crop the width."""
    h, w = img.shape[:2]
    resized = cv2.resize(
        img,
        (max(out_w, int(round(w * out_h / h))), out_h),
        interpolation=cv2.INTER_LANCZOS4,
    )
    left = (resized.shape[1] - out_w) // 2
    return resized[:, left : left + out_w]


# Features: 191 dims per cell, all cells at once
def _cells(img: NDArray, cols: int, rows: int, px: int) -> NDArray:
    """(H,W,C) -> (rows*cols, px, px, C) without copying pixel data twice."""
    c = img.shape[2]
    v = img.reshape(rows, px, cols, px, c).transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(v).reshape(rows * cols, px, px, c)


SMOOTH = np.array([1.0, 2.0, 1.0], np.float32)
DIFF = np.array([-1.0, 0.0, 1.0], np.float32)


def _sobel_batch(l_ch: NDArray) -> tuple[NDArray, NDArray]:
    """3x3 Sobel over a stack of cells, reflect-101 edges (as cv2 does).

    Sobel is separable -- smooth along one axis, difference along the other --
    so it is two 1-D passes per direction instead of a 3x3 window. Byte
    luminance widened to float32 makes every partial sum an exact integer, so
    the split is bit for bit the 3x3 answer (checked over random cells).

    Calling cv2.Sobel per cell instead is 3x quicker measured alone and 23%
    slower in place: this runs on four threads, and a Python loop over small
    cv2 calls spends its life handing the interpreter lock back and forth."""
    down = correlate1d(l_ch, SMOOTH, axis=1, mode="mirror")
    across = correlate1d(l_ch, SMOOTH, axis=2, mode="mirror")
    gx = correlate1d(down, DIFF, axis=2, mode="mirror")
    return gx, correlate1d(across, DIFF, axis=1, mode="mirror")


def _hist(v: NDArray, bins: int, hi: float, scale: float, w=None) -> NDArray:
    """Per-row histogram of `v` over [0, hi), normalised and scaled.

    Every row is offset into one flat bincount, so all of them are counted in
    a single pass: the same numbers as np.histogram per row, ~2x faster. `w`
    weights each sample, which is what lets this serve the gradient histogram
    (magnitude weighted) as well as the plain colour and LBP ones."""
    n = len(v)
    idx = np.clip((v * (bins / hi)).astype(np.int64), 0, bins - 1)
    idx += np.arange(n)[:, None] * bins
    h = np.bincount(idx.ravel(), weights=w, minlength=n * bins).reshape(n, bins)
    return (h / (h.sum(1, keepdims=True) + 1e-6) * scale).astype(np.float32)


def extract_features(cells_lab: NDArray) -> NDArray:
    """191-dim descriptor per cell: quadrant colour, histograms, gradients, LBP."""
    n, px = cells_lab.shape[0], cells_lab.shape[1]
    q = px // QUADRANTS  # the reference truncates the remainder; keep that
    k = q * QUADRANTS
    quad = cells_lab[:, :k, :k].reshape(n, QUADRANTS, q, QUADRANTS, q, 3)
    feat_quad = quad.mean(axis=(2, 4)).reshape(n, 75)

    flat = cells_lab.reshape(n, px * px, 3)
    feat_hist = np.concatenate(
        [_hist(flat[:, :, c], HIST_BINS, 255.0, 50.0) for c in range(3)], axis=1
    )

    gx, gy = _sobel_batch(cells_lab[..., 0])
    mag, ang = np.hypot(gx, gy), np.arctan2(gy, gx)

    # Orientation histogram, magnitude weighted: the angle shifted into
    # [0, 2pi) is exactly what `_hist` already bins.
    def grad(a: NDArray, m: NDArray) -> NDArray:
        return _hist(a.reshape(n, -1) + np.pi, GRAD_BINS, 2 * np.pi, 30.0, m.ravel())

    feat_grad = np.concatenate(
        [grad(ang[:, sy, sx], mag[:, sy, sx]) for sy, sx in _blocks(px)]
        + [grad(ang, mag)],
        axis=1,
    )

    lbp = _lbp_batch(np.clip(cells_lab[..., 0], 0, 255).astype(np.uint8))
    feat_lbp = np.concatenate(
        [
            _hist(lbp[:, sy, sx].reshape(n, -1), LBP_BINS, 256.0, 20.0)
            for sy, sx in _blocks(lbp.shape[1])
        ],
        axis=1,
    )
    return np.concatenate([feat_quad, feat_hist, feat_grad, feat_lbp], axis=1).astype(
        np.float32
    )


def _blocks(size: int) -> list[tuple[slice, slice]]:
    h = size // 2
    return [
        (slice(0, h), slice(0, h)),
        (slice(0, h), slice(h, 2 * h)),
        (slice(h, 2 * h), slice(0, h)),
        (slice(h, 2 * h), slice(h, 2 * h)),
    ]


def _lbp_batch(gray: NDArray) -> NDArray:
    """8-neighbour local binary pattern over a stack of cells."""
    c = gray[:, 1:-1, 1:-1]
    out = np.zeros_like(c)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    h, w = gray.shape[1], gray.shape[2]
    for i, (dy, dx) in enumerate(offsets):
        nb = gray[:, 1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]
        out |= (nb >= c).astype(np.uint8) << i
    return out


# Saliency (keyframe only)
_FACE_CASCADE: cv2.CascadeClassifier | None = None


def _face_boxes(gray: NDArray) -> NDArray:
    """Haar face boxes. Bundled with OpenCV, so this costs no new dependency.

    The cascade is the one place worth handing the cores back to OpenCV: it is
    a scan over image scales that returns the same boxes on any thread count
    (checked), and it drops from 198 ms to 34 at 1080p -- pinned, it would be
    a sixth of a keyframe by itself. The count is restored either way, so
    nothing outside this call sees a different OpenCV."""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(str(path))
    threads = cv2.getNumThreads()
    cv2.setNumThreads(-1)  # -1 restores the default, which is every core; 0 is off
    try:
        return _FACE_CASCADE.detectMultiScale(gray, 1.2, 5, minSize=(48, 48))
    finally:
        cv2.setNumThreads(threads)


def saliency_weights(bgr: NDArray, cols: int, rows: int, px: int) -> NDArray:
    """Per-cell importance, mean-normalised to 1.0. Drives cost and blend alpha."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    log = np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (5, 5), 1.5), cv2.CV_64F))
    log /= log.max() + 1e-6
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float64) / 255.0

    def per_cell(a: NDArray) -> NDArray:
        return a.reshape(rows, px, cols, px).mean(axis=(1, 3))

    weights = (
        0.2
        + per_cell((edges > 0).astype(np.float64)) * 1.5
        + per_cell(log) * 1.0
        + per_cell(sat) * 0.5
    )
    # Faces beat the centre-bias heuristic whenever they are found.
    faces = _face_boxes(gray)
    if len(faces):
        boost = np.ones((rows, cols))
        for x, y, w, h in faces:
            boost[y // px : (y + h) // px + 1, x // px : (x + w) // px + 1] = 1.6
    else:
        gy, gx = np.mgrid[0:rows, 0:cols]
        dist = np.hypot(gy - rows / 2.0, gx - cols / 2.0)
        boost = 1.0 - 0.3 * (dist / dist.max())
    weights *= boost
    return weights / (weights.mean() + 1e-6)


# Cost + assignment
def sq_dists(a: NDArray, b: NDArray, b_sq: NDArray | None = None) -> NDArray:
    """|a-b|^2 for every pair of rows, expanded as |a|^2 + |b|^2 - 2ab so the
    (rows_a, rows_b, dims) difference is never built. `b_sq` is the pool side
    when it has already been computed. Negatives are rounding, so they clamp."""
    sq = (b**2).sum(1) if b_sq is None else b_sq
    d = (a**2).sum(1)[:, None] + sq[None, :] - 2.0 * (a @ b.T)
    np.maximum(d, 0, out=d)
    return d


def cost_matrix(
    cell_feat: NDArray, cell_oklab: NDArray, pool: Pool, sal: NDArray
) -> NDArray:
    """Feature L2 plus an Oklab term on the near candidates, saliency weighted."""
    d2 = sq_dists(cell_feat, pool.features)
    cost = (d2 / (d2.max() + 1e-6)).astype(np.float64)
    k = min(OKLAB_COST_NEAR, len(pool))
    near = np.argpartition(cost, k - 1, axis=1)[:, :k]
    rows = np.arange(len(cost))[:, None]
    cost[rows, near] += (
        np.linalg.norm(cell_oklab[:, None, :] - pool.oklab[near], axis=2)
        * OKLAB_COST_WEIGHT
    )
    cost *= sal.reshape(-1, 1)
    return cost


def assign(cost: NDArray) -> NDArray:
    """Exact 1:1 assignment via a sparsified LAP, widening K until it is feasible.

    Sparsifying to the K cheapest tiles per cell reaches the same optimum as
    the dense solve while K is large enough for a perfect matching to survive.
    Below roughly 0.5x the cell count scipy raises outright, and between there
    and ~0.7x it silently returns a worse optimum -- so K is pinned to the cell
    count, never to a constant.

    Measured against `linear_sum_assignment` on 2,484 real cells, it is exact
    while the pool has room: identical at 1.6 and 1.3 tiles per cell, 0.007%
    worse at 1.09 and 0.02% at 1.03, where the cheapest K stop covering the
    tiles a perfect matching has to use. More photos, not a bigger K."""
    n_cells, n_tiles = cost.shape
    k = min(n_tiles, max(400, math.ceil(0.8 * n_cells)))
    while k < n_tiles:
        try:
            return _sparse_lap(cost, k)
        except ValueError:
            k = min(n_tiles, k * 2)
    return linear_sum_assignment(cost)[1].astype(np.int32)


def _sparse_lap(cost: NDArray, k: int) -> NDArray:
    n_cells = len(cost)
    idx = np.argpartition(cost, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n_cells), k)
    # +1.0 keeps every stored edge non-zero (scipy drops implicit zeros) and
    # shifts every full matching by the same constant, so the argmin is intact.
    vals = cost[np.arange(n_cells)[:, None], idx].ravel() + 1.0
    sparse = csr_matrix((vals, (rows, idx.ravel())), shape=cost.shape)
    _, col = min_weight_full_bipartite_matching(sparse)
    return col.astype(np.int32)


# Rerank: NCC shortlist, then SSIM
def _ssim(a: NDArray, b: NDArray) -> float:
    """SSIM with the canonical 11x11 Gaussian window (sigma 1.5)."""
    x, y = a.astype(np.float32), b.astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    xx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    yy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * xy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (xx + yy + c2)
    return float((num / den).mean())


def _ncc(cell_norm: NDArray, tile: NDArray) -> float:
    t = tile.reshape(-1).astype(np.float32)
    return float(cell_norm @ ((t - t.mean()) / (t.std() + 1e-8)) / cell_norm.size)


TWIN_CANDIDATES = 240  # cells considered as a swap partner, cheapest first


def separate_twins(
    assignment: NDArray, cost: NDArray, pool: Pool, cols: int
) -> NDArray:
    """Stop one photograph from landing in two touching cells.

    A photo enters the pool four times -- itself, mirrored, and at two
    brightnesses -- and the assignment is 1:1 over those variants, not over
    photographs. That is deliberate: reusing a good photograph beats reaching
    for a worse one, and it costs 27% more colour error to forbid. But when the
    two copies land side by side the mirror pair is unmistakable, and the
    picture reads as a bug rather than a mosaic.

    So the constraint is only where it shows: touching cells. Each offending
    cell trades tiles with whichever cell makes the cheapest swap that breaks
    the pair without making a new one. The 1:1 property is preserved exactly --
    a swap is a permutation -- and the cost it adds is a few hundredths."""
    out = assignment.copy()
    n_cells, n = len(out), pool.n_photos
    rows = n_cells // cols
    photo = out % n
    where = {int(t): i for i, t in enumerate(out)}  # tile -> the cell holding it

    def neighbours(i):
        y, x = divmod(i, cols)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                yield ny * cols + nx

    def clashes(i, tile):  # would `tile` touch its own photo at cell i?
        p = tile % n
        return any(photo[j] == p for j in neighbours(i))

    offenders = [i for i in range(n_cells) if clashes(i, int(out[i]))]
    if not offenders:
        return out
    # A small pool has fewer tiles than the shortlist wants, and argpartition
    # raises rather than clamping.
    k = min(TWIN_CANDIDATES, cost.shape[1])
    # Only the offending rows are ever read, and the partition is per row.
    order = dict(zip(offenders, np.argpartition(cost[offenders], k - 1, axis=1)[:, :k]))
    for i in offenders:
        ti = int(out[i])
        if not clashes(i, ti):  # an earlier swap already fixed this one
            continue
        best, gain = None, 0.0
        for tj in order[i].tolist():
            j = where.get(tj)
            if j is None or j == i:
                continue
            tj = int(out[j])
            # Both halves of the swap have to land clean, and the two cells
            # must not be neighbours of each other -- swapping a pair with
            # itself moves the clash rather than removing it.
            if j in set(neighbours(i)) or clashes(i, tj) or clashes(j, ti):
                continue
            delta = cost[i, tj] + cost[j, ti] - cost[i, ti] - cost[j, tj]
            if best is None or delta < gain:
                best, gain = j, float(delta)
        if best is None:
            continue
        j, tj = best, int(out[best])
        out[i], out[j] = tj, ti
        photo[i], photo[j] = tj % n, ti % n
        where[tj], where[ti] = i, j
    return out


def rerank(
    assignment: NDArray, cost: NDArray, cells_gray: NDArray, pool: Pool
) -> NDArray:
    """Re-pick each cell's tile by structure, not just colour statistics."""
    out = assignment.copy()
    used = set(out.tolist())
    # Six photos make 24 tiles, which is fewer than the shortlist wants, and
    # argpartition raises rather than clamping -- the same edge the cost and
    # twin stages already guard.
    k = min(RERANK_CANDIDATES, cost.shape[1])
    order = np.argpartition(cost, k - 1, axis=1)[:, :k]
    for i, cell in enumerate(cells_gray):
        current = int(out[i])
        cand = [c for c in order[i].tolist() if c != current and c not in used]
        if not cand:
            continue
        flat = cell.reshape(-1).astype(np.float32)
        flat = (flat - flat.mean()) / (flat.std() + 1e-8)
        best, best_score = current, _ssim(cell, pool.grays[current]) * RERANK_MARGIN
        shortlist = sorted(cand, key=lambda c: -_ncc(flat, pool.grays[c]))
        for c in shortlist[:RERANK_SHORTLIST]:
            score = _ssim(cell, pool.grays[c])
            if score > best_score:
                best, best_score = c, score
        if best != current:
            used.discard(current)
            used.add(best)
            out[i] = best
    return out


# Paste
def blend_alpha(sal: float, dark: bool) -> float:
    """How much target image bleeds through a cell."""
    a = BLEND_MIN + (BLEND_MAX - BLEND_MIN) * (min(max(sal, 0.5), 2.0) - 0.5) / 1.5
    return max(a, BLEND_DARK_FLOOR) if dark else a


def paste_cells(
    canvas: NDArray,
    cells_bgr: NDArray,
    cells_lab: NDArray,
    assignment: NDArray,
    pool: Pool,
    sal_flat: NDArray,
    cols: int,
    px: int,
    which: NDArray | None = None,
) -> None:
    """Recolour and write the given cells into `canvas` in place."""
    todo = range(len(assignment)) if which is None else which
    for i in todo:
        cell = cells_bgr[i]
        tile = transfer_cell(
            pool.tiles[assignment[i]], cell, cells_lab[i], MKL_STRENGTH, HIST_ALPHA
        )
        a = blend_alpha(float(sal_flat[i]), dark=bool(cell.mean() < BLEND_DARK_LEVEL))
        y, x = (i // cols) * px, (i % cols) * px
        canvas[y : y + px, x : x + px] = cv2.addWeighted(tile, 1.0 - a, cell, a, 0)


# One frame
CELL_WORKERS = 4  # per-cell work is split this many ways


def by_cells(fn, cells: NDArray) -> NDArray:
    """Run a per-cell function over chunks of a cell stack, on threads.

    Every cell is described from its own pixels alone, so the pieces glue back
    to the same bytes as one call (checked). Measured at 1080p: the mean Oklab
    of 700 cells is 408 ms serial and 147 ms here; their features, 219 and 86.
    Four, not twelve: the chunks are large and the work is memory bound, so
    more threads only add copies (measured worse at eight)."""
    if len(cells) < CELL_WORKERS:
        return fn(cells)
    with ThreadPoolExecutor(CELL_WORKERS) as ex:
        return np.concatenate(list(ex.map(fn, np.array_split(cells, CELL_WORKERS))))


def cell_oklab(cells_bgr: NDArray) -> NDArray:
    """Mean Oklab of every cell: a cube root per channel per pixel, and the
    most expensive thing a keyframe does outside the solver."""
    return by_cells(lambda chunk: bgr_to_oklab(chunk).mean(axis=(1, 2)), cells_bgr)


def cell_features(cells_lab: NDArray) -> NDArray:
    """The 191-dim descriptor of every cell, from uint8 Lab cells."""
    return by_cells(lambda chunk: extract_features(chunk.astype(np.float32)), cells_lab)


def render_frame(
    target: NDArray, pool: Pool, px: int, state=None
) -> tuple[NDArray, NDArray, NDArray | None]:
    """Build the raw (pre-postprocess) mosaic for one frame.

    `state` is None for a still image and a TemporalState for video; that is
    the only difference between the two modes. Returns the canvas, the tile
    chosen per cell, and which cells were repainted (None = all of them, which
    is what lets the make-up pass skip the rest of the frame)."""
    h, w = target.shape[:2]
    # The one guard worth its lines: a frame that does not divide into whole
    # cells is not an error anywhere below, it is a quietly cropped mosaic.
    if h % px or w % px or not (h and w):
        raise SystemExit(
            f"a {w}x{h} frame does not divide into {px}px cells. Size it with "
            f"plan_geometry() and fit_to_grid() before rendering."
        )
    cols, rows = w // px, h // px
    cells_bgr = _cells(target, cols, rows, px)
    # Lab stays uint8 here: a keyframe widens the whole stack to float32, an
    # intermediate frame only the handful of cells it looks at, and the paste
    # wants bytes either way.
    cells_lab = _cells(cv2.cvtColor(target, cv2.COLOR_BGR2LAB), cols, rows, px)
    if state is None or state.is_keyframe:
        sal = saliency_weights(target, cols, rows, px)
        oklab = cell_oklab(cells_bgr)
        cost = cost_matrix(cell_features(cells_lab), oklab, pool, sal)
        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)[..., None]
        assignment = separate_twins(
            rerank(assign(cost), cost, _cells(gray, cols, rows, px)[..., 0], pool),
            cost,
            pool,
            cols,
        )
        changed = None  # paste every cell
        if state is not None:
            state.reset(cells_bgr, assignment, sal)
    else:
        # No features here: only the handful of cells that moved need them.
        sal = state.saliency
        assignment, changed = state.update(cells_bgr, cells_lab, pool)

    # The canvas is the video's memory: cells nobody repainted stay as they were.
    canvas = (
        np.zeros((h, w, 3), np.uint8)
        if state is None or state.canvas is None
        else state.canvas
    )
    if state is not None:
        state.canvas = canvas
    paste_cells(
        canvas, cells_bgr, cells_lab, assignment, pool, sal.ravel(), cols, px, changed
    )
    return canvas, assignment, changed


def cell_box(changed: NDArray, cols: int, px: int) -> tuple[int, int, int, int] | None:
    """Pixel rectangle (y0, y1, x0, x1) enclosing the given cell indices.

    None when nothing changed. `postprocess` reads None as "do the whole
    frame", which is wasted work but never a wrong picture -- and that is a
    better default than the ValueError an empty min() would raise."""
    if not len(changed):
        return None
    ys, xs = changed // cols, changed % cols
    return (
        int(ys.min()) * px,
        (int(ys.max()) + 1) * px,
        int(xs.min()) * px,
        (int(xs.max()) + 1) * px,
    )


# Pool loading
def _linear_resize(bgr: NDArray, px: int) -> NDArray:
    """Downscale in linear light. Averaging gamma-encoded pixels darkens edges."""
    lin = SRGB_TO_LINEAR[bgr]
    small = cv2.resize(lin, (px, px), interpolation=cv2.INTER_AREA)
    srgb = np.where(
        small <= 0.0031308, small * 12.92, 1.055 * small ** (1 / 2.4) - 0.055
    )
    return np.clip(srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _photo_files(tile_dirs: Sequence[str | Path]) -> list[Path]:
    """Every image under every given folder. A folder can be dropped in twice
    (or nested inside another), hence the set."""
    return sorted(
        {
            p
            for d in tile_dirs
            for p in Path(d).rglob("*")
            if p.suffix.lower() in IMAGE_SUFFIXES
        }
    )


def _read_photos(files: Sequence[Path], tile_dirs, px: int) -> NDArray:
    """Decode, centre-crop and shrink every photo to one (n, px, px, 3) stack.

    The slowest step by far on a real photo folder: a 4000x3000 phone picture
    costs 92 ms to decode against 4 ms to describe, so this is what the cache
    is really for."""
    tiles = []
    for path in files:
        img = cv2.imread(str(path))
        if img is None:
            continue
        side = min(img.shape[:2])  # centre crop keeps aspect, squashing does not
        top, left = (img.shape[0] - side) // 2, (img.shape[1] - side) // 2
        tiles.append(_linear_resize(img[top : top + side, left : left + side], px))
    if not tiles:
        # SystemExit, like the other user-facing errors: a wrong folder is a
        # typo, and a typo does not deserve a traceback.
        raise SystemExit(
            f"no photos found in {', '.join(str(d) for d in tile_dirs)} -- looked in "
            f"every subfolder for {' '.join(IMAGE_SUFFIXES)}. Drop some photos in "
            f"and run again."
        )
    return np.stack(tiles)


def _augment(photos: NDArray, px: int) -> NDArray:
    """Each photo again mirrored and at +/-10% brightness. Four variants of
    2,000 photos match better than 8,000 distinct ones (measured)."""
    n = len(photos)
    base = np.empty((4 * n, px, px, 3), np.uint8)
    base[:n], base[n : 2 * n] = photos, photos[:, :, ::-1]
    # The brightness pair is built a slice at a time. A float32 view of the
    # whole photo stack is four bytes per pixel where the stack is one, i.e. as
    # large as the finished pool -- it was the memory peak, and nothing needs it
    # all at once.
    for lo in range(0, n, AUGMENT_CHUNK):
        hi = min(lo + AUGMENT_CHUNK, n)
        block = base[lo:hi].astype(np.float32)
        for q, gain in ((2, 1.10), (3, 0.90)):
            base[q * n + lo : q * n + hi] = np.clip(block * gain, 0, 255).astype(
                np.uint8
            )
    return base


def _describe(base: NDArray, px: int) -> tuple[NDArray, NDArray]:
    """The 191-dimension feature and the mean Oklab colour of every tile.

    Both are per tile, so they are taken a slice at a time. The scratch is
    linear in the slice and dwarfs the answer -- measured 0.83 GB for 8,000
    tiles, so ~3.3 GB to describe 32,000 at once and hand back 24 MB. A slice
    asks for a few hundred MB and gives the same numbers."""
    # The slice is divided by the worker count as well, so the scratch peak is
    # what it always was and the four pieces are simply in flight at once.
    step = max(1, DESCRIBE_PIXELS // (px * px * DESCRIBE_WORKERS))

    def one(lo: int) -> tuple[NDArray, NDArray]:
        chunk = base[lo : lo + step]
        lab = cv2.cvtColor(chunk.reshape(-1, px, 3), cv2.COLOR_BGR2LAB)
        return (
            extract_features(lab.reshape(chunk.shape).astype(np.float32)),
            bgr_to_oklab(chunk).mean(axis=(1, 2)),
        )

    with ThreadPoolExecutor(DESCRIBE_WORKERS) as ex:
        parts = list(ex.map(one, range(0, len(base), step)))
    return (
        np.concatenate([f for f, _ in parts]),
        np.concatenate([o for _, o in parts]),
    )


def _cache_file(files: Sequence[Path], px: int) -> Path:
    """One file per photo set and cell size.

    Editing, adding or removing a photo changes a size or a modification time,
    which changes this name -- that is the whole invalidation story, and it
    cannot go stale the way a "last built at" timestamp can.

    Paths are resolved first, or `--tiles ./photos` and `--tiles /home/me/photos`
    would each build and keep their own copy of the same pool."""
    key = hashlib.sha256(f"{CACHE_VERSION} {px}".encode())
    for f in sorted(p.resolve() for p in files):
        st = f.stat()
        key.update(f"{f} {st.st_size} {st.st_mtime_ns} ".encode())
    # Where each platform expects a cache to live, then the portable fallback.
    root = (
        os.environ.get("XDG_CACHE_HOME")
        or os.environ.get("LOCALAPPDATA")
        or Path.home() / ".cache"
    )
    return Path(root) / "sakuhart" / f"{key.hexdigest()[:32]}.npz"


def _cache_read(path: Path) -> tuple[NDArray, NDArray, NDArray] | None:
    """The three expensive arrays, or None. A cache that will not open is a
    miss and never an error: the pool can always be built again."""
    try:
        with np.load(path) as z:
            return z["photos"], z["features"], z["oklab"]
    except Exception:  # noqa: BLE001 -- missing, truncated, foreign: all a miss
        return None


def _cache_write(path: Path, photos: NDArray, feats: NDArray, oklab: NDArray) -> None:
    """Save atomically, then keep the directory from growing without end.

    Not being able to write costs the next run some time and nothing else, so
    a read-only or full disk is passed over in silence."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.part")  # not *.npz: see below
        with open(tmp, "wb") as fh:
            np.savez(fh, photos=photos, features=feats, oklab=oklab)
        os.replace(tmp, path)
        # Only finished files are named *.npz, so another process's half-written
        # cache is never a candidate here.
        done = sorted(path.parent.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        for stale in done[:-CACHE_KEEP]:
            stale.unlink(missing_ok=True)
    except OSError:
        pass


def load_pool(tile_dirs: Sequence[str | Path], px: int) -> Pool:
    """Read photo folders into a tile pool: every subfolder, centre-cropped.

    Second and later runs over the same folder come back from the cache, which
    stores the shrunken photos and their descriptions -- everything that costs
    real time. What it does not store is the x4 augmentation and the greyscale
    copy, which are a fraction of a second to rebuild and four times the bytes
    to keep."""
    files = _photo_files(tile_dirs)
    cached = _cache_read(_cache_file(files, px)) if files else None
    if cached is None:
        photos = _read_photos(files, tile_dirs, px)
        base = _augment(photos, px)
        features, oklab = _describe(base, px)
        _cache_write(_cache_file(files, px), photos, features, oklab)
    else:
        photos, features, oklab = cached
        base = _augment(photos, px)
    # One tall image instead of a Python loop: cvtColor is per pixel, so this
    # is the same bytes out, measured 1.7x faster.
    grays = cv2.cvtColor(base.reshape(-1, px, 3), cv2.COLOR_BGR2GRAY).reshape(
        base.shape[:3]
    )
    return Pool(base, grays, features, oklab, len(photos))
