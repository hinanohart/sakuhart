"""Command line entry point.

    sakuhart clip.mp4 --tiles ./photos -o out.mp4
    sakuhart portrait.jpg --tiles ./photos ./more_photos -o out.png
    sakuhart clip.mp4 --tiles ./photos --diagnose

A still image is a one-frame video, so both take the same path. Anything the
user can get wrong is raised as SystemExit: a clean line, not a traceback."""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from . import mosaic as mosaic
from . import __version__
from . import postprocess as post
from . import temporal as temporal
from . import video as video

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
DIAGNOSE_CHUNK = 64  # cells compared against the whole pool at a time
# 4:4:4, because chroma subsampling smears exactly the tile edges we are selling
JPEG_PARAMS = [
    cv2.IMWRITE_JPEG_QUALITY, 95,
    cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
]  # fmt: skip


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("sakuhart", description="Photomosaic video renderer")
    p.add_argument("--version", action="version", version=f"sakuhart {__version__}")
    p.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="image or video to rebuild from photos",
    )
    p.add_argument(
        "-t",
        "--tiles",
        type=Path,
        nargs="+",
        help="photo folders, searched recursively",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="render the bundled painting from the bundled photos, no arguments needed",
    )
    p.add_argument("-o", "--output", type=Path, help="output file")
    p.add_argument(
        "--cells",
        type=int,
        default=700,
        help="approximate cell count; snapped to a size that divides the frame",
    )
    p.add_argument(
        "--height",
        type=int,
        default=mosaic.OUT_HEIGHT,
        help=f"output height, default {mosaic.OUT_HEIGHT}; raise it when --cells "
        f"asks for a finer grid than a 20px cell allows at this height",
    )
    p.add_argument(
        "--diagnose",
        action="store_true",
        help="report how much of the colour error is the photo pool's fault",
    )
    return p


def run_still(
    args, img, pool: mosaic.Pool, cell_px: int, size: tuple[int, int]
) -> None:
    t0 = time.perf_counter()
    target = mosaic.fit_to_grid(img, *size)
    raw = mosaic.render_frame(target, pool, cell_px)[0]
    stats = post.capture_stats(raw)
    final = post.postprocess(raw, stats)  # one frame: no table to amortise
    out = args.output or args.input.with_name(args.input.stem + "_mosaic.png")
    jpeg = out.suffix.lower() in (".jpg", ".jpeg")
    # imwrite reports failure by returning False and refuses an extension it does
    # not know by raising: a missing folder or `-o out.mp4` on a still would
    # otherwise print "wrote" over a file that is not there, and exit 0.
    try:
        written = cv2.imwrite(str(out), final, JPEG_PARAMS if jpeg else [])
    except cv2.error:
        written = False
    if not written:
        raise SystemExit(
            f"cannot write {out} -- check that the folder exists and that the "
            f"suffix is an image format ({', '.join(mosaic.IMAGE_SUFFIXES)})."
        )
    # The video path prints its time; a still that asked for a fine grid can sit
    # here for a minute, and a number afterwards is the only thing that tells
    # the user whether that was normal.
    print(f"wrote {out} in {time.perf_counter() - t0:.1f}s")


def run_video(args, pool: mosaic.Pool, cell_px: int, info: video.VideoInfo) -> None:
    """`info` already carries the working size: scaling to the grid is one more
    ffmpeg filter, so frames arrive ready and we never resize one ourselves.

    The make-up pass runs one frame behind, on its own thread: it is the long
    pole of a frame and it needs nothing the next render produces, so the two
    overlap. The canvas is copied on the way in because the next render writes
    straight back into it."""
    cuts = video.find_cuts(args.input, info.fps)
    out = args.output or args.input.with_name(args.input.stem + "_mosaic.mp4")
    cols = info.width // cell_px
    state = temporal.TemporalState()
    stats = lut = made = None
    pending = None
    t0 = time.perf_counter()

    with video.Encoder(out, info) as enc, ThreadPoolExecutor(1) as makeup:
        for i, frame in enumerate(video.decode(args.input, info)):
            state.is_keyframe = keyframe = i == 0 or i in cuts
            raw, _, changed = mosaic.render_frame(frame, pool, cell_px, state)
            if keyframe:
                stats = post.capture_stats(raw)
                lut = post.build_luts(stats)
            if pending is not None:  # join the oldest, then queue this one
                made = pending.result()
                enc.write(made)
                pending = None
            if changed is None or len(changed):
                box = (
                    None if changed is None else mosaic.cell_box(changed, cols, cell_px)
                )
                pending = makeup.submit(
                    post.postprocess, raw.copy(), stats, lut, made, box
                )
            else:
                enc.write(made)  # nothing moved, and the chain is pure
            if i % 24 == 0:
                # n_frames is the container's own claim and containers lie --
                # a stream copy or a phone's variable frame rate leaves it
                # stale. Show it as the estimate it is; the decoder decides.
                print(
                    f"  frame {i}/~{info.n_frames}  {time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
        if pending is not None:
            enc.write(pending.result())

        # One frame in, one frame out -- the loop above cannot do otherwise, so
        # there is nothing here to warn about. What is worth saying is when the
        # container's own frame count was wrong, because that is what the
        # progress line was counting towards.
        if info.n_frames and enc.count != info.n_frames:
            print(
                f"note: {args.input.name} claimed {info.n_frames} frames and "
                f"decoded to {enc.count}; the output has {enc.count}"
            )
        enc.finish(args.input, info.has_audio)
    print(f"wrote {out} in {time.perf_counter() - t0:.1f}s")


def diagnose(pool: mosaic.Pool, target: np.ndarray, cell_px: int) -> None:
    """How good could ANY assignment be with this pool? (see pool_bound)"""
    cols, rows = target.shape[1] // cell_px, target.shape[0] // cell_px
    demand = mosaic.bgr_to_oklab(mosaic._cells(target, cols, rows, cell_px)).mean(
        axis=(1, 2)
    )
    bound, nearest, bands = pool_bound(demand, pool.oklab)
    print(f"pool bound     {bound:.2f}  (unreachable colour error, in Oklab x100)")
    print(f"  colour gap   {nearest:.2f}  the pool has no such colour")
    print(f"  stock gap    {bound - nearest:.2f}  the colour exists, the copies do not")
    short = []
    for lo, hi, ratio, resid in bands:
        flag = ""
        if ratio < 1.0:
            flag = "  <-- short"
            short.append(f"L {lo:.2f}-{hi:.2f}")
        print(
            f"  L {lo:.2f}-{hi:.2f}: supply/demand {ratio:5.2f}  residual {resid:5.2f}{flag}"
        )
    if short:
        print(
            f"\nAdd photos whose overall brightness falls in {', '.join(short)} "
            f"(L is 0=black to 1=white).\nMore photos in the bands that are "
            f"already full will not improve this picture."
        )


def pool_bound(demand: np.ndarray, supply: np.ndarray) -> tuple[float, float, list]:
    """Lower bound on the colour error, given that each photo is used once.

    This is a partial optimal transport problem: cells are demand, photos are
    supply with capacity one. Because capacity matters, the answer is NOT the
    distance between the two colour clouds -- a pool can contain the right
    colour and still fail for lack of copies. Splitting the bound into
    "colour missing" and "copies missing" tells the user which one to fix."""
    # A row at a time. The whole difference at once is (cells x tiles x 3)
    # float64 -- 442 MB at 576 cells and 32,000 tiles, three times what the
    # renderer itself uses, to produce a matrix an eighth of that size.
    cost = np.empty((len(demand), len(supply)))
    for lo in range(0, len(demand), DIAGNOSE_CHUNK):
        block = demand[lo : lo + DIAGNOSE_CHUNK]
        cost[lo : lo + DIAGNOSE_CHUNK] = (
            np.linalg.norm(block[:, None, :] - supply[None, :, :], axis=2) * 100.0
        )
    nearest = float(cost.min(axis=1).mean())
    assignment = mosaic.assign(np.asarray(cost, np.float64))  # already float64: no copy
    bound = float(cost[np.arange(len(demand)), assignment].mean())
    bands = []
    edges = [0.0, 0.35, 0.55, 0.75, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        d = (demand[:, 0] >= lo) & (demand[:, 0] < hi)
        s = (supply[:, 0] >= lo) & (supply[:, 0] < hi)
        if d.sum():
            resid = float(cost[d, :][np.arange(d.sum()), assignment[d]].mean())
            bands.append((lo, hi, s.sum() / d.sum(), resid))
    return bound, nearest, bands


DEMO_TARGET = Path(__file__).with_name("samples") / "demo_target.jpg"
DEMO_TILES = Path(__file__).with_name("samples") / "starter"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.demo:
        # The whole point is that this needs nothing: no photos to collect, no
        # paths to get right, no ffmpeg. One image, out of the box.
        args.input = args.input or DEMO_TARGET
        args.tiles = args.tiles or [DEMO_TILES]
        # Without this the bundled painting's mosaic lands beside the painting,
        # which lives inside the installed package -- somewhere no one looks,
        # and somewhere a non-root install cannot write at all.
        if args.input == DEMO_TARGET:
            args.output = args.output or Path("demo_mosaic.png")
        for path in (args.input, *args.tiles):
            if path.exists():
                continue
            # Two different failures wear the same shape here: a bundled file
            # that did not survive the install, and a path the user typed wrong.
            # Saying "bundled sample" about the second one sends them hunting in
            # the wrong place.
            if path in (DEMO_TARGET, DEMO_TILES):
                raise SystemExit(
                    f"--demo needs the bundled sample at {path}, which is missing. "
                    f"Pass your own with an input file and -t instead."
                )
            raise SystemExit(f"cannot find {path}")
    if args.input is None or not args.tiles:
        raise SystemExit(
            "give an image or video and -t with at least one photo folder, "
            "or run `sakuhart --demo` to see it work with the bundled samples."
        )
    img = src = None
    if args.input.suffix.lower() in VIDEO_SUFFIXES:
        video.require_ffmpeg()
        src = video.probe(args.input)
        h, w = src.height, src.width
    else:
        img = cv2.imread(str(args.input))
        if img is None:
            raise SystemExit(f"cannot read {args.input}")
        h, w = img.shape[:2]

    cell_px, out_w, out_h = mosaic.plan_geometry(w, h, args.cells, args.height)
    n_cells = (out_w // cell_px) * (out_h // cell_px)
    print(f"{w}x{h} -> {out_w}x{out_h}, {cell_px}px cells -> {n_cells} cells")

    pool = mosaic.load_pool(args.tiles, cell_px)
    if len(pool) < n_cells:
        raise SystemExit(
            f"{pool.n_photos} photos make {len(pool)} tiles (4 variants each), but "
            f"this mosaic needs {n_cells}, one per cell. Add photos, or lower --cells."
        )
    # From here on the decoder hands us frames already at the working size.
    work = replace(src, width=out_w, height=out_h) if src is not None else None

    if args.diagnose:
        frame = (
            next(video.decode(args.input, work))
            if img is None
            else mosaic.fit_to_grid(img, out_w, out_h)
        )
        diagnose(pool, frame, cell_px)
    elif work is not None:
        run_video(args, pool, cell_px, work)
    else:
        run_still(args, img, pool, cell_px, (out_w, out_h))
    return 0


if __name__ == "__main__":
    sys.exit(main())
