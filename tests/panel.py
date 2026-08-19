"""The five-motion panel: pan, subpix, drift, noise, fade.

Its job is not to check the pretty picture. It is to run the SAME frames down
two paths that must agree bit for bit:

  reference -- sequential, full-frame make-up, nothing shared
  engine    -- dirty-rect make-up on a worker thread, exactly as cli.run_video

This panel, and only this panel, caught a shared-buffer data race that six
straight clean runs of the main clip had missed: subpix and drift failed while
everything else passed. Determinism is checked too, since a race can also show
up as a run-to-run difference.

Like the golden, every input is bundled or generated here from bundled input,
and the geometry is the shipping default (700 cells).

  panel.py            check against panel.json
  panel.py --update   record it"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from sakuhart import mosaic as M  # noqa: E402
from sakuhart import postprocess as P  # noqa: E402
from sakuhart import temporal as T  # noqa: E402
from sakuhart import video as V  # noqa: E402

HERE = Path(__file__).resolve().parent
SAMPLES = Path(M.__file__).resolve().parent / "samples"
TILES = str(SAMPLES / "starter")
SOURCE = str(SAMPLES / "demo_target.jpg")
GOLDEN = HERE / "panel.json"
CELL, W, H, N = 54, 1890, 1080, 10  # the shipping default: 35 x 20 = 700 cells
KINDS = ("pan", "subpix", "drift", "noise", "fade")


def md5(a) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def base_plate() -> np.ndarray:
    """One oversized still, so every panel is a window onto the same picture."""
    img = cv2.imread(SOURCE)
    return cv2.resize(img, (W * 2, H * 2), interpolation=cv2.INTER_LANCZOS4)


def frames_for(kind: str, plate: np.ndarray) -> list[np.ndarray]:
    """The five motions. Each returns N frames of W x H."""
    out = []
    rng = np.random.default_rng(7)
    still = plate[300 : 300 + H, 300 : 300 + W]
    for i in range(N):
        if kind == "pan":  # whole frame moves: every cell is a new cell
            f = plate[300 : 300 + H, 300 + i * 24 : 300 + i * 24 + W]
        elif kind == "subpix":  # below one pixel: the dirty test's worst case
            mat = np.float32([[1, 0, i * 0.3], [0, 1, i * 0.17]])
            f = cv2.warpAffine(still, mat, (W, H))
        elif kind == "drift":  # nothing moves, the light changes
            f = cv2.convertScaleAbs(still, alpha=1.0 + i * 0.012, beta=i * 1.5)
        elif kind == "noise":  # a locked-off camera, sensor grain only
            f = np.clip(still + rng.normal(0, 2.0, (H, W, 3)), 0, 255).astype(np.uint8)
        elif kind == "fade":  # sustained real change: the one that must be here
            f = cv2.convertScaleAbs(still, alpha=1.0 - i / (N + 2))
        out.append(np.ascontiguousarray(f))
    return out


def encode(frames: list[np.ndarray], path: Path) -> None:
    """Through a real codec, so the panel sees the noise the engine will see."""
    proc = subprocess.Popen(
        [V.FFMPEG, "-v", "error", "-y", *V._RAW, "-s", f"{W}x{H}", "-r", "24"]
        + ["-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17"]
        # Pinned: x264 takes its thread count from the core count otherwise, and
        # a different count encodes different bytes -- see .gitignore.
        + ["-threads", "1"]
        + ["-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE,
    )
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"could not build {path.name}")


def reference(frames, pool) -> list[str]:
    """Full-frame make-up, sequential, no shared anything."""
    state, chain = T.TemporalState(), []
    stats = luts = None
    for i, frame in enumerate(frames):
        state.is_keyframe = i == 0
        raw, _, _ = M.render_frame(frame, pool, CELL, state)
        if i == 0:
            stats = P.capture_stats(raw)
            luts = P.build_luts(stats)
        made = P.postprocess(raw.copy(), stats, luts)  # box=None: whole frame
        chain.append(md5(made))
    return chain


def engine(frames, pool) -> list[str]:
    """Exactly cli.run_video: dirty rectangles, make-up one frame behind."""
    state, made, pending, chain = T.TemporalState(), None, None, []
    stats = luts = None
    cols = W // CELL
    with ThreadPoolExecutor(1) as makeup:
        for i, frame in enumerate(frames):
            state.is_keyframe = i == 0
            raw, _, changed = M.render_frame(frame, pool, CELL, state)
            if i == 0:
                stats = P.capture_stats(raw)
                luts = P.build_luts(stats)
            if pending is not None:
                made = pending.result()
                chain.append(md5(made))
                pending = None
            if changed is None or len(changed):
                box = None if changed is None else M.cell_box(changed, cols, CELL)
                pending = makeup.submit(
                    P.postprocess, raw.copy(), stats, luts, made, box
                )
            else:
                chain.append(md5(made))
        if pending is not None:
            chain.append(md5(pending.result()))
    return chain


def main() -> int:
    update = "--update" in sys.argv
    pool = M.load_pool([TILES], CELL)
    plate = base_plate()
    got, fails = {}, []
    for kind in KINDS:
        clip = HERE / f"panel_{kind}.mp4"
        if not clip.exists():
            encode(frames_for(kind, plate), clip)
        info = V.probe(clip)
        frames = [
            f.copy() for f in V.decode(clip, V.VideoInfo(W, H, info.fps, N, False))
        ]
        ref = reference(frames, pool)
        eng = engine(frames, pool)
        eng2 = engine(frames, pool)
        got[kind] = ref
        marks = []
        if eng != ref:
            bad = next(i for i, (a, b) in enumerate(zip(eng, ref)) if a != b)
            fails.append(f"{kind}: dirty path differs from full path at frame {bad}")
            marks.append("DIRTY!=FULL")
        if eng2 != eng:
            bad = next(i for i, (a, b) in enumerate(zip(eng2, eng)) if a != b)
            fails.append(f"{kind}: not deterministic, frame {bad}")
            marks.append("NONDETERMINISTIC")
        print(
            f"  {'FAIL' if marks else 'PASS'}  {kind:7s} {len(ref)} frames  "
            f"{' '.join(marks) or 'dirty==full, run==run'}"
        )
    if update:
        GOLDEN.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n")
        print(f"recorded {GOLDEN.name}")
        return 0
    if GOLDEN.exists():
        want = json.loads(GOLDEN.read_text())
        for kind, chain in got.items():
            ok = want.get(kind) == chain
            print(f"  {'PASS' if ok else 'FAIL'}  {kind:7s} against the recorded chain")
            if not ok:
                fails.append(f"{kind}: chain moved from the recorded golden")
    print(f"\nPANEL: {'all pass' if not fails else 'FAILURES'}")
    for f in fails:
        print(f"  ! {f}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
