"""Golden tests: the smallest set that would catch a regression from here.

Four layers:

  1. placement -- bit exact. Pool arrays, the assignment, the raw canvas.
  2. make-up   -- bit exact md5 AND a perceptual floor (SSIM against a stored
                  reference image), so a future change that is deliberately not
                  bit exact still has to say how far it moved.
  3. palette   -- mean Lab and a coarse Lab histogram of the raw canvas. md5
                  says "different" without saying how; this says whether the
                  colour left the building or a single tile moved.
  4. video     -- the whole frame chain, plus run-to-run determinism.

Every input is either bundled with the project or generated here from bundled
input: the photo pool is samples/starter (CC BY 2.0, credited in
ATTRIBUTION.md), the target is samples/demo_target.jpg, and the clip is a pan
across that target, encoded once and committed. Nothing else needs to exist for
this to run on someone else's machine, which is the whole point of a golden
test -- and the clip is committed rather than rebuilt because libx264 encodes
different bytes on a different core count, which decode to different pixels.

    golden.py            check against the recorded values
    golden.py --update   record them (only after reviewing the change!)"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
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
TARGET = str(SAMPLES / "demo_target.jpg")
CLIP = HERE / "golden_clip.mp4"
GOLDEN = HERE / "golden.json"
REF_IMAGE = HERE / "golden_ref.png"
CELL, W, H, FRAMES = 60, 960, 1080, 8
SSIM_FLOOR = 0.999  # a deliberately-not-bit-exact change must still clear this
PALETTE_L1 = 0.002  # and must not move this much colour mass either


def md5(a) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    x = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    y = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = cv2.GaussianBlur(x, (11, 11), 1.5), cv2.GaussianBlur(y, (11, 11), 1.5)
    xx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mx * mx
    yy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - my * my
    xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mx * my
    num = (2 * mx * my + c1) * (2 * xy + c2)
    den = (mx**2 + my**2 + c1) * (xx + yy + c2)
    return float((num / den).mean())


def palette(img: np.ndarray) -> dict:
    """Where the colour is, coarsely enough that one swapped tile does not move
    it and a changed colour pipeline cannot help but."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3).ravel()
    return {
        "mean_lab": [round(float(v), 3) for v in lab.reshape(-1, 3).mean(0)],
        "hist": [round(float(v), 8) for v in hist / hist.sum()],
    }


def make_clip() -> None:
    """A pan across the bundled target: every cell is a new cell every frame,
    which is the motion that exercises the most of the engine per second."""
    plate = cv2.resize(
        cv2.imread(TARGET), (W * 2, H * 2), interpolation=cv2.INTER_LANCZOS4
    )
    proc = subprocess.Popen(
        [V.FFMPEG, "-v", "error", "-y", *V._RAW, "-s", f"{W}x{H}", "-r", "24"]
        + ["-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17"]
        # Pinned: x264 takes its thread count from the core count otherwise, and
        # a different count encodes different bytes -- see .gitignore.
        + ["-threads", "1"]
        + ["-pix_fmt", "yuv420p", str(CLIP)],
        stdin=subprocess.PIPE,
    )
    for i in range(FRAMES):
        window = plate[300 : 300 + H, 300 + i * 24 : 300 + i * 24 + W]
        proc.stdin.write(np.ascontiguousarray(window).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("could not build the golden clip")


def collect() -> tuple[dict, np.ndarray]:
    """Everything the golden file records, in one pass."""
    got: dict = {}
    pool = M.load_pool([TILES], CELL)
    got["pool"] = {
        n: md5(getattr(pool, n)) for n in ("tiles", "grays", "features", "oklab")
    }
    got["pool_size"] = [len(pool), pool.n_photos]

    img = cv2.imread(TARGET)
    got["geometry"] = {
        "default_700_on_16_9": list(M.plan_geometry(1920, 1080, 700)),
        "270_on_this_target": list(M.plan_geometry(img.shape[1], img.shape[0], 270)),
    }
    target = M.fit_to_grid(img, W, H)  # pinned, so geometry is a separate axis
    raw, assignment, changed = M.render_frame(target, pool, CELL)
    got["still"] = {
        "assignment": md5(assignment),
        "raw": md5(raw),
        "changed_is_none": changed is None,
        "palette": palette(raw),
    }
    stats = P.capture_stats(raw)
    # No table here on purpose: this is exactly what cli.run_still does, and a
    # still that went through the table would be testing the video path.
    final = P.postprocess(raw, stats)
    got["still"]["final"] = md5(final)

    if not CLIP.exists():
        make_clip()
    info = V.probe(CLIP)
    frames = [f.copy() for f in V.decode(CLIP, info)]
    chain, repaints = [], []
    state = T.TemporalState()
    made = None
    cols = W // CELL
    for i, frame in enumerate(frames):
        state.is_keyframe = i == 0
        vraw, _, ch = M.render_frame(frame, pool, CELL, state)
        if i == 0:
            vstats = P.capture_stats(vraw)
            vluts = P.build_luts(vstats)
        repaints.append(None if ch is None else int(len(ch)))
        box = None if ch is None else M.cell_box(ch, cols, CELL)
        made = P.postprocess(vraw, vstats, vluts, made, box)
        chain.append(md5(made))
    got["video"] = {
        "frames": chain,
        "repaints": repaints,
        "decoded": len(frames),
        "lut": md5(vluts),  # the table is a video-only object now
    }
    return got, final


def main() -> int:
    update = "--update" in sys.argv
    got, final = collect()

    if update:
        GOLDEN.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n")
        cv2.imwrite(str(REF_IMAGE), final)
        print(f"recorded {GOLDEN.name} and {REF_IMAGE.name}")
        return 0

    want = json.loads(GOLDEN.read_text())
    fails = []

    def check(name, a, b):
        ok = a == b
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            fails.append(f"{name}: got {a!r}, want {b!r}")

    print("layer 1 -- placement (bit exact)")
    check("pool arrays", got["pool"], want["pool"])
    check("pool size", got["pool_size"], want["pool_size"])
    check("assignment", got["still"]["assignment"], want["still"]["assignment"])
    check("raw canvas", got["still"]["raw"], want["still"]["raw"])
    check("geometry", got["geometry"], want["geometry"])

    print("layer 2 -- make-up (bit exact + perceptual floor)")
    check("final frame md5", got["still"]["final"], want["still"]["final"])
    s = ssim(final, cv2.imread(str(REF_IMAGE)))
    ok = s >= SSIM_FLOOR
    print(f"  {'PASS' if ok else 'FAIL'}  final frame SSIM {s:.6f} >= {SSIM_FLOOR}")
    if not ok:
        fails.append(f"SSIM {s:.6f} below {SSIM_FLOOR}")

    print("layer 3 -- palette")
    now, then = got["still"]["palette"], want["still"]["palette"]
    l1 = float(np.abs(np.array(now["hist"]) - np.array(then["hist"])).sum())
    dl = max(abs(a - b) for a, b in zip(now["mean_lab"], then["mean_lab"]))
    ok = l1 <= PALETTE_L1
    print(f"  {'PASS' if ok else 'FAIL'}  colour moved {l1:.6f} <= {PALETTE_L1}")
    print(f"        mean Lab drifted at most {dl:.3f} of 255")
    if not ok:
        fails.append(f"palette moved {l1:.6f}, mean Lab by {dl:.3f}")

    print("layer 4 -- video")
    check("frames decoded", got["video"]["decoded"], want["video"]["decoded"])
    check("frame chain", got["video"]["frames"], want["video"]["frames"])
    check("repaint counts", got["video"]["repaints"], want["video"]["repaints"])
    again, _ = collect()
    check("run-to-run determinism", again["video"]["frames"], got["video"]["frames"])

    print(f"\nGOLDEN: {'all pass' if not fails else 'FAILURES'}")
    for f in fails:
        print(f"  ! {f}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
