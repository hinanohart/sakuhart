"""Video in, video out. Everything container-shaped is delegated to ffmpeg.

Decoding video ourselves would mean handling variable frame rate, rotation
metadata and pixel formats -- three sources of silent audio desync. ffmpeg
already does all of it, so we ask it for a raw BGR stream at a constant rate
and never think about containers again.

Hard rules learned the hard way:
  * never pass -shortest (it silently dropped 2 of 180 frames = audio drift)
  * never use a frame-altering filter (mpdecimate, tblend, minterpolate)
  * emit exactly as many frames as we were given
  * rate-convert in the filter graph only, never at the muxer as well"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NDArray = np.ndarray

QUEUE_DEPTH = 32  # deeper than it looks like it needs: shallow queues stall
QUEUE_TIMEOUT = 5.0  # seconds to wait for the writer thread while tearing down
PREFETCH = 3
# ffmpeg's scdet score is 0-100 and its own default cut threshold is 10. A
# real cut in live footage scores 10.7 and up; panning tops out around 7.6,
# so 10 separates them with room on both sides. (This was 35 -- a 0-1 scale
# read as if it were 0-100 -- which missed 46 of 47 cuts in a live-action
# supercut. Missing a cut is not a speed problem: the keyframe never moves,
# so one scene's colour statistics make up the whole film.)
SCENE_SCORE = 0.10  # x100 below; ffmpeg's own default
MAX_PIXELS = 64_000_000  # decompression-bomb guard
CRF = 17  # x264 quality: visually lossless on this kind of high-detail frame


# Fixed argument runs, kept out of the calls so each call reads as one line.
# Only paths and sizes are interpolated, and never by splitting a string.
def _tool(name: str) -> str:
    """Find ffmpeg/ffprobe: the system copy first, then a pip-installed one.

    Whatever is on PATH wins -- a user who installed ffmpeg deliberately gets
    that build. Failing that, imageio-ffmpeg carries its own ffmpeg binary, so
    `pip install` alone can be enough for the encoder. It does not ship ffprobe, and video needs
    both, so that fallback only ever covers a machine that has ffprobe but not ffmpeg."""
    found = shutil.which(name)
    if found:
        return found
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    return name  # let the failure come from the call, with its own message


FFMPEG = _tool("ffmpeg")
FFPROBE = _tool("ffprobe")

_PROBE = [FFPROBE, *"-v error -show_streams -show_format -print_format json".split()]
_SCDET_VF = "scdet=threshold=0,metadata=print:file=-"
_SCDET_OUT = "-f null -".split()
_RAW = "-pix_fmt bgr24 -f rawvideo".split()
_NULL_IN = "-f lavfi -i nullsrc=s=16x16:d=0.04".split()
_NULL_OUT = "-f null -".split()
_PASSTHROUGH: list[str] = []  # filled once by passthrough()
_X264 = f"-c:v libx264 -preset medium -crf {CRF} -pix_fmt yuv420p".split()
_COPY_AV = "-map 0:v:0 -map 1:a:0 -c:v copy -c:a copy".split()


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int
    has_audio: bool


def passthrough() -> list[str]:
    """The flag for "do not rate-convert; the filter graph already did".

    ffmpeg renamed it from -vsync to -fps_mode in 5.0, and 4.4 is still what
    Ubuntu 22.04 installs. Encoding one null frame asks the binary directly
    which spelling it takes -- surer than reading a version string, which git
    builds do not carry. Guessing wrong fails silently: an unknown option
    makes ffmpeg exit before writing a byte, and decode() would see no frames."""
    if not _PASSTHROUGH:
        modern = "-fps_mode passthrough".split()
        probe = subprocess.run(
            [FFMPEG, "-v", "error", *_NULL_IN, *modern, *_NULL_OUT],
            capture_output=True,
        )
        _PASSTHROUGH.extend(modern if probe.returncode == 0 else ["-vsync", "0"])
    return _PASSTHROUGH


def require_ffmpeg() -> None:
    if (
        not shutil.which(FFMPEG)
        and not Path(FFMPEG).exists()
        or not shutil.which(FFPROBE)
    ):
        raise SystemExit(
            "sakuhart needs ffmpeg and ffprobe on PATH for video. "
            "Images work without them."
        )


def _rate(text: str | None) -> float:
    """An ffprobe "num/den" rate as a float, or 0.0 if it says nothing."""
    num, _, den = (text or "").partition("/")
    try:
        return float(num) / float(den) if float(den or 0) else 0.0
    except ValueError:
        return 0.0


def probe(path: str | Path) -> VideoInfo:
    """Read stream metadata without decoding a single frame.

    A file the user picked by mistake is a typo, not a bug: ffprobe's exit
    status becomes the same one-line message the image path gives, never a
    traceback out of subprocess."""
    done = subprocess.run([*_PROBE, str(path)], capture_output=True, text=True)
    try:
        parsed = json.loads(done.stdout)
        streams = parsed["streams"]
        v = next(s for s in streams if s["codec_type"] == "video")
    except (ValueError, KeyError, StopIteration):
        raise SystemExit(
            f"cannot read {path} -- ffprobe found no video stream in it "
            f"(is the file complete?)"
        ) from None
    # Containers write 0/0 for an average they never computed -- a stream
    # muxed without a duration, most webm. Left at zero it is not an error
    # anywhere: `fps=0` makes ffmpeg refuse to open the decoder, decode()
    # returns no frames, and the run ends by announcing an empty file.
    fps = _rate(v.get("avg_frame_rate")) or _rate(v.get("r_frame_rate"))
    if not fps:
        raise SystemExit(f"{path} does not declare a frame rate to rebuild it at")
    w, h = int(v["width"]), int(v["height"])
    if w * h > MAX_PIXELS:
        raise SystemExit(f"{w}x{h} exceeds the {MAX_PIXELS} pixel guard")
    n = int(v.get("nb_frames") or 0)
    if not n:  # not all containers store it; derive from duration
        n = int(round(fps * float(parsed.get("format", {}).get("duration", 0))))
    return VideoInfo(w, h, fps, n, any(s["codec_type"] == "audio" for s in streams))


def find_cuts(path: str | Path, fps: float) -> set[int]:
    """Locate scene cuts in one cheap pre-pass, before any real work starts.

    Detecting cuts from our own churn statistics oscillates during pans and
    fires spurious keyframes; ffmpeg's scdet looks at the source and does not.

    The rate filter comes first so that "frame 30" here means the same picture
    as "frame 30" in `decode`. Without it a variable-rate phone clip is counted
    in its own uneven frames and every cut lands a few frames off."""
    proc = subprocess.run(
        [FFMPEG, "-v", "info", "-i", str(path), "-vf", f"fps={fps},{_SCDET_VF}"]
        + _SCDET_OUT,
        capture_output=True,
        text=True,
    )
    # A failed pre-pass returns no score lines, which reads exactly like "no
    # cuts" -- and a film rebuilt as one scene is the failure this whole
    # function exists to prevent. Better to say so than to look successful.
    if proc.returncode:
        raise SystemExit(
            f"ffmpeg could not scan {path} for scene cuts:\n"
            f"{proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'no output'}"
        )
    cuts, frame = set(), 0
    for line in proc.stdout.splitlines():
        if "lavfi.scd.score" in line:
            if float(line.rsplit("=", 1)[1]) >= SCENE_SCORE * 100:
                cuts.add(frame)
            frame += 1
    return cuts


def decode(path: str | Path, info: VideoInfo):
    """Yield frames as BGR uint8 arrays at a constant frame rate, already
    scaled and cropped to the working grid.

    bgr24 is requested directly: asking for rgb24 and converting is both
    slower and the classic way to ship a red/blue-swapped release.

    The fps filter already delivers a constant rate, so the muxer must not
    rate-convert on top of it: its default pass rounds the first timestamp of
    a stream-copied cut into a second copy of frame 0, and every frame after
    that sits one frame late against the audio (measured: a 24-frame cut came
    out 25 frames long, with frames 0 and 1 identical)."""
    vf = (
        f"fps={info.fps},scale={info.width}:{info.height}"
        f":force_original_aspect_ratio=increase,crop={info.width}:{info.height}"
    )
    size = info.width * info.height * 3
    proc = subprocess.Popen(
        [FFMPEG, "-v", "error", "-i", str(path), "-vf", vf, *passthrough(), *_RAW, "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # held back so it cannot land mid-progress-line
        bufsize=size * PREFETCH,
    )
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) == size:
                yield np.frombuffer(buf, np.uint8).reshape(info.height, info.width, 3)
            elif buf:
                raise SystemExit(f"{path} ends inside a frame -- the file is cut short")
            else:
                break
        # An ffmpeg that failed on the first frame gives a clean pipe and no
        # frames, which every stage downstream would happily accept: the run
        # would announce an empty video. Its exit status is the only warning.
        if proc.wait() != 0:
            last = (proc.stderr.read().decode(errors="replace") or "").strip()
            raise SystemExit(
                f"ffmpeg could not decode {path} (exit {proc.returncode}): "
                f"{last.splitlines()[-1] if last else 'no message'}"
            )
    finally:
        # kill() before closing the pipe: a caller that stops early (--diagnose
        # reads one frame) would otherwise get "broken pipe" shouted at them.
        proc.kill()
        proc.stdout.close()
        proc.stderr.close()
        proc.wait()


class Encoder:
    """x264 writer on its own thread, so encoding overlaps rendering.

    Use it as a context manager. Leaving the block without `finish()` -- an
    exception, a Ctrl-C -- kills ffmpeg and removes the half-written file
    instead of leaving both behind."""

    def __init__(self, path: Path, info: VideoInfo) -> None:
        # ".part.mp4", not ".mp4.part": ffmpeg picks its muxer from the final
        # extension, and an unknown one makes it refuse to open the file.
        self.dst = path
        self.tmp = path.with_suffix(".part" + path.suffix)
        self.mux = path.with_suffix(".mux" + path.suffix)
        self.proc = subprocess.Popen(
            [FFMPEG, "-v", "error", "-y", *_RAW, "-s", f"{info.width}x{info.height}"]
            + ["-r", str(info.fps), "-i", "-", *_X264, str(self.tmp)],
            stdin=subprocess.PIPE,
        )
        self.q: queue.Queue = queue.Queue(maxsize=QUEUE_DEPTH)
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()
        self.count = 0
        self.done = False

    def __enter__(self) -> Encoder:
        return self

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False

    def _pump(self) -> None:
        """Feed ffmpeg. If it dies, keep draining -- a producer already blocked
        on a full queue has to wake up and see the error, not wait forever."""
        try:
            while (frame := self.q.get()) is not None:
                self.proc.stdin.write(frame.tobytes())
        except BaseException as exc:  # noqa: BLE001 -- re-raised in write()
            self.error = exc
            while self.q.get() is not None:
                pass

    def _check(self) -> None:
        if self.error is None:
            return
        raise SystemExit(
            f"the encoder stopped after {self.count} frames "
            f"(ffmpeg exit {self.proc.poll()}): {type(self.error).__name__}. "
            f"Out of disk space, or ffmpeg was killed."
        )

    def write(self, frame: NDArray) -> None:
        self._check()  # before the put, so a dead pump can never block us
        self.q.put(frame)
        self.count += 1

    def finish(self, src: Path, has_audio: bool) -> None:
        """Drain, mux the original audio back in, and publish atomically."""
        self.q.put(None)
        self.thread.join()
        self._check()
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise SystemExit(
                f"ffmpeg failed to encode {self.dst} (exit {self.proc.returncode})"
            )
        done = self.tmp
        if has_audio:
            done = self.mux
            mux = subprocess.run(
                [FFMPEG, "-v", "error", "-y", "-i", str(self.tmp), "-i", str(src)]
                + [*_COPY_AV, str(done)],
                capture_output=True,
                text=True,
            )
            if mux.returncode:
                # The picture is already encoded and correct; only the audio
                # copy failed. Keep it -- close() would delete minutes of work
                # over a codec the container will not take.
                kept = self.dst.with_name(
                    f"{self.dst.stem}_video_only{self.dst.suffix}"
                )
                os.replace(self.tmp, kept)
                self.mux.unlink(missing_ok=True)  # ffmpeg's half-written attempt
                self.done = True  # nothing left for close() to clean up
                last = (mux.stderr or "").strip().splitlines()
                raise SystemExit(
                    f"cannot copy the audio of {src} into {self.dst} "
                    f"(ffmpeg exit {mux.returncode}): "
                    f"{last[-1] if last else 'no message'}. "
                    f"The video itself is finished and saved as {kept}."
                )
            self.tmp.unlink()
        os.replace(done, self.dst)  # atomic: readers never see a partial file
        self.done = True

    def close(self) -> None:
        """Stop ffmpeg and drop the partial files. Idempotent; a no-op after a
        successful finish()."""
        if self.done:
            return
        self.done = True
        if self.proc.poll() is None:
            self.proc.kill()
        try:  # wake the pump wherever it is waiting
            self.q.put(None, timeout=QUEUE_TIMEOUT)
        except queue.Full:
            pass
        self.thread.join(timeout=QUEUE_TIMEOUT)
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.wait()
        self.tmp.unlink(missing_ok=True)
        self.mux.unlink(missing_ok=True)
