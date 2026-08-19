# How sakuhart works

The README gives the five steps. This is what is underneath them: why each one is there, what the
numbers are, and what was tried and thrown away.

Every constant quoted here was read out of the source while writing this. Measurements are
labelled — **(measured)** was re-run for this document, **(log)** comes from the project's own
records, and **(source)** is quoted from a comment sitting next to the code it measures.

The whole pipeline, in order, with where each part is explained:

```
  the frame  ->  grid  ->  describe  ->  assign  ->  re-pick  ->  separate  ->  recolour  ->  finish
                 §3.1      §3.3         §3.4       §3.5        §3.6        §4           §6
                                          |
  your photos ->  read + cache  ---------/
                  §3.2
                                     video only:  inherit the last frame's answer  §5
```

---

## 1. Why the obvious method fails

A photomosaic is one picture assembled out of many small pictures. Robert Silvers popularised the
form in the 1990s [1]. The obvious way to build one is:

> for every cell of the target, find the photo whose average colour is closest, and paste it.

That produces a picture nobody wants to look at, and the reason is worth stating plainly: **that
rule lets one photo win everywhere it fits.** A clear sky is thousands of cells asking the same
question, so they all get the same answer — your three bluest photos, tiled across a third of the
frame. The result reads as a repeating texture, not a picture, and human vision is unusually good at
spotting exactly that kind of repetition.

Allow each photo to be used **once** and the failure disappears. But the problem changes shape: it
stops being *n* independent lookups and becomes one global decision, because the cells now compete.
A cell that takes its first choice has taken it from someone else. That is the **assignment
problem**, and solving it rather than approximating it is most of what sakuhart is.

One consequence falls out immediately. If a photo covers one cell, a picture of *n* cells needs at
least *n* tiles. sakuhart makes **four tiles out of every photo** — the photo, its mirror, and the
two at ±10% brightness — so the floor is *n*/4 photos. Mirroring and brightness are not padding:
they move the photo far enough in the descriptor for the four variants to land in genuinely
different places, while leaving the subject recognisable.

## 2. Colour: why Oklab

Colours are compared in **Oklab** [2].

What is needed is a colour space where *numerical distance predicts visible difference*. sRGB fails
badly at this — black to dark grey is numerically tiny and visually large. CIELAB is the classical
answer, but its blue region is known to be uneven, and blue is exactly where skies and shadows live.
Oklab was fitted to fix that while staying cheap: a 3×3 matrix, a cube root, another 3×3 matrix.

It is used in four places:

* the colour term that refines the assignment cost,
* the vibrance stage, which works in Oklch (Oklab in polar form) so it can lift chroma without
  touching lightness or hue,
* the pool diagnostic (§7), where the whole point is that the number means something to the eye,
* and the tile swap on video (§5), which uses its own version of the colour term.

The sRGB→linear step inside it is a 256-entry lookup table rather than a `pow()` per pixel: the
input is a byte, so there are only 256 possible answers.

## 3. Choosing which photo goes where

### 3.1 Where the cells come from

Before anything is compared, the frame has to become a grid, and the grid is not free to be any
shape. Three rules decide it:

* **The frame is scaled to the output height** (1080 by default) and its width trimmed to a whole
  number of cells. Because the grid divides the frame exactly, the "build it oversized and resize at
  the end" stage that most implementations need does not exist here.
* **A cell is never smaller than 20px.** Below that a tile stops reading as a photograph, which is
  the one thing the tool cannot trade away. It is also a hard cap on how fine `--cells` can go: at
  height 1080 a 16:9 frame tops out at 5,184 cells and a 9:16 one at 1,620 **(measured)**. Raising
  `--height` is the only way past it, which is why it is an option and not a constant.
* **The cell size must divide the height exactly**, so `--cells` is a *request*. sakuhart picks the
  divisor whose cell count lands nearest to it, at this frame's shape — asking for 700 gives exactly
  700 on a 16:9 frame (54px cells), 729 on a square one and 675 on a 1000×1080 one (40px both).
  Sizing off a
  hardcoded 1920×1080 instead would make `--cells` mean something different at every aspect ratio:
  0.17× on a 9:16 phone clip, 30× on a 100:1 strip.

Two guards sit here rather than later: an odd width is nudged by one cell, because libx264 refuses
an odd side and would otherwise kill the run *after* the pool was built, and an extreme aspect ratio
is refused outright before it asks for a 108,000×1,080 working frame.

### 3.2 Reading the photo folder once

Describing a folder is the slowest thing sakuhart does — a 4000×3000 phone photo costs about 92 ms
to decode against 4 ms to describe — so the shrunken photos and their descriptions are cached.

The cache key is a hash of **every photo's resolved path, size and modification time, plus the cell
size**. That is the whole invalidation story, and it is deliberately not a "last built at" timestamp:
editing, adding or removing a photo changes the key, so a stale cache cannot be read by accident.
Paths are resolved first, or `-t ./photos` and `-t /home/me/photos` would each build their own copy
of the same pool. The cell size is in the key because photos are stored at exactly the size that
setting needs — which is why changing `--cells` can rebuild it.

Writing is atomic — a temporary name first, then a rename — and only finished files are called
`*.npz`, so another process's half-written cache is never picked up. The eight most recent photo
sets are kept, and nothing here is allowed to be fatal: a cache that will not open is a miss, not an
error.

### 3.3 What "close" means

Each cell is described by **191 numbers**, computed in CIELAB over the cell's pixels:

| how many | what it captures |
|---|---|
| 75 | average colour of each of 5×5 blocks — *where* the colour sits inside the cell |
| 36 | 12-bin histogram of each of the 3 channels — the colours present, ignoring position |
| 40 | gradient-orientation histograms, 8 bins, over 4 sub-blocks and the whole cell — edge direction |
| 40 | local binary pattern histograms [3], 10 bins, over 4 sub-blocks — fine texture |

Colour alone cannot tell a cell that is half black and half white from a cell that is uniformly
grey. The gradient and texture terms are what carry that difference. The 5×5 blocks divide the cell by
integer division and drop the remainder, so on a 54px cell the outer 4px sit outside that first
term — kept deliberately, because the golden tests are pinned to it.

The cost between a cell and a tile is the squared distance between their 191-vectors, divided by the
largest value in the matrix so it lands in a sane range. Then, **for the 100 cheapest candidates of
each cell only**, an Oklab colour distance is added at weight **0.20**. Restricting it is
deliberate: colour is a tie-breaker between plausible tiles, not a way for a structurally wrong tile
to win by being the right shade.

Finally every row is multiplied by a **saliency weight**, normalised so the frame averages 1.0:

```
0.2 + 1.5 × (Canny edge density) + 1.0 × (Laplacian-of-Gaussian energy) + 0.5 × (HSV saturation)
```

then scaled by **1.6** inside any face box a Haar cascade [4] finds, or — when no face is found — by
a mild centre bias running from 1.0 at the middle to 0.7 at the corners. Because the solver
minimises a *total*, scaling a row scales how much that cell's mistakes count. The effect is that
good tiles are spent on faces and edges, and flat dark corners are asked to make do.

### 3.4 Solving it, on a thinned graph

The assignment problem is solvable in polynomial time [5][6], but the matrix is `cells × tiles` —
576 cells against a 32,000-tile pool is already 18 million entries, and both sides grow.

Almost all of it is irrelevant: no cell is ever matched to its 20,000th choice. So sakuhart keeps
only the **K cheapest tiles per cell** and solves that sparse problem instead.

This is free only under a condition, and the condition is where the danger is. Throwing edges away
is harmless *provided the winning edges survive and a full matching still exists*. As K shrinks,
both fail — in different ways:

* below roughly **0.5 × cells**, no perfect matching exists in the thinned graph and scipy raises;
* just above that, scipy **does not raise** — it quietly returns a **worse** answer.

The second is the one to design against, because nothing tells you it happened. So K follows the
**cell count** rather than being a fixed number, with a floor for small frames:

```python
k = min(n_tiles, max(400, math.ceil(0.8 * n_cells)))
```

with a doubling retry if scipy raises, and a dense solve as the last resort.

How close the thinned answer is depends on how tight the pool is:

* On a real 576-cell frame against a 2,800-tile pool — a generous 4.9 tiles per cell — the thinned
  solver returned the dense optimum **exactly** (22.5364594744, twelve significant figures) at every K tried,
  from 100 up to the full 2,800 **(measured)**. The silently-worse band never appeared.
* On 2,484 cells the answer stayed identical at 1.6 and 1.3 tiles per cell, then drifted:
  **0.007% worse at 1.09×, 0.02% at 1.03×** **(log)**.

These agree rather than conflict. With five tiles per cell there is so much slack that even a small
K keeps everything that matters. The failure belongs to a *tight* pool — which is exactly the case a
user creates by supplying the bare minimum of photos. **The fix for it is more photographs, not a
bigger K.**

One implementation trap, silent when wrong: scipy's sparse matrices drop stored zeros, so a
zero-cost edge would vanish from the graph and the matching would be solved on a different problem.
sakuhart adds **1.0 to every stored edge**. Every full matching has exactly `n_cells` edges, so all
of them shift by the same amount and the winner is unchanged.

### 3.5 A second look, by shape

The cost is built from statistics, and two photos with the same statistics can look nothing alike —
same histogram, same gradient energy, completely different picture. So after the assignment every
cell gets a second look:

1. take its **25** cheapest candidates,
2. rank them by normalised cross-correlation against the cell in grayscale — one dot product each,
3. score the best **4** with **SSIM** [7],
4. swap only if the challenger beats the incumbent by a factor of **1.002**.

The two-stage shortlist exists because SSIM costs five Gaussian blurs where NCC costs a dot product;
scoring all 25 with SSIM would dominate the frame. The 1.002 margin is hysteresis. Without it, ties
resolve on floating-point noise, which on video means tiles flipping back and forth for no visible
reason. A `used` set is carried through the pass, so one-to-one survives it.

### 3.6 Keeping the twins apart

A photograph enters the pool four times, and the assignment is one-to-one over those **variants**,
not over photographs. So no cell repeats a *tile*, but a *photograph* can appear up to four times.
That is deliberate — four variants of 2,000 photographs beat 8,000 distinct ones, measured, because
a good photograph is worth reusing.

It stops being fine when two of them land side by side: a mirror pair in touching cells is
unmistakable, and the picture reads as a bug rather than a mosaic. At 750 cells from a
2,972-photograph folder there were 33 such pairs **(log)**.

Banning reuse outright fixes it and costs **27% more colour error (log)** — the assignment gives up its
best answer everywhere to prevent a problem that shows in a few places. So the rule is applied only
where it shows: each offending cell trades tiles with whichever cell makes the cheapest swap that
breaks the pair without creating a new one. A trade is a permutation, so one-to-one stays exact, and
the added colour error is **under 0.2% (log)**.

It is best-effort, not a guarantee: a cell looks for a partner only among its own cheap candidates,
and if none can take the trade the pair stays. On the bundled demo — 750 cells, so 188 photographs is the floor
— every clash was cleared: 30 down to 0 at that floor, 18 down to 0 with the bundled 1,000
**(measured)**.

## 4. Making the chosen photo fit

The right photo is still the wrong colour. Closing that gap is easy; closing it *without flattening
the photo into a colour swatch* is the problem — if the tile stops looking like a photograph, the
whole point is gone.

Three methods, in order of how much of the colour distribution they force to match:

| method | what it matches | cost |
|---|---|---|
| per-channel mean and standard deviation [8] | the diagonal of the covariance | trivial |
| linear Monge–Kantorovich [9] | the **full** covariance, cross-channel terms included | two 3×3 eigendecompositions |
| histogram matching | the entire distribution, shape and all | one pass over 256 bins |

sakuhart applies the second at **strength 0.50**, then blends the third over it at **α = 0.30**.

The MKL map is the closed-form optimal transport between two Gaussians: the unique linear `T` with
`T Σ_src Tᵀ = Σ_dst` that minimises transport cost. The code computes it in CIELAB through a
symmetric eigendecomposition, which is the numerically well-behaved way to write
`Σ_src^(-1/2) (Σ_src^(1/2) Σ_dst Σ_src^(1/2))^(1/2) Σ_src^(-1/2)` — that is the two eigendecompositions
in the table above.

**Both are used partially, and that is the entire design.** At full strength each matches its target
exactly and the tile turns into a swatch. Half a covariance match plus a third of a histogram match
puts the tile close enough to read correctly at arm's length while leaving enough of the original to
read as a photograph up close.

The last few percent are bought differently: the target cell is **blended over** the finished tile,
**5% to 14%** by saliency — with a floor of **30% for dark cells** (mean BGR below 60). That floor
is an admission. Photo folders run short of genuinely dark tiles, so dark cells are the ones the
assignment can least afford, and letting the target show through is cheaper than pretending
otherwise.

## 5. Video: holding still

Rendering each frame independently looks wrong even when every individual frame is good. Sensor
noise alone — differences invisible in the source — reshuffles roughly **20%** of a fresh assignment
between consecutive frames **(log)**, and a fifth of the image changing 24 times a second reads as
boiling.

So a frame **inherits** the previous frame's assignment and may only disturb it where the picture
really changed. Which cells moved is decided from cell means alone; the expensive 191-number
descriptions are then built **only for those cells**. Three gates, cheapest first:

| gate | trigger | what it allows |
|---|---|---|
| repaint | the cell's mean moved more than **2.0** per channel | re-run the colour transfer with the **same tile** |
| re-evaluate | drift since the last full look passed **6.0** | the cell may swap tiles |
| safety valve | brightness moved more than **8.0** from when the tile was chosen | swap allowed regardless |

Most changes in real footage are the first kind, and answering them with a repaint instead of a swap
is where the stability comes from: across 7,617 real swaps, **59% improved the picture by less than
1 dE** [10] — visible churn buying nothing **(log)**. A candidate swap has one more gate: it must
cut that cell's cost **below 0.90** of the incumbent's, or it is refused — a tie counts as no gain.

The safety valve exists for slow fades: no single frame moves enough to trip the other gates, so
without it the assignment drifts arbitrarily far from correct while every gate stays shut.

Two things about the swap are easy to assume and wrong. It does **not** re-use the cost from §3.3:
the video path scores candidates on the raw feature distance plus an Oklab term applied to *every*
tile rather than the nearest hundred, and it does not weight by saliency at all. And **saliency
itself is frozen at the keyframe** — the same weights drive both the cost and the blend for every
frame until the next cut. That is the same reasoning as the frozen tone statistics in §6: a weight
that is recomputed per frame makes the whole picture breathe.

**The swap search is greedy on purpose.** It takes each candidate cell in turn and gives it the best
tile nobody is using. The exact re-solve was tried and it is *worse*: churn rises **+4% to +218%**
for a fidelity gain under **0.2 dE** **(log)**. The greedy method's lag is not an error being
tolerated — it acts as temporal smoothing, and the exact solution is optimal for the wrong
objective. Per-frame fidelity is not what a viewer is judging.

Scene cuts are the exception: inherited state is simply wrong across a cut, so everything is rebuilt
from scratch. They are found in a cheap pre-pass before any real work starts: ffmpeg's `scdet`
filter runs with `threshold=0`, which makes it *score* every frame rather than judge them, and
sakuhart calls anything scoring **10 or more** a cut. Ten is ffmpeg's own default on its 0–100 scale.

The scale is worth naming: read as if it were 0–1, that same number once **missed 46 of 47 cuts in
live-action footage (source)**. "Fixing" the 10 to 0.10 is that bug coming back.

## 6. The finishing pass, and how video skips most of it

The finishing chain — gamma and shadow lift, high-frequency lift, vibrance, saturation, contrast,
colour harmony, unsharp — splits cleanly in two, and keeping them apart is what makes video
affordable:

* **Pixel-independent** stages (vibrance → saturation → contrast → harmony) depend only on a pixel's
  own colour plus a few frozen frame statistics. For video they are baked into a **128³ lookup
  table** once per keyframe and afterwards applied by interpolation — seconds of chain become a
  table lookup for every frame until the next cut.
* **Spatial** stages (the high-frequency lift and the unsharp) read neighbouring pixels and cannot
  be tabulated. They run on every frame.

A **still image never builds the table**. 128³ is 2.1 million lattice points and a 1080p frame is
2.0 million pixels, so tabulating costs more arithmetic than colouring the pixels directly — and the
direct answer is the exact one. The table exists to be *reused*, which only happens in video.

The pass also runs **one frame behind, on its own thread**. It is the long pole of a frame and it
needs nothing the next render produces, so the two overlap instead of queueing.

Frame statistics are captured at keyframes and **held constant** until the next one. Recomputing
them per frame makes the saturation pulse: the statistic moves a little, every pixel follows it, and
the whole image breathes.

Video gets one more saving. When only a few cells were repainted, only the part of the frame those
cells can *reach* needs recomputing. The reach is finite and small: the two blurs extend **7 px and
2 px**, so **9 rows** above and below the changed band is everything that can differ. That band is
recomputed and the rest is copied from the previous frame's answer, **byte for byte identical** —
`tests/panel.py` proves it frame by frame against the full-frame result.

Two details make it exact rather than nearly exact:

* The band keeps the **full width of the frame** and only cuts rows. OpenCV's separable blurs vary
  their vectorisation with row length, so a narrower crop comes back 1–3 levels different at
  identical borders. Row length never changes, so cutting rows is free of that.
* Past **70%** of the frame the bookkeeping stops paying and the whole frame goes through.

This has to be exact, not close: a "close enough" dirty-rectangle scheme leaves seams along the
rectangle edges, and seams are precisely the artefact this tool cannot afford.

## 7. How good could this folder possibly get?

`--diagnose` exists because the honest answer to "why does my mosaic look wrong" is usually **your
photos**, not the algorithm — and that claim has to come with a number.

Take the cells' Oklab means as *demand* and the **tiles'** — all four variants of every photo — as
*supply with capacity one*, and solve the
same assignment on colour alone — a partial optimal transport problem with unit capacities [11]. The
result is a **lower bound** on the colour error that *any* arrangement of this folder could reach. It is not the distance between the two colour clouds:
capacity is the whole point, and a folder can hold exactly the right colour and still fail for lack
of copies of it. So the bound is reported in two halves:

* **colour gap** — nearest-neighbour distance, ignoring capacity. *Your folder does not contain this
  colour at all.*
* **stock gap** — the remainder. *It does, and you have run out.*

They want opposite fixes: the first needs *different* photos, the second needs *more* of the ones
already working. The frame is then split into four brightness bands, and each band's supply/demand
ratio and residual error are printed — so the advice is specific instead of "add more photos".

On the reference measurement the bound was **12.52** — 2.52 of it colour, 10.01 stock, the two halves
rounded separately — against an actual placement of **13.52 (log)**. That places the result **8%
above the best any algorithm could have done with that folder**, and nine tenths of the error that
remained belonged to the folder, not to the algorithm.

This also killed a rule that used to be in the README. "Supply at least 1.5× as many tiles as cells"
was withdrawn because the bound falls **monotonically from 1.0× to 15.2× with no knee** **(log)**.
There is no threshold to cross. More photos always help, by steadily less — and printing the real
bound is more useful than any fixed multiplier.

## 8. What did not work

Negative results, kept because they cost time to find.

| tried | result |
|---|---|
| Exact re-solve of the assignment on every video frame | Rejected. Churn +4–218% for under 0.2 dE of fidelity **(log)**. The greedy lag is useful smoothing. |
| Detecting scene cuts from our own churn statistics | Rejected. Oscillates during pans and fires spurious keyframes. ffmpeg's `scdet` looks at the source and does not. |
| A constant K in the thinned assignment | Rejected. Returns a silently worse optimum between roughly 0.55n and 0.68n **(log)** — a wrong answer with no error. |
| CLAHE (contrast-limited adaptive histogram equalisation) | Removed. It survived for a while gated by a texture measure, and measured no better than the plain chain; deleting it also shrank the dirty-rectangle reach from ~200 px to 9 **(log)**. |
| `-shortest` when muxing the audio back | Banned. Silently dropped 2 of 180 frames — audio drift **(log)**. |
| Frame-altering ffmpeg filters (`mpdecimate`, `tblend`, `minterpolate`) | Banned. Frames in must equal frames out. |
| A 256³ colour table | Rejected for video. Exact to 0.0002 of a level where 128³ is off by 0.80 on average — but 13× the build, paid at every cut **(log)**. |
| float64 in the high-frequency stage | Rejected. 3× the cost for a 0.0007% difference **(log)**. |
| A module-level scratch buffer for that stage | Rejected — a data race, since the finishing pass runs on its own thread. Thread-local instead. |
| Per-cell `cv2.Sobel` in the descriptor | Rejected. 3× faster alone and bit-identical **(source)**, but inside the threaded per-cell loop the interpreter lock made video **23% slower** end to end. |
| Releasing OpenCV's own threads (`cv2.setNumThreads(-1)`) globally | Rejected. ~4% on video **(source)**, at the price of results that depend on the machine's core count. Kept only inside face detection, where the output is checked to be identical either way (198 ms → 34). |
| Nested parallelism (BLAS threads inside the parallel loops) | Rejected. Slower than either alone, and OpenBLAS is not bit-reproducible across thread counts, which breaks the golden tests. |
| "Tiles ≥ 1.5 × cells" as advice | Withdrawn — no knee exists (§7). Replaced by `--diagnose`. |
| Per-tile Lab statistics | Not carried over: written, never read. |
| Oklch hue-rotated colour variants (258 lines) | Not carried over: disabled by default and unused. |

## References

Every entry was checked against a primary or author-controlled source.

```bibtex
% [1] the form itself
@book{silvers1997photomosaics,
    title     = {Photomosaics},
    author    = {Robert Silvers},
    year      = {1997},
    publisher = {Henry Holt and Company},
}
@misc{silvers2000patent,
    title  = {Digital composition of a mosaic image},
    author = {Robert S. Silvers},
    year   = {2000},
    note   = {US Patent 6,137,498; filed 1997, granted 24 Oct 2000},
    url    = {https://patents.google.com/patent/US6137498A/en},
}

% [2] the colour space (S2)
@misc{ottosson2020oklab,
    title  = {A perceptual color space for image processing},
    author = {Bj\"orn Ottosson},
    year   = {2020},
    url    = {https://bottosson.github.io/posts/oklab/},
}

% [3] the texture half of the cell descriptor (S3.3)
@article{ojala2002lbp,
    title   = {Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns},
    author  = {Timo Ojala and Matti Pietik\"ainen and Topi M\"aenp\"a\"a},
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
    volume  = {24},
    number  = {7},
    pages   = {971--987},
    year    = {2002},
    doi     = {10.1109/TPAMI.2002.1017623},
}

% [4] face detection behind the saliency weight (S3.3)
@inproceedings{viola2001rapid,
    title     = {Rapid Object Detection using a Boosted Cascade of Simple Features},
    author    = {Paul Viola and Michael Jones},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    volume    = {1},
    pages     = {511--518},
    year      = {2001},
    doi       = {10.1109/CVPR.2001.990517},
}

% [5] the assignment problem (S3.4)
@article{kuhn1955hungarian,
    title   = {The Hungarian method for the assignment problem},
    author  = {Harold W. Kuhn},
    journal = {Naval Research Logistics Quarterly},
    volume  = {2},
    number  = {1--2},
    pages   = {83--97},
    year    = {1955},
    doi     = {10.1002/nav.3800020109},
}

% [6] how it is solved in practice (S3.4)
@article{jonker1987shortest,
    title   = {A shortest augmenting path algorithm for dense and sparse linear assignment problems},
    author  = {Roy Jonker and Anton Volgenant},
    journal = {Computing},
    volume  = {38},
    pages   = {325--340},
    year    = {1987},
    doi     = {10.1007/BF02278710},
}

% [7] the structure score in the second look (S3.5)
@article{wang2004ssim,
    title   = {Image Quality Assessment: From Error Visibility to Structural Similarity},
    author  = {Zhou Wang and Alan C. Bovik and Hamid R. Sheikh and Eero P. Simoncelli},
    journal = {IEEE Transactions on Image Processing},
    volume  = {13},
    number  = {4},
    pages   = {600--612},
    year    = {2004},
    doi     = {10.1109/TIP.2003.819861},
}

% [8] the colour transfer sakuhart does not use (S4)
@article{reinhard2001color,
    title   = {Color Transfer between Images},
    author  = {Erik Reinhard and Michael Ashikhmin and Bruce Gooch and Peter Shirley},
    journal = {IEEE Computer Graphics and Applications},
    volume  = {21},
    number  = {5},
    pages   = {34--41},
    year    = {2001},
    doi     = {10.1109/38.946629},
}

% [9] the colour transfer it does use (S4)
@inproceedings{pitie2007linear,
    title     = {The linear Monge-Kantorovitch linear colour mapping for example-based colour transfer},
    author    = {Fran\c{c}ois Piti\'e and Anil Kokaram},
    booktitle = {4th European Conference on Visual Media Production (CVMP)},
    publisher = {IET},
    year      = {2007},
    doi       = {10.1049/cp:20070055},
    note      = {The word "linear" does appear twice in the published title.},
}

% [10] the colour difference the video gates are measured in (S5)
@article{sharma2005ciede2000,
    title   = {The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and Mathematical Observations},
    author  = {Gaurav Sharma and Wencheng Wu and Edul N. Dalal},
    journal = {Color Research \& Application},
    volume  = {30},
    number  = {1},
    pages   = {21--30},
    year    = {2005},
    doi     = {10.1002/col.20070},
}

% [11] the framing of the pool diagnostic (S7)
@article{peyre2019ot,
    title   = {Computational Optimal Transport},
    author  = {Gabriel Peyr\'e and Marco Cuturi},
    journal = {Foundations and Trends in Machine Learning},
    volume  = {11},
    number  = {5--6},
    pages   = {355--607},
    year    = {2019},
    eprint  = {1803.00567},
    archivePrefix = {arXiv},
    url     = {https://arxiv.org/abs/1803.00567},
}
```
