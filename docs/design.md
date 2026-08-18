# BlazeFaceONNX Design

This document describes the design of BlazeFaceONNX: what the exported models
contain, how the postprocessing is formulated in ONNX-native terms, and the
reasoning behind the main design decisions. Read it alongside the code in
`blazefaceonnx/` — the document explains the design; the code is the
authoritative reference for the implementation.

## Goal

BlazeFace, like most SSD-style detectors, does not produce usable detections
directly. The network emits raw per-anchor regressions and logits, and turning
them into final boxes requires anchor decoding, score activation, and — in
MediaPipe's pipeline — *weighted* non-maximum suppression. When a BlazeFace
network is exported to ONNX on its own, every consumer has to reimplement this
postprocessing: carry an anchor table, port the decode math, and reproduce
weighted NMS, which no standard runtime provides.

BlazeFaceONNX removes that burden by baking the entire postprocessing chain
into the ONNX graph. The end-to-end model maps an input image directly to
final detections; the only things a consumer supplies besides the image are
three runtime-tunable thresholds.

## Model set

Two BlazeFace variants are exported, each as two ONNX files:

| Variant | Input | Anchors | Use case |
|---------|-------|---------|----------|
| `front` | 128×128 | 896 | front-facing camera, close-range faces |
| `back`  | 256×256 | 896 | rear-facing camera, farther/smaller faces |

- **Base model** (`blazeface_{variant}_base.onnx`) — the convolutional network
  only: image → `raw_boxes (1, 896, 16)` + `raw_scores (1, 896, 1)`. For users
  who want to apply their own postprocessing.
- **End-to-end model** (`blazeface_{variant}_e2e.onnx`) — network plus full
  postprocessing: image → `detections (S, 17)`, where each row is
  `[ymin, xmin, ymax, xmax, 6×(kp_x, kp_y), score]` in normalized `[0, 1]`
  coordinates and `S` is dynamic (possibly 0).

The network weights come from
[zmurez/MediaPipePyTorch](https://github.com/zmurez/MediaPipePyTorch)
(vendored as the git submodule `external/MediaPipePytorch`), which is Zak
Murez's PyTorch conversion of Google's original MediaPipe TFLite models.

## Architecture

The package has three modules, one per concern:

- `blazefaceonnx/anchor_decode.py` — the anchor-decoding ONNX function.
- `blazefaceonnx/weighted_nms.py` — the IoU-matrix and weighted-NMS ONNX
  functions.
- `blazefaceonnx/export.py` — the export pipeline: exports the base network
  from PyTorch, builds the postprocessing graph, and merges the two into the
  end-to-end model.

The postprocessing is authored with [onnxscript](https://github.com/microsoft/onnxscript)
as ONNX *functions* rather than exported from PyTorch. The PyTorch reference
implementation in the submodule is treated purely as a behavioral
specification: each stage is re-derived as a vectorized graph of standard
ONNX operators (opset 20), not traced from Python control flow.

The end-to-end assembly connects the base model's `raw_boxes`/`raw_scores`
outputs to a postprocessing chain of three stages:

1. **Anchor decoding** — raw regressions → absolute coordinates.
2. **Score processing** — logits clipped to `[-100, 100]`, then sigmoid.
3. **Weighted NMS** — final selection and blending.

The per-variant anchor table and decode scale are embedded in the graph as
constants (initializers), so they are invisible to the consumer. The three
NMS parameters are deliberately left as runtime inputs instead:

| Input | Type | Meaning |
|-------|------|---------|
| `max_output_boxes` | int64 scalar | maximum number of detections |
| `iou_threshold` | float32 scalar | IoU threshold for suppression and blend grouping |
| `score_threshold` | float32 scalar | minimum confidence |

Anchors are a fixed property of the network architecture; thresholds are
application-tuning knobs. Baking the former and exposing the latter gives a
model that is both self-contained and tunable without re-export.

## Anchor decoding

`decode_boxes` implements the decoding performed by MediaPipe's
`TfLiteTensorsToDetectionsCalculator`. Each of the 896 anchors is described
as `[cx, cy, w, h]`; each raw regression row holds a box center/size offset
followed by six keypoint offsets, all in input-pixel units. Decoding divides
by the input scale (128 or 256) and transforms by the anchor:

```
cx = raw_cx / scale * a_w + a_cx    w = raw_w / scale * a_w
cy = raw_cy / scale * a_h + a_cy    h = raw_h / scale * a_h
```

The box is then converted to corner format `[ymin, xmin, ymax, xmax]` — the
layout the ONNX `NonMaxSuppression` operator expects — and the 12 keypoint
values are decoded with the same anchor transform in one broadcast by
reshaping them to `(N, 6, 2)` against anchor centers/sizes of shape
`(N, 1, 2)`.

## Weighted NMS

MediaPipe's face pipeline uses *weighted* NMS: instead of merely discarding
boxes that overlap a higher-scoring winner, it blends each winner's
coordinates with its overlapping candidates using score-weighted averaging.
This stabilizes boxes and keypoints across video frames. The original
algorithm is greedy and iterative — pop the highest-scoring candidate,
gather all remaining candidates with IoU above the threshold, average them,
remove them from the pool, repeat — which does not map naturally onto a
dataflow graph.

### Decoupled formulation

The key observation is that weighted NMS factors into two independent
sub-problems:

1. **Winner selection.** Which boxes anchor a final detection? This is
   exactly standard NMS, so the implementation delegates it to the native
   ONNX `NonMaxSuppression` operator (`center_point_box=0`) rather than
   re-deriving suppression logic. `max_output_boxes`, `iou_threshold`, and
   `score_threshold` pass straight through to the operator.

2. **Blending.** For each winner, define its cluster as every candidate whose
   IoU with the winner exceeds `iou_threshold` *and* whose score exceeds
   `score_threshold` (the winner itself always qualifies). All 16 regression
   values are averaged over the cluster with the candidates' scores as
   weights. This is fully vectorized: an `(S, N)` IoU matrix between the `S`
   winners and all `N` candidates, masked and multiplied against scores,
   yields blend weights that reduce over the candidate axis in one shot.

The final score of each detection is the winner's own score — matching the
original MediaPipe behavior, and deliberately *not* the score-averaging done
by the PyTorch port (see [Design decisions](#design-decisions)).

### Equivalence with the iterative algorithm

The decoupled formulation is not a literal transcription of the greedy loop:
in the iterative version a blended candidate is consumed and can never
contribute to a later winner, whereas here a candidate contributes to every
winner it overlaps. The two behaviors coincide in practice:

- NMS guarantees winners are mutually non-overlapping (pairwise IoU at or
  below the threshold), so their clusters are largely disjoint; a candidate
  can only be shared between winners in narrow geometric configurations.
- For 0 or 1 final detections the results are exactly identical, and
  BlazeFace scenes typically contain very few faces.

The formulation also accepts a deliberate redundancy: IoU is computed once
inside `NonMaxSuppression` and again for the blend mask. Sharing that work
would require reimplementing suppression manually; recomputing a `(S, N)`
matrix for tiny `S` is far cheaper than giving up the native operator.

### Empty results

`NonMaxSuppression` may select nothing. The function branches on the
selection count with an ONNX `If` node: the empty branch emits a `(0, 17)`
tensor, the other performs the blend. Consequently the model's output shape
`(S, 17)` is dynamic and `S = 0` is a well-formed result — consumers should
handle it, and no padding scheme is imposed by the model.

## Design decisions

**Batch size 1.** The exported models accept exactly one image. BlazeFace is
a real-time, single-frame detector, and batching weighted NMS would require
ragged per-image detection counts inside the graph — significant complexity
for no practical benefit in the model's intended use.

**Winner score, not averaged score.** The original MediaPipe implementation
assigns each blended detection the score of its NMS winner; the PyTorch port
averages the cluster's scores instead. This project follows MediaPipe, since
its behavior is the specification being reproduced.

**Pre-exported models are committed.** The four ONNX files in `output/` are
small (~0.5–0.8 MB each) and checked into the repository, so consumers can
use them without installing anything beyond an ONNX runtime. The export
pipeline strips the debug metadata that `torch.onnx.export` would otherwise
embed (source-file stack traces from the exporting machine).

## Export constraints

Two onnx/onnxscript behaviors shape the implementation of `export.py` and are
the reason the dependency versions are pinned in `uv.lock`:

- onnxscript's `to_model_proto()` includes only directly-called local
  functions, so `compute_iou_matrix` (called transitively via `weighted_nms`)
  is re-attached manually, and the `this` domain (onnxscript's domain for
  local functions) is added to each function's opset imports so the ONNX
  checker accepts function calls inside `If` subgraphs.
- `onnx.compose.merge_models` runs the strict model checker, which rejects
  the `this` domain in nested subgraphs. The base and postprocessing models
  are therefore merged by a small purpose-built routine (`_merge_models`)
  that connects `raw_boxes`/`raw_scores`, prefixes the postprocessing
  graph's intermediate names to avoid collisions, and unions the opset
  imports and function definitions.
