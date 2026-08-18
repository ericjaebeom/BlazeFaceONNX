# BlazeFaceONNX

Export [BlazeFace](https://arxiv.org/abs/1907.05047) face detection models to ONNX, with postprocessing (anchor decoding, score sigmoid, weighted NMS) baked into the graph — the end-to-end model maps an image directly to final detections, so consumers need no anchor tables, decode math, or NMS code.

Two model variants are available:

| Variant | Input size | Intended use |
|---------|-----------|--------------|
| `front` | 128x128 | Front-facing camera (close-range faces) |
| `back` | 256x256 | Rear-facing camera (farther/smaller faces) |

Each variant produces two ONNX files:

- **Base model** — the neural network only (image to raw predictions)
- **End-to-end model** — neural network + full postprocessing (image to final detections)

## Pre-exported models

The exported models are committed in `output/`:

```
output/
  blazeface_front_base.onnx
  blazeface_front_e2e.onnx
  blazeface_back_base.onnx
  blazeface_back_e2e.onnx
```

To run inference you only need one of these files and `onnxruntime` — no other setup. See the [inference example](#inference-example-python) below. The Setup and Exporting sections are only needed to re-export the models yourself.

## Setup

Clone with submodules (the PyTorch BlazeFace weights live in `external/MediaPipePytorch`):

```bash
git clone --recurse-submodules https://github.com/ericjaebeom/BlazeFaceONNX.git
```

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Exporting models

```bash
# Front variant
uv run python -m blazefaceonnx.export --variant front --output-dir output/

# Back variant
uv run python -m blazefaceonnx.export --variant back --output-dir output/
```

## Output models

### Base model

Runs the BlazeFace convolutional network. Useful when you want to apply your own postprocessing.

**Inputs:**

| Name | Type | Shape |
|------|------|-------|
| `input` | float32 | `(1, 3, H, W)` — `H=W=128` for front, `256` for back |

Input values must be normalized to `[0, 1]` range (e.g. divide uint8 pixels by 255).

**Outputs:**

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `raw_boxes` | float32 | `(1, 896, 16)` | Raw regression outputs (not yet decoded) |
| `raw_scores` | float32 | `(1, 896, 1)` | Raw classification logits (pre-sigmoid) |

### End-to-end model

Runs the full detection pipeline: neural network, anchor box decoding, score sigmoid, and weighted non-maximum suppression. Returns ready-to-use detections.

**Inputs:**

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `input` | float32 | `(1, 3, H, W)` | Same as base model |
| `max_output_boxes` | int64 | scalar | Maximum number of detections to return |
| `iou_threshold` | float32 | scalar | IoU threshold for NMS suppression/blending (e.g. `0.3`) |
| `score_threshold` | float32 | scalar | Minimum confidence to keep a detection (e.g. `0.5`) |

Anchors and decode scale are embedded in the model as constants.

**Outputs:**

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `detections` | float32 | `(S, 17)` | `S` detections (dynamic, may be 0) |

Each detection row has 17 values:

```
[ymin, xmin, ymax, xmax, kp0_x, kp0_y, kp1_x, kp1_y, kp2_x, kp2_y, kp3_x, kp3_y, kp4_x, kp4_y, kp5_x, kp5_y, score]
```

- Bounding box coordinates and keypoints are in normalized `[0, 1]` space relative to the input image
- The 6 keypoints are: right eye, left eye, nose tip, mouth center, right ear tragion, left ear tragion
- `score` is the post-sigmoid detection confidence

### Inference example (Python)

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("output/blazeface_front_e2e.onnx")

# Prepare a (1, 3, 128, 128) float32 image in [0, 1] range
image = np.random.rand(1, 3, 128, 128).astype(np.float32)

detections = sess.run(None, {
    "input": image,
    "max_output_boxes": np.array(10, dtype=np.int64),
    "iou_threshold": np.array(0.3, dtype=np.float32),
    "score_threshold": np.array(0.75, dtype=np.float32),
})[0]  # (S, 17)

for det in detections:
    ymin, xmin, ymax, xmax = det[:4]
    score = det[16]
    print(f"Face: ({xmin:.3f}, {ymin:.3f}) - ({xmax:.3f}, {ymax:.3f}), score={score:.3f}")
```

## Demos

```bash
# Detect faces in an image (supply any image file)
uv run python demo/image_demo.py output/blazeface_front_e2e.onnx path/to/image.jpg

# Live webcam detection
uv run python demo/livestream_demo.py output/blazeface_front_e2e.onnx --camera 0
```

## Running tests

```bash
uv run pytest tests/ -v
```

## Design

See [docs/design.md](docs/design.md) for the project design, including the ONNX-native weighted NMS formulation.

## License

Apache-2.0 — see [LICENSE](LICENSE).

The model weights originate from Google's [MediaPipe](https://github.com/google-ai-edge/mediapipe) BlazeFace models (Apache-2.0), converted to PyTorch by [zmurez/MediaPipePyTorch](https://github.com/zmurez/MediaPipePyTorch) (building on [hollance/BlazeFace-PyTorch](https://github.com/hollance/BlazeFace-PyTorch)). The ONNX files in `output/` are derived from those weights.
