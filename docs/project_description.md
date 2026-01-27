# BlazeFace End-to-End ONNX Model Export

## Task Introduction

Our task is to export an ONNX model for the MediaPipe Face Detection Model (BlazeFace), also including its post-processing parts (i.e. anchor decoding, weighted nms) in order to create a single encapsulated end-to-end ONNX model for clean and portable BlazeFace inference.

## `zmurez/MediaPipePyTorch`

The task is primarily based on the `zmurez/MediaPipePyTorch` GitHub repository, which is added as a git submodule in the `external/MediaPipePyTorch` directory of the current repository (`MediaPipeONNX`).

The `MediaPipePyTorch` repository provides clean ports of the MediaPipe tflite models to PyTorch, including PyTorch model definitions, model weights (as `.pt` state dict), post-processing utilities.

More specifically, (from its root directory) `blazebase.py` defines base classes for PyTorch detector models (`BlazeBase`; specifically `BlazeDetector` in our case), which contains several methods for weight/anchor loading, pre/post-processing, wrapped model inference etc.

`blazeface.py` defines a subclass specific to the BlazeFace model (for both "front/short-range" model and "back/full-range" model; henceforward just "front" and "back"). We can actually export a complete ONNX model of BlazeFace by exporting this `BlazeFace` PyTorch model via `torch.onnx.export`, which wouldn't include its pre/post-processing functions.

The anchors for the front/back BlazeFace models are provided as `anchors_face.npy` and `anchors_face_back.npy`, respectively (whose usages can be inferred from the postprocessing methods of the `BlazeDetector` baseclass).

Similarly, the weights are provided as `blazeface.pth` and `blazefaceback.pth`.

## ONNX Export Task

Given these `MediaPipePyTorch` implementations, our task would focus on exporting an end-to-end ONNX model that encapsulates its [BlazeFace -> postprocessing] inference pipeline (input remains the same as the raw `BlazeFace model`).

We would first export an ONNX model of the `BlazeFace` PyTorch model in isolation using `torch.onnx.export`. This model (i.e. without postprocessing) should support being used independently. Let's call this ONNX model "BlazeFace base ONNX model" (e.g. the exported ONNX model name might be `blazeface_{front/back}_base.onnx`).

We would then have to define custom ONNX functions, possibly via onnxscript (or direct onnx graph manipulation), for its post-processing logic (i.e. anchor decoding, weighted nms etc.). These functions would be appended to the BlazeFace ONNX model, with their inputs being the output of the BlazeONNX model along with some additional postprocessing parameters depending on their implementations.

Let's call this resulting model "BlazeFace end-to-end ONNX model" (e.g. the exported ONNX model name might be `blazeface_{front/back}_e2e.onnx`).

So ultimately, our "export pipeline" should yield both BlazeFace ONNX model and its end-to-end version, and support both its front and back variants (e.g. configurable by some flag).

The current `MediaPipeONNX` repository is for implementing this "export pipeline" for this task. We can freely implement various scripts, utilities, function definitions, tests in any desired structure for this task within this repository.

We're currently using (or would use) Python 3.12, with the following core dependencies currently being used (or would be used):
```
# These are just the current latest versions installed from pip 
torch (2.10.0)
onnx (1.20.1)
onnxscript (0.5.7)
onnxruntime (1.23.2)
opencv-python (4.13.0)
```

We can freely add additional dependencies if needed for the task.

We aim ONNX opset >= 20.

## Requirements

### Design Approaches

The implementation of the export pipeline should be **correct and clean**: e.g. it should be correctly using external library APIs (e.g. torch, onnx, onnxscript), and correctly defining custom ONNX functions (i.e. ensuring both logical and code correctness), while having clean designs across its export pipeline.

The design should prefer minimalism that does not make any sacrifice in the logic (analogous to Occam's Razor).

### Custom ONNX Functions

Custom ONNX functions should be carefully designed and implemented, by carefully chooosing appropriate set of ONNX operators for each part of the postprocessing algorithm. The ONNX function implementation thus can be different from the Python method implementations (e.g. by having ONNX-friendly vectorized flow/op designs) as long as it maintains "high-level equivalence" to the original postprocessing algorithm.

While we use the PyTorch port implementation for the `BlazeFace` model export, its post-processing algorithm implementations are just reference implementations; we should freely design custom (ONNX-native) implementation of the similar high-level logic for the post-processing function implementations. So for custom ONNX graphs/functions, try to design from first principles considering our task.

### Batch Size 1

While the `zmurez/MediaPipePyTorch` implementation of `BlazeFace` supports arbitrary batch size for its input, our export pipeline should only support single sample inference (batch size = 1), for the following two reasons:
* the postprocessing function implementation (especially the weighted nms) would be unnecessarily complex
* the BlazeFace model is intended for live-stream inferencing, thus the batch size 1 is natural (the original MediaPipe model supports only a single image/frame at a time for this reason)

### Scores in Weighted NMS

For the weighted nms algorithm, the original MediaPipe code assigns the score of the most confident detection to the weighted detection, while the PyTorch port take the average score of the overlapping detections.

We should use the logic of the original MediaPipe code, which takes the **maximum score of the overlapping detections.**