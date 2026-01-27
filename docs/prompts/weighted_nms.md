The project task description is provided in `docs/project_description.md` document. Read this document carefully to understand the main task of this project. You can explore the `external/MediaPipePytorch` submodule to better understand the project.

Assuming you have understood the project, let me explain the sub-task that we'll cover in the current session.

We should design an ONNX function implementation of the weighted NMS postprocessing algorithm as you can see from the project document. I believe this is the most complex part of this project, so we'll design and implement this function first in isolation (bottom-up approach).

For this, I'd first like to make a clean and optimized design for the function. So let's analyze the problem first.

## Weighted NMS ONNX Function Design

### Use of `NonMaxSuppression`

Weighted NMS has the same box selection logic as typical NMS, so the only difference would be its "box (regression coordinates) blending" logic, and the rest would be same. Given this observation, I think we might exploit the existing ONNX `NonMaxSuppression` operator.

More specifically, if we do typical NMS with `NonMaxSuppression` first (covers the hard selection part) and apply the weighted averaging logic (after re-computing overlapping boxes for the selected boxes), then the result should be "equivalent" (to be explained below) to the weighted NMS algorithm.

With this design, the weighted average computation could be vectorized since the ONNX `NonMaxSuppression` op handles most of the complex iterative processes.

While this introduces some redundancy (i.e. IoU computed twice), the native `NonMaxSuppression` operation and vectorized computations would provide advantage over the complete manual implementation (not to mention the implementation complexity).

So let's try to use the `NonMaxSuppression` operator.

Equivalence note: In the original iterative algorithm, once a box is consumed by blending with a winner, it is removed from the candidate pool for subsequent winners. In our approach, a box could theoretically contribute to multiple winners' blends. However, two winners cannot have IoU > threshold with each other (otherwise one would suppress the other), so the set of boxes overlapping winner A and the set overlapping winner B are largely disjoint. For BlazeFace's practical output (0–2 detections), this is a non-issue.

Actually, NMS is a specific greedy algorithm for selecting a set of non-overlapping unique boxes given multiple boxes with score and IoU requirements. Therefore, removal of candidates at each step does not uniquely define the "cluster membership" of the removed candidates. So it's better to view this version as a decoupled winner selection (NMS) and smooth blending (weighted average), where the cluster membership is actually/separately defined in the latter phase based on the IoU similarity metric given the selected winners.

Regardless of the correctness, the result would be practically indistinguishable to the original weighted NMS outputs in our case, and would be exactly equivalent when having 0-1 detections, so we are going to use this algorithm anyway.

### Algorithm Design

The `BlazeFace` model outputs the two tensors (batch size 1 in our case):
* regressions: (B, num_anchors, num_coords) = (1, 896, 16)
* scores: (B, num_anchors, num_classes) = (1, 896, 1)

with the first 4 elements of the regressions being the box data, compatible to the ONNX `NonMaxSuppression` input.

We can provide the `boxes` and `scores` (adjusted from the raw tensors) input to `NonMaxSuppression`, along with the three additional parameters required for NMS:
* `max_output_boxes_per_class`
* `iou_threshold`
* `score_threshold`

I think we can make them additional inputs of the end-to-end ONNX model (possibly after renaming `max_output_boxes_per_class` to `max_output_boxes` since we have only one class).

Then we will get some indices from the `NonMaxSuppression` as an output. Since both batch size and class num are 1, we would only have to consider the `box_index` among the selected index format `[batch_index, class_index, box_index]`.

We can compute the number of selected boxes (e.g. `num_selected_boxes`) based on the shape of the returned indices. Note that, at this point (after NMS), we would mostly have only a few or none of the boxes in practice (especially for BlazeFace).

Anyway, for each of the selected boxes, we should find its overlapping boxes (including itself) by computing IoUs again, then take the weighted average of the regression coordinates, and take its own score, yielding the final regression values (16) and the score (1).

I think there would exist many different (but equivalent) algorithms to compute theses. Let's think of one possible (but not optimal) algorithm:

---

For non-empty selected boxes (i.e. `num_selected_boxes > 0`), we could vectorize the computation of the *IoU matrix* of size `num_selected_boxes` x `num_anchors`. We could use broadcasted/vectorized logic similar to what's shown in `experiments/iou.py` (numpy function implementation).

Then we could create an *overlap mask* by thresholding the IoU matrix with `iou_threshold`. Similarly, we could create a global *score mask* (as a vector) by thresholding the `scores` with `score_threshold`. Broadcasted logical AND (or multiplication if using 0/1 masks) between the two masks creates the *blend mask* (of the size of the overlap mask) within which weighted average should be computed for each selected box.

Here, we can "zero-mask" the `scores` with the blend mask, then compute the "global" weighted average of regression coordinates based on the zero-masked scores, for each selected box with vectorized computation. Non-overlapping or low-score boxes are cleanly excluded by zero masking here, thus the result is mathematically equivalent to the original weighted NMS.

---

We might want to avoid redundant computations on IoU matrix or weighted average for those are semantically excluded from the computations (e.g. non-overlapping or low-score boxes).

So alternatively, we could first apply the `Gather` operation on both regressions and the scores by the score mask. And then we could apply the same IoU matrix and weighted average computation logic within the reduced regressions and score slices (instead of the whole 896 boxes). Given the fact that we would only have a few (and mostly one if not empty) selected boxes for BlazeFace inference (especially for its "front" variant), this would reasonably reduce the computation near minimum while still avoiding conditional/iterative flows.

---

There might be different possible designs optimized for this part, so we would have to keep investigating the design. By the way, I think the overall postprocessing calculations would be quite small anyway (compared to the main model), so should always keep in mind the complexity-optimization trade-off.

We might want to add one conditional flow to skip the weighted averaging subgraph if empty indices returned from `NonMaxSuppression` (i.e. `num_selected_boxes == 0`) (or different way if it is possible to unify the computation without introducing the conditional flow).

We have to define the output shape of the tensor for both branches. While we might not want dynamic/empty output shape for the final ONNX model, as a modular ONNX function, I think it'd be better to output the dynamic output shape tensor that can have empty dimension (`num_selected_boxes == 0`), and leave the padding or any other processing (required to ensure static output shape; if needed) to the ONNX model (graph) that uses this function.

So the final output of the function would be filtered/blended regressions and scores tensors (or their concatenation), with a dynamic dimension with the size of `num_selected_boxes`.

## Task

After you carefully review the provided contexts and descriptions, I'd like you to design an ONNX function implementation of the weighted nms post processing algorithm, deeply considering the design/implementation contexts/requirements/approaches. Your design doesn't have to be identical to my suggested ones, so think of the various strategies from first principles considering the problem.

After that, provide me a design blueprint for the ONNX function before we can proceed with implementing the function.