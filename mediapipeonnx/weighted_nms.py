"""Weighted NMS ONNX function for BlazeFace postprocessing.

Implements the weighted non-maximum suppression algorithm as an ONNX function
using onnxscript. The algorithm performs standard NMS for winner selection via
the ONNX NonMaxSuppression operator, then blends each winner's regression
coordinates with its overlapping detections using score-weighted averaging.

Box format: [ymin, xmin, ymax, xmax] (ONNX NonMaxSuppression center_point_box=0).
"""

import onnx
import onnxscript
from onnx import TensorProto, helper
from onnxscript import FLOAT, INT64, opset20 as op

NUM_COORDS = 16


@onnxscript.script()
def compute_iou_matrix(boxes_a: FLOAT, boxes_b: FLOAT) -> FLOAT:
    """Compute pairwise IoU matrix between two sets of bounding boxes.

    Args:
        boxes_a: (M, 4) in [ymin, xmin, ymax, xmax] format.
        boxes_b: (N, 4) in [ymin, xmin, ymax, xmax] format.

    Returns:
        (M, N) IoU matrix.
    """
    # Reshape for broadcasting: (M, 1, 4) vs (1, N, 4)
    a = op.Unsqueeze(boxes_a, op.Constant(value_ints=[1]))
    b = op.Unsqueeze(boxes_b, op.Constant(value_ints=[0]))

    # Split into individual coordinates — each becomes (..., 1) on last dim
    a_ymin, a_xmin, a_ymax, a_xmax = op.Split(a, axis=2, num_outputs=4)
    b_ymin, b_xmin, b_ymax, b_xmax = op.Split(b, axis=2, num_outputs=4)

    # Intersection rectangle
    inter_ymin = op.Max(a_ymin, b_ymin)
    inter_xmin = op.Max(a_xmin, b_xmin)
    inter_ymax = op.Min(a_ymax, b_ymax)
    inter_xmax = op.Min(a_xmax, b_xmax)

    zero = op.Constant(value_float=0.0)
    inter_h = op.Clip(inter_ymax - inter_ymin, zero)
    inter_w = op.Clip(inter_xmax - inter_xmin, zero)
    inter_area = inter_h * inter_w

    # Individual areas
    area_a = (a_ymax - a_ymin) * (a_xmax - a_xmin)
    area_b = (b_ymax - b_ymin) * (b_xmax - b_xmin)

    # IoU = intersection / union, squeeze trailing singleton dim
    # Clip union to avoid division by zero for degenerate boxes.
    union = op.Clip(area_a + area_b - inter_area, op.Constant(value_float=1e-6))
    iou = op.Squeeze(inter_area / union, op.Constant(value_ints=[2]))

    return iou


@onnxscript.script()
def weighted_nms(
    regressions: FLOAT,
    scores: FLOAT,
    max_output_boxes: INT64,
    iou_threshold: FLOAT,
    score_threshold: FLOAT,
) -> FLOAT:
    """Weighted NMS postprocessing for BlazeFace.

    Performs standard NMS for winner selection, then blends each winner's
    regression coordinates with overlapping detections via score-weighted
    averaging. The final score for each detection is the NMS-selected
    winner's own score.

    Args:
        regressions: (N, 16) decoded regression coordinates.
            Format: [ymin, xmin, ymax, xmax, kp1_x, kp1_y, ..., kp6_x, kp6_y]
        scores: (N,) detection confidence scores (post-sigmoid).
        max_output_boxes: scalar int64, maximum number of detections.
        iou_threshold: scalar float, IoU threshold for suppression and blend
            grouping.
        score_threshold: scalar float, minimum confidence threshold.

    Returns:
        (S, 17) concatenation of blended regressions (16) and max score (1),
        where S is the number of selected detections (dynamic, may be 0).
    """
    # ── Phase 1: Winner Selection ──────────────────────────────────────

    # Extract bounding boxes from regressions and reshape for NMS input
    all_boxes = op.Slice(
        regressions,
        op.Constant(value_ints=[0]),
        op.Constant(value_ints=[4]),
        op.Constant(value_ints=[1]),
    )  # (N, 4)

    boxes_nms = op.Unsqueeze(all_boxes, op.Constant(value_ints=[0]))  # (1, N, 4)
    scores_nms = op.Unsqueeze(scores, op.Constant(value_ints=[0, 1]))  # (1, 1, N)

    selected_indices = op.NonMaxSuppression(
        boxes_nms,
        scores_nms,
        max_output_boxes,
        iou_threshold,
        score_threshold,
        center_point_box=0,
    )  # (S, 3): [batch_index, class_index, box_index]

    # ── Branch on whether any boxes were selected ──────────────────────

    num_selected = op.Shape(selected_indices, start=0, end=1)
    is_empty = op.Squeeze(op.Equal(num_selected, op.Constant(value_ints=[0])))

    if is_empty:
        output = op.ConstantOfShape(op.Constant(value_ints=[0, 17]))
    else:
        # ── Phase 2: Weighted Blending ─────────────────────────────────

        # 2a. Extract winner box indices from NMS output
        box_indices = op.Gather(
            selected_indices, op.Constant(value_int=2), axis=1
        )  # (S,)

        # 2b. Gather winner boxes for IoU computation
        winner_boxes = op.Gather(all_boxes, box_indices, axis=0)  # (S, 4)

        # 2c. IoU matrix: winners vs all anchor boxes
        iou_matrix = compute_iou_matrix(winner_boxes, all_boxes)  # (S, N)

        # 2d. Build blend mask: overlap AND score above threshold
        iou_mask = op.Greater(iou_matrix, iou_threshold)  # (S, N)
        score_mask = op.Greater(scores, score_threshold)  # (N,)
        blend_mask = op.And(iou_mask, score_mask)  # (S, N) via broadcast

        # 2e. Zero-masked scores as blend weights
        blend_weights = scores * op.Cast(blend_mask, to=onnx.TensorProto.FLOAT)
        # (S, N): non-overlapping / low-score entries are zeroed out

        # 2f. Weighted average of regression coordinates
        #   weights:     (S, N)  -> (S, N, 1)
        #   regressions: (N, 16) -> (1, N, 16)
        #   product:                (S, N, 16)
        weights_3d = op.Unsqueeze(blend_weights, op.Constant(value_ints=[2]))
        reg_3d = op.Unsqueeze(regressions, op.Constant(value_ints=[0]))

        weighted_sum = op.ReduceSum(
            weights_3d * reg_3d, op.Constant(value_ints=[1]), keepdims=0
        )  # (S, 16)
        weight_total = op.ReduceSum(
            blend_weights, op.Constant(value_ints=[1]), keepdims=1
        )  # (S, 1)
        blended_regressions = weighted_sum / weight_total  # (S, 16)

        # 2g. Winner scores (the NMS-selected boxes' own scores)
        winner_scores = op.Unsqueeze(
            op.Gather(scores, box_indices, axis=0),  # (S,)
            op.Constant(value_ints=[1]),
        )  # (S, 1)

        # Concatenate regressions and scores into (S, 17) output
        output = op.Concat(blended_regressions, winner_scores, axis=1)

    return output


def make_weighted_nms_model() -> onnx.ModelProto:
    """Create a standalone ONNX model for the weighted_nms function.

    The onnxscript @script decorator produces a function proto without input
    shape metadata (only dtype is captured by FLOAT/INT64 annotations). This
    helper adds the concrete shape information needed for shape inference
    and standalone execution with ONNX runtimes.

    Returns:
        An onnx.ModelProto ready for inference.
    """
    model = weighted_nms.to_model_proto()

    # Map input names to their (elem_type, shape) specifications.
    # "N" is a symbolic dim (dynamic number of anchors).
    input_shapes = {
        "regressions": (TensorProto.FLOAT, ["N", NUM_COORDS]),
        "scores": (TensorProto.FLOAT, ["N"]),
        "max_output_boxes": (TensorProto.INT64, []),  # scalar
        "iou_threshold": (TensorProto.FLOAT, []),  # scalar
        "score_threshold": (TensorProto.FLOAT, []),  # scalar
    }

    new_inputs = []
    for inp in model.graph.input:
        elem_type, dims = input_shapes[inp.name]
        shape_dims = []
        for d in dims:
            dim = onnx.TensorShapeProto.Dimension()
            if isinstance(d, str):
                dim.dim_param = d
            else:
                dim.dim_value = d
            shape_dims.append(dim)
        new_inp = helper.make_tensor_value_info(inp.name, elem_type, None)
        new_inp.type.tensor_type.shape.ClearField("dim")
        new_inp.type.tensor_type.shape.dim.extend(shape_dims)
        new_inputs.append(new_inp)

    del model.graph.input[:]
    model.graph.input.extend(new_inputs)

    return model
