"""Anchor box decoding ONNX function for BlazeFace postprocessing.

Converts raw neural network regression outputs to absolute coordinates
using pre-computed anchor boxes. Implements the decoding logic from
MediaPipe's TfLiteTensorsToDetectionsCalculator as an ONNX function
using onnxscript.

Anchor format: [cx, cy, w, h] (center-x, center-y, width, height).
Output format: [ymin, xmin, ymax, xmax, kp1_x, kp1_y, ..., kp6_x, kp6_y].
"""

import onnx
import onnxscript
from onnx import TensorProto, helper
from onnxscript import FLOAT, opset20 as op

NUM_COORDS = 16


@onnxscript.script()
def decode_boxes(raw_boxes: FLOAT, anchors: FLOAT, scale: FLOAT) -> FLOAT:
    """Decode raw regression outputs to absolute coordinates using anchors.

    Args:
        raw_boxes: (N, 16) raw regressions from the neural network.
        anchors:   (N, 4) anchor boxes in [cx, cy, w, h] format.
        scale:     scalar (128.0 for front, 256.0 for back).

    Returns:
        (N, 16) decoded boxes in [ymin, xmin, ymax, xmax, kp1_x, kp1_y, ...] format.
    """
    # Split anchor columns: each (N, 1)
    a_cx, a_cy, a_w, a_h = op.Split(anchors, axis=1, num_outputs=4)

    # Split raw box center/size predictions: each (N, 1)
    raw_cx = op.Slice(
        raw_boxes,
        op.Constant(value_ints=[0]),
        op.Constant(value_ints=[1]),
        op.Constant(value_ints=[1]),
    )
    raw_cy = op.Slice(
        raw_boxes,
        op.Constant(value_ints=[1]),
        op.Constant(value_ints=[2]),
        op.Constant(value_ints=[1]),
    )
    raw_w = op.Slice(
        raw_boxes,
        op.Constant(value_ints=[2]),
        op.Constant(value_ints=[3]),
        op.Constant(value_ints=[1]),
    )
    raw_h = op.Slice(
        raw_boxes,
        op.Constant(value_ints=[3]),
        op.Constant(value_ints=[4]),
        op.Constant(value_ints=[1]),
    )

    # Decode box center and size
    cx = raw_cx / scale * a_w + a_cx
    cy = raw_cy / scale * a_h + a_cy
    w = raw_w / scale * a_w
    h = raw_h / scale * a_h

    # Convert center+size to corner format: each (N, 1)
    half = op.Constant(value_float=2.0)
    ymin = cy - h / half
    xmin = cx - w / half
    ymax = cy + h / half
    xmax = cx + w / half

    # Decode keypoints via reshape trick
    # raw keypoints: (N, 12) -> (N, 6, 2) where last dim = [x, y]
    kp_raw = op.Slice(
        raw_boxes,
        op.Constant(value_ints=[4]),
        op.Constant(value_ints=[16]),
        op.Constant(value_ints=[1]),
    )  # (N, 12)

    n_shape = op.Shape(raw_boxes, start=0, end=1)  # [N]
    kp_shape = op.Concat(n_shape, op.Constant(value_ints=[6, 2]), axis=0)
    kp_3d = op.Reshape(kp_raw, kp_shape)  # (N, 6, 2)

    # Anchor centers and sizes for keypoint broadcast: (N, 1, 2)
    # keypoint x uses anchor cx and w; keypoint y uses anchor cy and h
    anchor_centers_3d = op.Reshape(
        op.Concat(a_cx, a_cy, axis=1),  # (N, 2)
        op.Concat(n_shape, op.Constant(value_ints=[1, 2]), axis=0),
    )  # (N, 1, 2)

    anchor_sizes_3d = op.Reshape(
        op.Concat(a_w, a_h, axis=1),  # (N, 2)
        op.Concat(n_shape, op.Constant(value_ints=[1, 2]), axis=0),
    )  # (N, 1, 2)

    kp_decoded = kp_3d / scale * anchor_sizes_3d + anchor_centers_3d  # (N, 6, 2)

    # Reshape back to (N, 12)
    kp_flat_shape = op.Concat(n_shape, op.Constant(value_ints=[12]), axis=0)
    kp_flat = op.Reshape(kp_decoded, kp_flat_shape)  # (N, 12)

    # Concat [ymin, xmin, ymax, xmax, kp_flat] -> (N, 16)
    decoded = op.Concat(ymin, xmin, ymax, xmax, kp_flat, axis=1)

    return decoded


def make_decode_boxes_model() -> onnx.ModelProto:
    """Create a standalone ONNX model for the decode_boxes function.

    Returns:
        An onnx.ModelProto ready for inference.
    """
    model = decode_boxes.to_model_proto()

    input_shapes = {
        "raw_boxes": (TensorProto.FLOAT, ["N", NUM_COORDS]),
        "anchors": (TensorProto.FLOAT, ["N", 4]),
        "scale": (TensorProto.FLOAT, []),  # scalar
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
