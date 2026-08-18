"""BlazeFace ONNX export pipeline.

Exports two ONNX models per variant (front/back):
- Base model: BlazeFace neural network only (image -> raw predictions)
- End-to-end model: Base + postprocessing (image -> final detections)

Usage:
    python -m blazefaceonnx.export --variant front --output-dir output/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxscript
import torch
from onnx import TensorProto, helper
from onnxscript import FLOAT, INT64, opset20 as op

from blazefaceonnx.anchor_decode import decode_boxes
from blazefaceonnx.weighted_nms import compute_iou_matrix, weighted_nms

SUBMODULE_DIR = Path(__file__).resolve().parent.parent / "external" / "MediaPipePytorch"

VARIANT_CONFIG = {
    "front": {
        "input_size": 128,
        "scale": 128.0,
        "weights": "blazeface.pth",
        "anchors": "anchors_face.npy",
        "back_model": False,
    },
    "back": {
        "input_size": 256,
        "scale": 256.0,
        "weights": "blazefaceback.pth",
        "anchors": "anchors_face_back.npy",
        "back_model": True,
    },
}


# ── Postprocessing wrapper ─────────────────────────────────────────────


@onnxscript.script()
def postprocess(
    raw_boxes: FLOAT,
    raw_scores: FLOAT,
    anchors: FLOAT,
    scale: FLOAT,
    max_output_boxes: INT64,
    iou_threshold: FLOAT,
    score_threshold: FLOAT,
) -> FLOAT:
    """Postprocessing chain: decode + sigmoid + weighted NMS.

    Args:
        raw_boxes:          (1, 896, 16) raw box regressions.
        raw_scores:         (1, 896, 1) raw classification logits.
        anchors:            (896, 4) anchor boxes in [cx, cy, w, h] format.
        scale:              scalar decode scale (128.0 or 256.0).
        max_output_boxes:   scalar int64, max detections.
        iou_threshold:      scalar float, IoU threshold for NMS.
        score_threshold:    scalar float, minimum confidence.

    Returns:
        (S, 17) final detections [ymin, xmin, ymax, xmax, kps..., score].
    """
    # Squeeze batch dimension
    boxes = op.Squeeze(raw_boxes, op.Constant(value_ints=[0]))  # (896, 16)
    scores_2d = op.Squeeze(raw_scores, op.Constant(value_ints=[0]))  # (896, 1)

    # Decode boxes
    decoded = decode_boxes(boxes, anchors, scale)  # (896, 16)

    # Score processing: clamp + sigmoid + squeeze
    scores_clipped = op.Clip(scores_2d, op.Constant(value_float=-100.0), op.Constant(value_float=100.0))
    scores = op.Squeeze(op.Sigmoid(scores_clipped), op.Constant(value_ints=[1]))  # (896,)

    # Weighted NMS
    detections = weighted_nms(
        decoded, scores, max_output_boxes, iou_threshold, score_threshold
    )

    return detections  # (S, 17)


# ── Base model export ──────────────────────────────────────────────────


def _strip_torch_metadata(model: onnx.ModelProto) -> None:
    """Remove torch.onnx debug metadata (in place).

    torch.onnx.export attaches per-node metadata_props (stack traces, FX node
    names, module hierarchies) that embed absolute paths from the exporting
    machine. None of it is needed for inference.
    """
    del model.metadata_props[:]

    def _strip_graph(graph: onnx.GraphProto) -> None:
        for node in graph.node:
            del node.metadata_props[:]
            for attr in node.attribute:
                if attr.g.ByteSize() > 0:
                    _strip_graph(attr.g)
                for g in attr.graphs:
                    _strip_graph(g)

    _strip_graph(model.graph)


def export_base_model(variant: str, output_path: Path) -> onnx.ModelProto:
    """Export the BlazeFace base neural network to ONNX.

    Args:
        variant: "front" or "back".
        output_path: Where to save the .onnx file.

    Returns:
        The exported ONNX ModelProto.
    """
    config = VARIANT_CONFIG[variant]

    # Temporarily add submodule to sys.path for blazebase import
    submod_str = str(SUBMODULE_DIR)
    sys.path.insert(0, submod_str)
    try:
        from blazeface import BlazeFace

        model = BlazeFace(back_model=config["back_model"])
        model.load_weights(str(SUBMODULE_DIR / config["weights"]))
        model.eval()
    finally:
        sys.path.remove(submod_str)

    input_size = config["input_size"]
    dummy = torch.randn(1, 3, input_size, input_size)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=20,
        input_names=["input"],
        output_names=["raw_boxes", "raw_scores"],
    )

    # torch.onnx.export may produce a separate .data file for weights.
    # Reload with external data resolved, then re-save as a single file.
    proto = onnx.load(str(output_path), load_external_data=True)
    data_path = Path(str(output_path) + ".data")
    if data_path.exists():
        data_path.unlink()
    _strip_torch_metadata(proto)
    onnx.save(proto, str(output_path), save_as_external_data=False)

    return proto


# ── End-to-end model assembly ──────────────────────────────────────────


def _build_postprocess_model(variant: str) -> onnx.ModelProto:
    """Build the postprocessing ONNX model with anchors/scale as initializers.

    Args:
        variant: "front" or "back".

    Returns:
        ONNX ModelProto with anchors and scale embedded as constants.
    """
    config = VARIANT_CONFIG[variant]

    model = postprocess.to_model_proto()

    # onnxscript's to_model_proto() only includes directly called functions,
    # not transitively called ones. compute_iou_matrix is called by
    # weighted_nms, so we need to add it manually from weighted_nms's model.
    existing_names = {f.name for f in model.functions}
    if "compute_iou_matrix" not in existing_names:
        wnms_model = weighted_nms.to_model_proto()
        for f in wnms_model.functions:
            if f.name == "compute_iou_matrix":
                model.functions.append(f)
                break

    # Ensure all functions that call other local functions have the 'this'
    # domain in their opset_import. The ONNX checker requires this for
    # function calls in nested subgraphs (e.g., If/Loop bodies).
    this_opset = onnx.helper.make_opsetid("this", 1)
    for func in model.functions:
        domains = {o.domain for o in func.opset_import}
        if "this" not in domains:
            func.opset_import.append(this_opset)

    # Add shape metadata to inputs
    input_shapes = {
        "raw_boxes": (TensorProto.FLOAT, [1, "N", 16]),
        "raw_scores": (TensorProto.FLOAT, [1, "N", 1]),
        "anchors": (TensorProto.FLOAT, ["N", 4]),
        "scale": (TensorProto.FLOAT, []),
        "max_output_boxes": (TensorProto.INT64, []),
        "iou_threshold": (TensorProto.FLOAT, []),
        "score_threshold": (TensorProto.FLOAT, []),
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

    # Embed anchors and scale as initializers
    anchors_data = np.load(
        str(SUBMODULE_DIR / config["anchors"])
    ).astype(np.float32)
    scale_data = np.array(config["scale"], dtype=np.float32)

    anchors_init = onnx.numpy_helper.from_array(anchors_data, name="anchors")
    scale_init = onnx.numpy_helper.from_array(scale_data, name="scale")

    model.graph.initializer.append(anchors_init)
    model.graph.initializer.append(scale_init)

    # Remove anchors and scale from graph inputs (they are now constants)
    remaining_inputs = [
        inp for inp in model.graph.input if inp.name not in ("anchors", "scale")
    ]
    del model.graph.input[:]
    model.graph.input.extend(remaining_inputs)

    return model


def _merge_models(m1: onnx.ModelProto, m2: onnx.ModelProto) -> onnx.ModelProto:
    """Merge base model (m1) and postprocessing model (m2) into one graph.

    Connects m1's outputs (raw_boxes, raw_scores) to m2's inputs of the
    same names. m2's remaining inputs become the merged model's inputs.
    All function definitions from m2 are carried over.

    This manual merge avoids onnx.compose.merge_models which runs
    check_model internally and fails on the 'this' domain used by
    onnxscript local functions in nested subgraphs.
    """
    # Prefix m2 node names to avoid collisions
    prefix = "postprocess_"

    # Build name mapping for m2's internal names
    m2_rename = {}
    # The connected outputs: m1 output names are used directly.
    # m2's graph outputs keep their names so the merged model exposes them
    # unprefixed (e.g. "detections").
    connected = {"raw_boxes", "raw_scores"}
    preserved = connected | {out.name for out in m2.graph.output}

    # Rename m2 intermediate values (not names shared with m1 or graph outputs)
    for node in m2.graph.node:
        for i, name in enumerate(node.output):
            if name and name not in preserved:
                new_name = prefix + name
                m2_rename[name] = new_name

    def _rename(name: str) -> str:
        return m2_rename.get(name, name)

    def _rename_graph(graph):
        """Recursively rename values in a graph."""
        for node in graph.node:
            for i, name in enumerate(node.input):
                node.input[i] = _rename(name)
            for i, name in enumerate(node.output):
                node.output[i] = _rename(name)
            # Recurse into subgraphs (If/Loop/Scan nodes)
            for attr in node.attribute:
                if attr.g and attr.g.ByteSize() > 0:
                    _rename_graph(attr.g)
                for g in attr.graphs:
                    _rename_graph(g)

    _rename_graph(m2.graph)

    # Build merged graph
    # Nodes: m1 nodes + m2 nodes
    all_nodes = list(m1.graph.node) + list(m2.graph.node)

    # Inputs: m1's inputs + m2's non-connected inputs
    all_inputs = list(m1.graph.input)
    for inp in m2.graph.input:
        if inp.name not in connected:
            all_inputs.append(inp)

    # Outputs: m2's outputs only (the final detections)
    all_outputs = list(m2.graph.output)

    # Initializers: m1's + m2's
    all_initializers = list(m1.graph.initializer) + list(m2.graph.initializer)

    # Value info: m1's + m2's
    all_value_info = list(m1.graph.value_info)
    # Add m1 output type info as intermediate value info
    for out in m1.graph.output:
        all_value_info.append(out)

    merged_graph = helper.make_graph(
        all_nodes,
        "blazeface_e2e",
        all_inputs,
        all_outputs,
        initializer=all_initializers,
        value_info=all_value_info,
    )

    # Collect opset imports (union of both models)
    opset_map = {}
    for opset in m1.opset_import:
        key = opset.domain
        opset_map[key] = max(opset_map.get(key, 0), opset.version)
    for opset in m2.opset_import:
        key = opset.domain
        opset_map[key] = max(opset_map.get(key, 0), opset.version)

    opset_imports = []
    for domain, version in opset_map.items():
        opset_imports.append(helper.make_opsetid(domain, version))

    e2e = helper.make_model(merged_graph, opset_imports=opset_imports)
    e2e.ir_version = m1.ir_version

    # Carry over function definitions from m2
    e2e.functions.extend(m2.functions)

    return e2e


def export_e2e_model(
    variant: str, output_path: Path, base_model: onnx.ModelProto | None = None
) -> onnx.ModelProto:
    """Export the end-to-end BlazeFace model (base + postprocessing).

    Args:
        variant: "front" or "back".
        output_path: Where to save the .onnx file.
        base_model: Already-exported base ModelProto to reuse. If None, the
            base model is exported alongside the e2e model.

    Returns:
        The merged ONNX ModelProto.
    """
    if base_model is None:
        base_path = output_path.parent / f"blazeface_{variant}_base.onnx"
        base_model = export_base_model(variant, base_path)
    m1 = base_model

    # Build postprocessing model
    m2 = _build_postprocess_model(variant)

    # Merge manually to avoid onnx.compose.merge_models' strict check_model
    # which fails on the 'this' domain used by onnxscript local functions
    # inside nested subgraphs.
    e2e = _merge_models(m1, m2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(e2e, str(output_path))

    return e2e


# ── Main entry point ───────────────────────────────────────────────────


def export(variant: str = "front", output_dir: str = "output") -> tuple[Path, Path]:
    """Export both base and e2e models for the given variant.

    Args:
        variant: "front" or "back".
        output_dir: Output directory.

    Returns:
        Tuple of (base_path, e2e_path).
    """
    out = Path(output_dir)
    base_path = out / f"blazeface_{variant}_base.onnx"
    e2e_path = out / f"blazeface_{variant}_e2e.onnx"

    base_model = export_base_model(variant, base_path)
    export_e2e_model(variant, e2e_path, base_model=base_model)

    return base_path, e2e_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export BlazeFace ONNX models")
    parser.add_argument(
        "--variant",
        choices=["front", "back"],
        default="front",
        help="Model variant to export (default: front)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output/)",
    )
    args = parser.parse_args()

    base_path, e2e_path = export(args.variant, args.output_dir)
    print(f"Base model: {base_path}")
    print(f"E2E model:  {e2e_path}")
