"""Tests for the BlazeFace ONNX export pipeline.

Verifies base model export, end-to-end model assembly, and numerical
equivalence between the ONNX pipeline and the PyTorch reference
implementation.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from mediapipeonnx.export import (
    SUBMODULE_DIR,
    VARIANT_CONFIG,
    export_base_model,
    export_e2e_model,
)

# Ensure submodule is importable for reference pipeline
_submod_str = str(SUBMODULE_DIR)


def _get_pytorch_model(variant: str):
    """Load the PyTorch BlazeFace model for reference comparison."""
    config = VARIANT_CONFIG[variant]
    sys.path.insert(0, _submod_str)
    try:
        from blazeface import BlazeFace

        model = BlazeFace(back_model=config["back_model"])
        model.load_weights(str(SUBMODULE_DIR / config["weights"]))
        model.eval()
    finally:
        sys.path.remove(_submod_str)
    return model


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def front_base_model():
    """Export front base model to a temp file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "blazeface_front_base.onnx"
        model = export_base_model("front", path)
        yield model, str(path)


@pytest.fixture(scope="module")
def front_e2e_model():
    """Export front e2e model to a temp file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        e2e_path = Path(tmpdir) / "blazeface_front_e2e.onnx"
        model = export_e2e_model("front", e2e_path)
        yield model, str(e2e_path)


@pytest.fixture(scope="module")
def back_base_model():
    """Export back base model to a temp file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "blazeface_back_base.onnx"
        model = export_base_model("back", path)
        yield model, str(path)


@pytest.fixture(scope="module")
def back_e2e_model():
    """Export back e2e model to a temp file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        e2e_path = Path(tmpdir) / "blazeface_back_e2e.onnx"
        model = export_e2e_model("back", e2e_path)
        yield model, str(e2e_path)


# ── Base model tests ──────────────────────────────────────────────────


class TestBaseModelFront:
    """Front variant base model export and validation."""

    def test_model_validates(self, front_base_model):
        model, _ = front_base_model
        onnx.checker.check_model(model)

    def test_output_names(self, front_base_model):
        model, _ = front_base_model
        output_names = [o.name for o in model.graph.output]
        assert "raw_boxes" in output_names
        assert "raw_scores" in output_names

    def test_input_name(self, front_base_model):
        model, _ = front_base_model
        input_names = [i.name for i in model.graph.input]
        assert "input" in input_names

    def test_inference_shapes(self, front_base_model):
        _, path = front_base_model
        sess = ort.InferenceSession(path)
        dummy = np.random.randn(1, 3, 128, 128).astype(np.float32)
        raw_boxes, raw_scores = sess.run(None, {"input": dummy})
        assert raw_boxes.shape == (1, 896, 16)
        assert raw_scores.shape == (1, 896, 1)

    def test_matches_pytorch(self, front_base_model):
        """Base ONNX output should match PyTorch forward pass."""
        _, path = front_base_model
        sess = ort.InferenceSession(path)

        rng = np.random.RandomState(42)
        dummy_np = rng.randn(1, 3, 128, 128).astype(np.float32)
        dummy_pt = torch.from_numpy(dummy_np)

        # ONNX inference
        raw_boxes_onnx, raw_scores_onnx = sess.run(None, {"input": dummy_np})

        # PyTorch inference
        pt_model = _get_pytorch_model("front")
        with torch.no_grad():
            pt_out = pt_model(dummy_pt)
        raw_boxes_pt = pt_out[0].numpy()
        raw_scores_pt = pt_out[1].numpy()

        np.testing.assert_allclose(raw_boxes_onnx, raw_boxes_pt, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(
            raw_scores_onnx, raw_scores_pt, rtol=1e-3, atol=1e-4
        )


class TestBaseModelBack:
    """Back variant base model export and validation."""

    def test_model_validates(self, back_base_model):
        model, _ = back_base_model
        onnx.checker.check_model(model)

    def test_inference_shapes(self, back_base_model):
        _, path = back_base_model
        sess = ort.InferenceSession(path)
        dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)
        raw_boxes, raw_scores = sess.run(None, {"input": dummy})
        assert raw_boxes.shape == (1, 896, 16)
        assert raw_scores.shape == (1, 896, 1)

    def test_matches_pytorch(self, back_base_model):
        _, path = back_base_model
        sess = ort.InferenceSession(path)

        rng = np.random.RandomState(99)
        dummy_np = rng.randn(1, 3, 256, 256).astype(np.float32)
        dummy_pt = torch.from_numpy(dummy_np)

        raw_boxes_onnx, raw_scores_onnx = sess.run(None, {"input": dummy_np})

        pt_model = _get_pytorch_model("back")
        with torch.no_grad():
            pt_out = pt_model(dummy_pt)
        raw_boxes_pt = pt_out[0].numpy()
        raw_scores_pt = pt_out[1].numpy()

        np.testing.assert_allclose(raw_boxes_onnx, raw_boxes_pt, rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(
            raw_scores_onnx, raw_scores_pt, rtol=1e-3, atol=1e-4
        )


# ── End-to-end model tests ───────────────────────────────────────────


class TestE2EModelFront:
    """Front variant end-to-end model assembly and validation."""

    def test_model_validates(self, front_e2e_model):
        model, _ = front_e2e_model
        onnx.checker.check_model(model)

    def test_input_names(self, front_e2e_model):
        model, _ = front_e2e_model
        input_names = {i.name for i in model.graph.input}
        assert "input" in input_names
        assert "max_output_boxes" in input_names
        assert "iou_threshold" in input_names
        assert "score_threshold" in input_names
        # Anchors and scale should NOT be inputs (embedded as initializers)
        assert "anchors" not in input_names
        assert "scale" not in input_names

    def test_output_names(self, front_e2e_model):
        model, _ = front_e2e_model
        output_names = [o.name for o in model.graph.output]
        assert len(output_names) == 1

    def test_has_function_definitions(self, front_e2e_model):
        model, _ = front_e2e_model
        func_names = {f.name for f in model.functions}
        assert "decode_boxes" in func_names
        assert "weighted_nms" in func_names
        assert "compute_iou_matrix" in func_names

    def test_inference_runs(self, front_e2e_model):
        """E2E model should run without errors on random input."""
        _, path = front_e2e_model
        sess = ort.InferenceSession(path)
        dummy = np.random.randn(1, 3, 128, 128).astype(np.float32)
        results = sess.run(
            None,
            {
                "input": dummy,
                "max_output_boxes": np.array(10, dtype=np.int64),
                "iou_threshold": np.array(0.3, dtype=np.float32),
                "score_threshold": np.array(0.75, dtype=np.float32),
            },
        )
        assert len(results) == 1
        detections = results[0]
        assert detections.ndim == 2
        assert detections.shape[1] == 17

    def test_numerical_equivalence(self, front_e2e_model):
        """E2E ONNX output should match PyTorch reference pipeline."""
        _, path = front_e2e_model
        sess = ort.InferenceSession(path)
        config = VARIANT_CONFIG["front"]

        # Load reference components
        pt_model = _get_pytorch_model("front")
        anchors_np = np.load(
            str(SUBMODULE_DIR / config["anchors"])
        ).astype(np.float32)
        anchors_pt = torch.from_numpy(anchors_np)

        # Use a deterministic input
        rng = np.random.RandomState(777)
        dummy_np = rng.randn(1, 3, 128, 128).astype(np.float32)
        dummy_pt = torch.from_numpy(dummy_np)

        # PyTorch reference pipeline
        with torch.no_grad():
            pt_out = pt_model(dummy_pt)
        raw_boxes_pt = pt_out[0]
        raw_scores_pt = pt_out[1]

        # Run PyTorch postprocessing
        pt_model.anchors = anchors_pt
        detections_pt = pt_model._tensors_to_detections(
            raw_boxes_pt, raw_scores_pt, anchors_pt
        )
        # detections_pt is a list (one per batch). Take first batch.
        if len(detections_pt[0]) > 0:
            nms_pt = pt_model._weighted_non_max_suppression(detections_pt[0])
            if len(nms_pt) > 0:
                ref_output = torch.stack(nms_pt).numpy()
            else:
                ref_output = np.zeros((0, 17), dtype=np.float32)
        else:
            ref_output = np.zeros((0, 17), dtype=np.float32)

        # ONNX e2e pipeline — use the same thresholds as the PyTorch reference
        # (min_score_thresh=0.75, min_suppression_threshold=0.3)
        onnx_output = sess.run(
            None,
            {
                "input": dummy_np,
                "max_output_boxes": np.array(100, dtype=np.int64),
                "iou_threshold": np.array(0.3, dtype=np.float32),
                "score_threshold": np.array(0.75, dtype=np.float32),
            },
        )[0]

        # Both should produce the same number of detections
        assert onnx_output.shape[0] == ref_output.shape[0], (
            f"Detection count mismatch: ONNX={onnx_output.shape[0]}, "
            f"PyTorch={ref_output.shape[0]}"
        )

        if ref_output.shape[0] > 0:
            # Sort both by score descending for stable comparison
            onnx_sorted = onnx_output[np.argsort(-onnx_output[:, 16])]
            ref_sorted = ref_output[np.argsort(-ref_output[:, 16])]

            # Compare bounding box coordinates and keypoints
            np.testing.assert_allclose(
                onnx_sorted[:, :16], ref_sorted[:, :16], rtol=1e-3, atol=1e-4
            )
            # Compare scores (PyTorch reference uses average score for blended,
            # our ONNX uses winner score — both valid but may differ slightly)
            # Check that scores are at least close
            np.testing.assert_allclose(
                onnx_sorted[:, 16], ref_sorted[:, 16], rtol=0.1, atol=0.05
            )


class TestE2EModelBack:
    """Back variant end-to-end model tests."""

    def test_model_validates(self, back_e2e_model):
        model, _ = back_e2e_model
        onnx.checker.check_model(model)

    def test_inference_runs(self, back_e2e_model):
        _, path = back_e2e_model
        sess = ort.InferenceSession(path)
        dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)
        results = sess.run(
            None,
            {
                "input": dummy,
                "max_output_boxes": np.array(10, dtype=np.int64),
                "iou_threshold": np.array(0.3, dtype=np.float32),
                "score_threshold": np.array(0.75, dtype=np.float32),
            },
        )
        assert len(results) == 1
        detections = results[0]
        assert detections.ndim == 2
        assert detections.shape[1] == 17
