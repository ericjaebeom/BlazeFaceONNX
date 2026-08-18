"""Tests for the weighted NMS ONNX function.

Verifies the onnxscript implementation against a numpy reference
implementation of the weighted NMS algorithm. Tests cover:

- Empty result (no detections pass score threshold)
- Single detection (no blending needed)
- Overlapping detections (blending with score-weighted averaging)
- Non-overlapping detections (multiple independent outputs)
- Equivalence to reference on realistic BlazeFace-like data
"""

import tempfile
import os

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from blazefaceonnx.weighted_nms import (
    weighted_nms,
    compute_iou_matrix,
    make_weighted_nms_model,
)


# ── Numpy reference implementation ────────────────────────────────────


def _iou_matrix_np(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (M, 4) and (N, 4) boxes in [ymin,xmin,ymax,xmax]."""
    a = boxes_a[:, None, :]  # (M, 1, 4)
    b = boxes_b[None, :, :]  # (1, N, 4)

    inter_ymin = np.maximum(a[..., 0], b[..., 0])
    inter_xmin = np.maximum(a[..., 1], b[..., 1])
    inter_ymax = np.minimum(a[..., 2], b[..., 2])
    inter_xmax = np.minimum(a[..., 3], b[..., 3])

    inter_h = np.clip(inter_ymax - inter_ymin, 0, None)
    inter_w = np.clip(inter_xmax - inter_xmin, 0, None)
    inter_area = inter_h * inter_w

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = np.clip(area_a + area_b - inter_area, 1e-6, None)
    return inter_area / union


def reference_weighted_nms(
    regressions: np.ndarray,
    scores: np.ndarray,
    max_output_boxes: int,
    iou_threshold: float,
    score_threshold: float,
) -> np.ndarray:
    """Numpy reference of the iterative weighted NMS algorithm.

    Greedy NMS with score-weighted coordinate blending. The final score
    is the winner's own score (the highest-scoring box in each group).

    Returns:
        (S, 17) array of blended detections, or (0, 17) if none selected.
    """
    # Filter by score threshold
    mask = scores > score_threshold
    if not np.any(mask):
        return np.zeros((0, 17), dtype=np.float32)

    candidate_indices = np.where(mask)[0]
    candidate_reg = regressions[candidate_indices]
    candidate_scores = scores[candidate_indices]

    # Sort by descending score
    order = np.argsort(-candidate_scores)
    candidate_indices = candidate_indices[order]
    candidate_reg = candidate_reg[order]
    candidate_scores = candidate_scores[order]

    boxes = candidate_reg[:, :4]
    output = []
    remaining = np.arange(len(candidate_scores))

    while len(remaining) > 0 and len(output) < max_output_boxes:
        # Pick the highest-scoring remaining candidate
        best = remaining[0]
        best_box = boxes[best : best + 1]

        # Compute IoU with all remaining candidates
        ious = _iou_matrix_np(best_box, boxes[remaining])[0]

        # Find overlapping group
        overlap_mask = ious > iou_threshold
        overlapping = remaining[overlap_mask]
        remaining = remaining[~overlap_mask]

        # Score-weighted average of regression coordinates
        overlap_scores = candidate_scores[overlapping]
        overlap_regs = candidate_reg[overlapping]

        total = overlap_scores.sum()
        blended_reg = (overlap_regs * overlap_scores[:, None]).sum(axis=0) / total
        winner_score = overlap_scores[0]  # highest-scoring (sorted descending)

        output.append(np.concatenate([blended_reg, [winner_score]]))

    if not output:
        return np.zeros((0, 17), dtype=np.float32)
    return np.array(output, dtype=np.float32)


# ── Test fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def onnx_session():
    """Create an onnxruntime session for the weighted_nms model."""
    model = make_weighted_nms_model()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(model.SerializeToString())
        model_path = f.name

    sess = ort.InferenceSession(model_path)
    yield sess
    os.unlink(model_path)


def _run_onnx(
    sess: ort.InferenceSession,
    regressions: np.ndarray,
    scores: np.ndarray,
    max_output_boxes: int = 10,
    iou_threshold: float = 0.3,
    score_threshold: float = 0.75,
) -> np.ndarray:
    """Run the ONNX weighted NMS function."""
    return sess.run(
        None,
        {
            "regressions": regressions.astype(np.float32),
            "scores": scores.astype(np.float32),
            "max_output_boxes": np.array(max_output_boxes, dtype=np.int64),
            "iou_threshold": np.array(iou_threshold, dtype=np.float32),
            "score_threshold": np.array(score_threshold, dtype=np.float32),
        },
    )[0]


# ── Helper to build test data ────────────────────────────────────────


def _make_detection(
    ymin: float, xmin: float, ymax: float, xmax: float, score: float, seed: int = 0
) -> tuple[np.ndarray, float]:
    """Create a single detection (regression row + score).

    Keypoint coordinates (indices 4-15) are filled deterministically from seed.
    """
    rng = np.random.RandomState(seed)
    reg = np.zeros(16, dtype=np.float32)
    reg[:4] = [ymin, xmin, ymax, xmax]
    reg[4:] = rng.rand(12).astype(np.float32)
    return reg, np.float32(score)


# ── Tests ─────────────────────────────────────────────────────────────


class TestModelExport:
    """Tests for ONNX model export and validation."""

    def test_model_validates(self):
        model = make_weighted_nms_model()
        onnx.checker.check_model(model)

    def test_model_has_local_function(self):
        model = make_weighted_nms_model()
        func_names = [f.name for f in model.functions]
        assert "compute_iou_matrix" in func_names

    def test_model_inputs_have_shapes(self):
        model = make_weighted_nms_model()
        for inp in model.graph.input:
            shape = inp.type.tensor_type.shape
            if inp.name == "regressions":
                assert len(shape.dim) == 2
                assert shape.dim[1].dim_value == 16
            elif inp.name == "scores":
                assert len(shape.dim) == 1


class TestEmptyResult:
    """No detections pass the score threshold."""

    def test_all_below_threshold(self, onnx_session):
        N = 20
        reg, _ = _make_detection(0.1, 0.1, 0.3, 0.3, 0.0)
        regressions = np.tile(reg, (N, 1))
        scores = np.full(N, 0.1, dtype=np.float32)

        out = _run_onnx(onnx_session, regressions, scores, score_threshold=0.75)
        assert out.shape == (0, 17)

    def test_empty_matches_reference(self, onnx_session):
        N = 10
        reg, _ = _make_detection(0.0, 0.0, 0.5, 0.5, 0.0)
        regressions = np.tile(reg, (N, 1))
        scores = np.full(N, 0.3, dtype=np.float32)

        onnx_out = _run_onnx(onnx_session, regressions, scores, score_threshold=0.75)
        ref_out = reference_weighted_nms(regressions, scores, 10, 0.3, 0.75)
        assert onnx_out.shape == ref_out.shape == (0, 17)


class TestSingleDetection:
    """Exactly one detection passes the threshold with no overlapping peers."""

    def test_single_passthrough(self, onnx_session):
        """Single detection should be returned unchanged."""
        N = 10
        regressions = np.zeros((N, 16), dtype=np.float32)
        scores = np.full(N, 0.1, dtype=np.float32)

        # Place one valid detection
        regressions[3], scores[3] = _make_detection(
            0.2, 0.3, 0.5, 0.6, 0.9, seed=42
        )

        out = _run_onnx(onnx_session, regressions, scores, score_threshold=0.75)
        assert out.shape == (1, 17)
        np.testing.assert_allclose(out[0, :16], regressions[3], rtol=1e-5)
        np.testing.assert_allclose(out[0, 16], 0.9, rtol=1e-5)

    def test_single_matches_reference(self, onnx_session):
        N = 10
        regressions = np.zeros((N, 16), dtype=np.float32)
        scores = np.full(N, 0.1, dtype=np.float32)
        regressions[5], scores[5] = _make_detection(
            0.1, 0.1, 0.4, 0.4, 0.85, seed=7
        )

        onnx_out = _run_onnx(onnx_session, regressions, scores, score_threshold=0.5)
        ref_out = reference_weighted_nms(regressions, scores, 10, 0.3, 0.5)

        assert onnx_out.shape == ref_out.shape
        np.testing.assert_allclose(onnx_out, ref_out, rtol=1e-5)


class TestOverlappingDetections:
    """Multiple overlapping detections should be blended."""

    def test_identical_boxes_blended(self, onnx_session):
        """Two identical boxes: weighted average of their regressions."""
        N = 10
        regressions = np.zeros((N, 16), dtype=np.float32)
        scores = np.full(N, 0.1, dtype=np.float32)

        regressions[0], scores[0] = _make_detection(
            0.2, 0.2, 0.5, 0.5, 0.8, seed=1
        )
        regressions[1], scores[1] = _make_detection(
            0.2, 0.2, 0.5, 0.5, 0.9, seed=2
        )

        out = _run_onnx(
            onnx_session, regressions, scores,
            iou_threshold=0.3, score_threshold=0.5,
        )
        assert out.shape == (1, 17)

        # Expected: weighted average of coords, max score
        w0, w1 = 0.8, 0.9
        expected_reg = (w0 * regressions[0] + w1 * regressions[1]) / (w0 + w1)
        np.testing.assert_allclose(out[0, :16], expected_reg, rtol=1e-4)
        np.testing.assert_allclose(out[0, 16], 0.9, rtol=1e-5)

    def test_partial_overlap_blended(self, onnx_session):
        """Two boxes with significant overlap should be blended."""
        N = 10
        regressions = np.zeros((N, 16), dtype=np.float32)
        scores = np.full(N, 0.1, dtype=np.float32)

        # Two boxes with ~0.5 IoU (significant overlap)
        regressions[0], scores[0] = _make_detection(
            0.1, 0.1, 0.5, 0.5, 0.95, seed=10
        )
        regressions[1], scores[1] = _make_detection(
            0.2, 0.2, 0.6, 0.6, 0.85, seed=11
        )
        # IoU = intersection / union = (0.3*0.3) / (0.4*0.4 + 0.4*0.4 - 0.09)
        #      = 0.09 / 0.23 ≈ 0.39 > 0.3

        out = _run_onnx(
            onnx_session, regressions, scores,
            iou_threshold=0.3, score_threshold=0.5,
        )
        assert out.shape == (1, 17)

        w0, w1 = 0.95, 0.85
        expected_reg = (w0 * regressions[0] + w1 * regressions[1]) / (w0 + w1)
        np.testing.assert_allclose(out[0, :16], expected_reg, rtol=1e-4)
        np.testing.assert_allclose(out[0, 16], 0.95, rtol=1e-5)

    def test_three_overlapping_blended(self, onnx_session):
        """Three highly overlapping boxes should all be blended into one."""
        N = 10
        regressions = np.zeros((N, 16), dtype=np.float32)
        scores = np.full(N, 0.1, dtype=np.float32)

        regressions[0], scores[0] = _make_detection(
            0.1, 0.1, 0.5, 0.5, 0.9, seed=20
        )
        regressions[1], scores[1] = _make_detection(
            0.1, 0.1, 0.5, 0.5, 0.8, seed=21
        )
        regressions[2], scores[2] = _make_detection(
            0.1, 0.1, 0.5, 0.5, 0.85, seed=22
        )

        out = _run_onnx(
            onnx_session, regressions, scores,
            iou_threshold=0.3, score_threshold=0.5,
        )
        assert out.shape == (1, 17)

        w = np.array([0.9, 0.8, 0.85])
        expected_reg = (
            w[0] * regressions[0] + w[1] * regressions[1] + w[2] * regressions[2]
        ) / w.sum()
        np.testing.assert_allclose(out[0, :16], expected_reg, rtol=1e-4)
        np.testing.assert_allclose(out[0, 16], 0.9, rtol=1e-5)

    def test_overlap_matches_reference(self, onnx_session):
        N = 10
        regressions = np.zeros((N, 16), dtype=np.float32)
        scores = np.full(N, 0.1, dtype=np.float32)

        regressions[0], scores[0] = _make_detection(
            0.1, 0.1, 0.5, 0.5, 0.9, seed=30
        )
        regressions[1], scores[1] = _make_detection(
            0.15, 0.15, 0.55, 0.55, 0.8, seed=31
        )

        onnx_out = _run_onnx(
            onnx_session, regressions, scores,
            iou_threshold=0.3, score_threshold=0.5,
        )
        ref_out = reference_weighted_nms(regressions, scores, 10, 0.3, 0.5)

        assert onnx_out.shape == ref_out.shape
        np.testing.assert_allclose(onnx_out, ref_out, rtol=1e-4)


class TestNonOverlapping:
    """Multiple detections that don't overlap should all be returned."""

    def test_two_separate_detections(self, onnx_session):
        N = 10
        regressions = np.zeros((N, 16), dtype=np.float32)
        scores = np.full(N, 0.1, dtype=np.float32)

        # Far apart boxes
        regressions[0], scores[0] = _make_detection(
            0.0, 0.0, 0.1, 0.1, 0.85, seed=40
        )
        regressions[5], scores[5] = _make_detection(
            0.8, 0.8, 0.95, 0.95, 0.95, seed=41
        )

        out = _run_onnx(
            onnx_session, regressions, scores,
            iou_threshold=0.3, score_threshold=0.5,
        )
        assert out.shape == (2, 17)
        # NMS returns in descending score order
        np.testing.assert_allclose(out[0, :16], regressions[5], rtol=1e-5)
        np.testing.assert_allclose(out[0, 16], 0.95, rtol=1e-5)
        np.testing.assert_allclose(out[1, :16], regressions[0], rtol=1e-5)
        np.testing.assert_allclose(out[1, 16], 0.85, rtol=1e-5)

    def test_non_overlap_matches_reference(self, onnx_session):
        N = 10
        regressions = np.zeros((N, 16), dtype=np.float32)
        scores = np.full(N, 0.1, dtype=np.float32)

        regressions[0], scores[0] = _make_detection(
            0.0, 0.0, 0.15, 0.15, 0.9, seed=50
        )
        regressions[5], scores[5] = _make_detection(
            0.7, 0.7, 0.9, 0.9, 0.8, seed=51
        )

        onnx_out = _run_onnx(
            onnx_session, regressions, scores,
            iou_threshold=0.3, score_threshold=0.5,
        )
        ref_out = reference_weighted_nms(regressions, scores, 10, 0.3, 0.5)

        assert onnx_out.shape == ref_out.shape
        np.testing.assert_allclose(onnx_out, ref_out, rtol=1e-4)


class TestMaxOutputBoxes:
    """The max_output_boxes parameter should limit output count."""

    def test_limit_to_one(self, onnx_session):
        N = 10
        regressions = np.zeros((N, 16), dtype=np.float32)
        scores = np.full(N, 0.1, dtype=np.float32)

        regressions[0], scores[0] = _make_detection(
            0.0, 0.0, 0.1, 0.1, 0.9, seed=60
        )
        regressions[5], scores[5] = _make_detection(
            0.8, 0.8, 0.95, 0.95, 0.85, seed=61
        )

        out = _run_onnx(
            onnx_session, regressions, scores,
            max_output_boxes=1, iou_threshold=0.3, score_threshold=0.5,
        )
        assert out.shape == (1, 17)
        # Should keep the highest-scoring detection
        np.testing.assert_allclose(out[0, 16], 0.9, rtol=1e-5)


class TestMixedScenario:
    """Realistic scenario: some overlapping, some non-overlapping, some below threshold."""

    def test_mixed_matches_reference(self, onnx_session):
        """Cluster A (overlapping), detection B (isolated), rest below threshold."""
        N = 15
        regressions = np.zeros((N, 16), dtype=np.float32)
        scores = np.full(N, 0.1, dtype=np.float32)

        # Cluster A: two overlapping boxes in top-left
        regressions[0], scores[0] = _make_detection(
            0.1, 0.1, 0.4, 0.4, 0.95, seed=70
        )
        regressions[1], scores[1] = _make_detection(
            0.15, 0.12, 0.42, 0.38, 0.88, seed=71
        )

        # Detection B: isolated in bottom-right
        regressions[7], scores[7] = _make_detection(
            0.7, 0.7, 0.9, 0.9, 0.8, seed=72
        )

        iou_thresh = 0.3
        score_thresh = 0.5

        onnx_out = _run_onnx(
            onnx_session, regressions, scores,
            iou_threshold=iou_thresh, score_threshold=score_thresh,
        )
        ref_out = reference_weighted_nms(
            regressions, scores, 10, iou_thresh, score_thresh
        )

        assert onnx_out.shape == ref_out.shape
        np.testing.assert_allclose(onnx_out, ref_out, rtol=1e-4)


class TestBlazeFaceRealistic:
    """Test with dimensions matching real BlazeFace output (896 anchors)."""

    def test_896_anchors_sparse_detections(self, onnx_session):
        """896 anchors, only a couple pass threshold — typical BlazeFace case."""
        N = 896
        rng = np.random.RandomState(99)

        # Background: valid but low-score boxes spread across the image
        regressions = np.zeros((N, 16), dtype=np.float32)
        for i in range(N):
            y, x = rng.rand(2) * 0.8
            h, w = 0.05 + rng.rand(2) * 0.15
            regressions[i, 0] = y
            regressions[i, 1] = x
            regressions[i, 2] = y + h
            regressions[i, 3] = x + w
            regressions[i, 4:] = rng.rand(12)
        scores = rng.rand(N).astype(np.float32) * 0.3  # all below 0.3

        # Two overlapping face detections
        regressions[100], scores[100] = _make_detection(
            0.3, 0.4, 0.6, 0.7, 0.92, seed=100
        )
        regressions[101], scores[101] = _make_detection(
            0.32, 0.42, 0.62, 0.72, 0.88, seed=101
        )

        iou_thresh = 0.3
        score_thresh = 0.75

        onnx_out = _run_onnx(
            onnx_session, regressions, scores,
            iou_threshold=iou_thresh, score_threshold=score_thresh,
        )
        ref_out = reference_weighted_nms(
            regressions, scores, 10, iou_thresh, score_thresh
        )

        assert onnx_out.shape == ref_out.shape
        np.testing.assert_allclose(onnx_out, ref_out, rtol=1e-4)
