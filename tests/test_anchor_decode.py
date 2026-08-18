"""Tests for the anchor box decoding ONNX function.

Verifies the onnxscript decode_boxes implementation against a numpy
reference port of BlazeFace's _decode_boxes method. Tests cover:

- Model validation
- Output shape correctness
- Zero raw boxes produce anchor-centered zero-size boxes
- Known-value decoding against hand-computed results
- Equivalence to numpy reference with real anchors (896) and random data
- Both front (scale=128) and back (scale=256) configurations
"""

import os
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from blazefaceonnx.anchor_decode import decode_boxes, make_decode_boxes_model

SUBMODULE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "external", "MediaPipePytorch"
)


# ── Numpy reference implementation ────────────────────────────────────


def decode_boxes_np(
    raw_boxes: np.ndarray, anchors: np.ndarray, scale: float
) -> np.ndarray:
    """Numpy reference of BlazeFace _decode_boxes.

    Ported from BlazeDetector._decode_boxes in zmurez/MediaPipePyTorch
    (blazebase.py, Apache-2.0).

    Args:
        raw_boxes: (N, 16) raw regressions.
        anchors:   (N, 4) in [cx, cy, w, h] format.
        scale:     scalar.

    Returns:
        (N, 16) decoded in [ymin, xmin, ymax, xmax, kp1_x, kp1_y, ...] format.
    """
    boxes = np.zeros_like(raw_boxes)

    a_cx = anchors[:, 0]
    a_cy = anchors[:, 1]
    a_w = anchors[:, 2]
    a_h = anchors[:, 3]

    x_center = raw_boxes[:, 0] / scale * a_w + a_cx
    y_center = raw_boxes[:, 1] / scale * a_h + a_cy
    w = raw_boxes[:, 2] / scale * a_w
    h = raw_boxes[:, 3] / scale * a_h

    boxes[:, 0] = y_center - h / 2.0  # ymin
    boxes[:, 1] = x_center - w / 2.0  # xmin
    boxes[:, 2] = y_center + h / 2.0  # ymax
    boxes[:, 3] = x_center + w / 2.0  # xmax

    for k in range(6):
        offset = 4 + k * 2
        boxes[:, offset] = raw_boxes[:, offset] / scale * a_w + a_cx  # kp_x
        boxes[:, offset + 1] = raw_boxes[:, offset + 1] / scale * a_h + a_cy  # kp_y

    return boxes


# ── Test fixtures ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def onnx_session():
    """Create an onnxruntime session for the decode_boxes model."""
    model = make_decode_boxes_model()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        f.write(model.SerializeToString())
        model_path = f.name

    sess = ort.InferenceSession(model_path)
    yield sess
    os.unlink(model_path)


def _run_onnx(
    sess: ort.InferenceSession,
    raw_boxes: np.ndarray,
    anchors: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Run the ONNX decode_boxes function."""
    return sess.run(
        None,
        {
            "raw_boxes": raw_boxes.astype(np.float32),
            "anchors": anchors.astype(np.float32),
            "scale": np.array(scale, dtype=np.float32),
        },
    )[0]


# ── Tests ─────────────────────────────────────────────────────────────


class TestModelExport:
    """Tests for ONNX model export and validation."""

    def test_model_validates(self):
        model = make_decode_boxes_model()
        onnx.checker.check_model(model)

    def test_model_inputs_have_shapes(self):
        model = make_decode_boxes_model()
        for inp in model.graph.input:
            shape = inp.type.tensor_type.shape
            if inp.name == "raw_boxes":
                assert len(shape.dim) == 2
                assert shape.dim[1].dim_value == 16
            elif inp.name == "anchors":
                assert len(shape.dim) == 2
                assert shape.dim[1].dim_value == 4
            elif inp.name == "scale":
                assert len(shape.dim) == 0


class TestOutputShape:
    """Output shapes must match input dimensions."""

    def test_output_shape_small(self, onnx_session):
        N = 10
        raw_boxes = np.zeros((N, 16), dtype=np.float32)
        anchors = np.ones((N, 4), dtype=np.float32) * 0.5
        out = _run_onnx(onnx_session, raw_boxes, anchors, 128.0)
        assert out.shape == (N, 16)

    def test_output_shape_896(self, onnx_session):
        N = 896
        raw_boxes = np.zeros((N, 16), dtype=np.float32)
        anchors = np.ones((N, 4), dtype=np.float32) * 0.5
        out = _run_onnx(onnx_session, raw_boxes, anchors, 128.0)
        assert out.shape == (N, 16)


class TestZeroRawBoxes:
    """Zero raw boxes should decode to anchor-center-based boxes with zero size."""

    def test_zero_decode(self, onnx_session):
        N = 5
        raw_boxes = np.zeros((N, 16), dtype=np.float32)
        # Anchors at (0.5, 0.5) with width=0.2, height=0.3
        anchors = np.tile(
            np.array([[0.5, 0.5, 0.2, 0.3]], dtype=np.float32), (N, 1)
        )

        out = _run_onnx(onnx_session, raw_boxes, anchors, 128.0)

        # With zero raw, decoded center = anchor center, size = 0
        # ymin = cy - 0 = 0.5, xmin = cx - 0 = 0.5, ymax = 0.5, xmax = 0.5
        expected_box = [0.5, 0.5, 0.5, 0.5]
        # Keypoints all at anchor center: (cx, cy) = (0.5, 0.5)
        expected_kp = [0.5, 0.5] * 6

        for i in range(N):
            np.testing.assert_allclose(out[i, :4], expected_box, atol=1e-6)
            np.testing.assert_allclose(out[i, 4:], expected_kp, atol=1e-6)


class TestKnownValues:
    """Hand-computed decode results."""

    def test_known_decode(self, onnx_session):
        # Single anchor at center (0.5, 0.5) with size (0.2, 0.3)
        anchors = np.array([[0.5, 0.5, 0.2, 0.3]], dtype=np.float32)
        scale = 128.0

        # raw_boxes: cx_raw=128, cy_raw=128, w_raw=128, h_raw=128, then zeros for kps
        raw_boxes = np.zeros((1, 16), dtype=np.float32)
        raw_boxes[0, 0] = 128.0  # raw cx
        raw_boxes[0, 1] = 128.0  # raw cy
        raw_boxes[0, 2] = 128.0  # raw w
        raw_boxes[0, 3] = 128.0  # raw h

        out = _run_onnx(onnx_session, raw_boxes, anchors, scale)

        # cx = 128/128 * 0.2 + 0.5 = 0.7
        # cy = 128/128 * 0.3 + 0.5 = 0.8
        # w  = 128/128 * 0.2 = 0.2
        # h  = 128/128 * 0.3 = 0.3
        # ymin = 0.8 - 0.15 = 0.65
        # xmin = 0.7 - 0.1 = 0.6
        # ymax = 0.8 + 0.15 = 0.95
        # xmax = 0.7 + 0.1 = 0.8
        np.testing.assert_allclose(out[0, 0], 0.65, atol=1e-5)
        np.testing.assert_allclose(out[0, 1], 0.6, atol=1e-5)
        np.testing.assert_allclose(out[0, 2], 0.95, atol=1e-5)
        np.testing.assert_allclose(out[0, 3], 0.8, atol=1e-5)

    def test_keypoint_decode(self, onnx_session):
        anchors = np.array([[0.5, 0.5, 0.2, 0.3]], dtype=np.float32)
        scale = 128.0

        raw_boxes = np.zeros((1, 16), dtype=np.float32)
        # Set first keypoint: raw_kp_x=64, raw_kp_y=64
        raw_boxes[0, 4] = 64.0  # kp0 x
        raw_boxes[0, 5] = 64.0  # kp0 y

        out = _run_onnx(onnx_session, raw_boxes, anchors, scale)

        # kp0_x = 64/128 * 0.2 + 0.5 = 0.6
        # kp0_y = 64/128 * 0.3 + 0.5 = 0.65
        np.testing.assert_allclose(out[0, 4], 0.6, atol=1e-5)
        np.testing.assert_allclose(out[0, 5], 0.65, atol=1e-5)


class TestReferenceEquivalence:
    """Equivalence between ONNX and numpy reference implementations."""

    def test_random_data_front(self, onnx_session):
        """Front model: scale=128, random data."""
        rng = np.random.RandomState(42)
        N = 100
        raw_boxes = rng.randn(N, 16).astype(np.float32) * 50
        anchors = np.column_stack(
            [
                rng.rand(N) * 0.8 + 0.1,  # cx
                rng.rand(N) * 0.8 + 0.1,  # cy
                rng.rand(N) * 0.3 + 0.05,  # w
                rng.rand(N) * 0.3 + 0.05,  # h
            ]
        ).astype(np.float32)
        scale = 128.0

        onnx_out = _run_onnx(onnx_session, raw_boxes, anchors, scale)
        ref_out = decode_boxes_np(raw_boxes, anchors, scale)

        np.testing.assert_allclose(onnx_out, ref_out, rtol=1e-5, atol=1e-6)

    def test_random_data_back(self, onnx_session):
        """Back model: scale=256, random data."""
        rng = np.random.RandomState(99)
        N = 100
        raw_boxes = rng.randn(N, 16).astype(np.float32) * 80
        anchors = np.column_stack(
            [
                rng.rand(N) * 0.8 + 0.1,
                rng.rand(N) * 0.8 + 0.1,
                rng.rand(N) * 0.3 + 0.05,
                rng.rand(N) * 0.3 + 0.05,
            ]
        ).astype(np.float32)
        scale = 256.0

        onnx_out = _run_onnx(onnx_session, raw_boxes, anchors, scale)
        ref_out = decode_boxes_np(raw_boxes, anchors, scale)

        np.testing.assert_allclose(onnx_out, ref_out, rtol=1e-5, atol=1e-6)

    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "external",
                "MediaPipePytorch",
                "anchors_face.npy",
            )
        ),
        reason="Anchor file not available",
    )
    def test_real_anchors_front(self, onnx_session):
        """Real 896 anchors from front model."""
        anchors = np.load(
            os.path.join(SUBMODULE_DIR, "anchors_face.npy")
        ).astype(np.float32)
        rng = np.random.RandomState(123)
        raw_boxes = rng.randn(896, 16).astype(np.float32) * 30
        scale = 128.0

        onnx_out = _run_onnx(onnx_session, raw_boxes, anchors, scale)
        ref_out = decode_boxes_np(raw_boxes, anchors, scale)

        np.testing.assert_allclose(onnx_out, ref_out, rtol=1e-5, atol=1e-6)

    @pytest.mark.skipif(
        not os.path.exists(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "external",
                "MediaPipePytorch",
                "anchors_face_back.npy",
            )
        ),
        reason="Anchor file not available",
    )
    def test_real_anchors_back(self, onnx_session):
        """Real 896 anchors from back model."""
        anchors = np.load(
            os.path.join(SUBMODULE_DIR, "anchors_face_back.npy")
        ).astype(np.float32)
        rng = np.random.RandomState(456)
        raw_boxes = rng.randn(896, 16).astype(np.float32) * 50
        scale = 256.0

        onnx_out = _run_onnx(onnx_session, raw_boxes, anchors, scale)
        ref_out = decode_boxes_np(raw_boxes, anchors, scale)

        np.testing.assert_allclose(onnx_out, ref_out, rtol=1e-5, atol=1e-6)
