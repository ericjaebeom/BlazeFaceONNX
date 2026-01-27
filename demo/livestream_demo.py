"""Live-stream BlazeFace end-to-end ONNX inference demo."""

import argparse

import cv2
import numpy as np
import onnxruntime as ort


def parse_args():
    p = argparse.ArgumentParser(description="BlazeFace E2E live-stream demo")
    p.add_argument("model", help="Path to end-to-end .onnx model")
    p.add_argument("--camera", type=int, default=0, help="Camera device index")
    p.add_argument("--max-output-boxes", type=int, default=1)
    p.add_argument("--iou-threshold", type=float, default=0.35)
    p.add_argument("--score-threshold", type=float, default=0.5)
    return p.parse_args()


def get_model_input_size(session):
    """Detect front (128) vs back (256) from the input tensor shape."""
    input_shape = session.get_inputs()[0].shape  # (1, 3, H, W)
    return int(input_shape[2]), int(input_shape[3])


def preprocess(frame_bgr, model_h, model_w):
    """Resize with letterboxing to model input size.

    Returns the CHW float32 tensor and (scale, pad_top, pad_left) for
    mapping detections back to original coordinates.
    """
    h, w = frame_bgr.shape[:2]
    scale = min(model_w / w, model_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame_bgr, (new_w, new_h))

    pad_top = (model_h - new_h) // 2
    pad_left = (model_w - new_w) // 2
    canvas = np.zeros((model_h, model_w, 3), dtype=np.uint8)
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    # HWC BGR -> CHW RGB, float32 [0, 1]
    tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return tensor[np.newaxis], scale, pad_top, pad_left


def draw_detections(frame_bgr, detections, scale, pad_top, pad_left, model_h, model_w):
    """Draw bounding boxes and keypoints on the frame."""
    h, w = frame_bgr.shape[:2]
    s = max(h, w) / 500.0
    box_thick = max(1, round(2 * s))
    txt_scale = 0.5 * s
    txt_thick = max(1, round(s))
    kp_radius = max(1, round(2 * s))

    for det in detections:
        ymin, xmin, ymax, xmax = det[0], det[1], det[2], det[3]
        score = det[16]

        # Normalized [0,1] -> pixel coords in padded image -> original image
        px_xmin = xmin * model_w - pad_left
        px_xmax = xmax * model_w - pad_left
        px_ymin = ymin * model_h - pad_top
        px_ymax = ymax * model_h - pad_top

        ox1 = int(px_xmin / scale)
        oy1 = int(px_ymin / scale)
        ox2 = int(px_xmax / scale)
        oy2 = int(px_ymax / scale)

        cv2.rectangle(frame_bgr, (ox1, oy1), (ox2, oy2), (0, 255, 0), box_thick)
        cv2.putText(
            frame_bgr, f"{score:.2f}", (ox1, oy1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, txt_scale, (0, 255, 0), txt_thick,
        )

        # 6 keypoints starting at index 4, each (x, y)
        for i in range(6):
            kp_x = det[4 + i * 2] * model_w - pad_left
            kp_y = det[4 + i * 2 + 1] * model_h - pad_top
            kx = int(kp_x / scale)
            ky = int(kp_y / scale)
            cv2.circle(frame_bgr, (kx, ky), kp_radius, (0, 0, 255), -1)


def main():
    args = parse_args()

    sess = ort.InferenceSession(args.model)
    model_h, model_w = get_model_input_size(sess)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tensor, scale, pad_top, pad_left = preprocess(frame, model_h, model_w)

        detections = sess.run(None, {
            "input": tensor,
            "max_output_boxes": np.array(args.max_output_boxes, dtype=np.int64),
            "iou_threshold": np.array(args.iou_threshold, dtype=np.float32),
            "score_threshold": np.array(args.score_threshold, dtype=np.float32),
        })[0]

        draw_detections(frame, detections, scale, pad_top, pad_left, model_h, model_w)
        cv2.imshow("BlazeFace Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
