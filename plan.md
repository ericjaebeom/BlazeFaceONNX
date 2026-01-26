* num_anchors: 896

* predicted_tensors:
    * regressions: (num_anchors, 16)
    * probs: (num_anchors, 1)

* detections: (num_anchors, 17)

* filtered_detections: (num_faces, 17)

---

* predicted_tensors_to_detections(regressions, probs) -> detections
    * decode_regressions(regressions, anchors) -> decoded_regressions: (num_anchors, 16)

* weighted_non_max_suppression(detections, ...) -> filtered_detections

---

* weighted average of probabilities (since the regressions are weighted averaged)
    * is always between the MediaPipe implementation and the PyTorch port
