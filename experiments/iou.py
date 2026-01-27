import numpy as np

def calculate_iou_matrix(boxes1, boxes2):
    """
    Computes the pairwise Intersection over Union (IoU) matrix between two lists of bounding boxes.
    
    Args:
        boxes1 (np.ndarray): Array of bounding boxes with shape (N, 4).
                             Format: (x1, y1, x2, y2).
        boxes2 (np.ndarray): Array of bounding boxes with shape (M, 4).
                             Format: (x1, y1, x2, y2).
                             
    Returns:
        np.ndarray: IoU matrix with shape (N, M), where value [i, j] is the 
                    IoU between boxes1[i] and boxes2[j].
    """
    # Ensure inputs are numpy arrays
    boxes1 = np.asarray(boxes1)
    boxes2 = np.asarray(boxes2)
    
    # 1. Calculate the area of each box in both sets
    # Area = (x2 - x1) * (y2 - y1)
    # We allow x2 to be equal to x1 (area 0) but usually x2 > x1.
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # 2. Calculate the coordinates of the intersection rectangles
    # We use broadcasting: 
    # boxes1[:, None, 0] has shape (N, 1)
    # boxes2[:, 0]       has shape (M,)
    # result             has shape (N, M)
    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[:, 3])
    
    # 3. Calculate intersection area
    # Use .clip(min=0) to handle cases where there is no intersection (negative width/height)
    inter_w = (inter_x2 - inter_x1).clip(min=0)
    inter_h = (inter_y2 - inter_y1).clip(min=0)
    inter_area = inter_w * inter_h
    
    # 4. Calculate Union area
    # Union = Area1 + Area2 - Intersection
    # We need to broadcast area1 (N,) and area2 (M,) to (N, M)
    union_area = area1[:, None] + area2[None, :] - inter_area
    
    # 5. Calculate IoU
    # Add a small epsilon to avoid division by zero if union is 0
    iou_matrix = inter_area / (union_area + 1e-6)
    
    return iou_matrix