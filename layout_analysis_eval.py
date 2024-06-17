from typing import List, Tuple
import numpy as np

def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Each box is defined by a tuple (top, left, bottom, right).
    """
    top1, left1, bottom1, right1 = box1
    top2, left2, bottom2, right2 = box2

    intersect_top = max(top1, top2)
    intersect_left = max(left1, left2)
    intersect_bottom = min(bottom1, bottom2)
    intersect_right = min(right1, right2)

    intersect_area = max(0, intersect_bottom - intersect_top) * max(0, intersect_right - intersect_left)
    box1_area = (bottom1 - top1) * (right1 - left1)
    box2_area = (bottom2 - top2) * (right2 - left2)
    union_area = box1_area + box2_area - intersect_area

    if union_area == 0:
        return 0

    return intersect_area / union_area

def calculate_metrics(ground_truth: List[Tuple[int, int, int, int]], 
                      detected: List[Tuple[int, int, int, int]], 
                      iou_threshold: float = 0.5) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F-score based on ground truth and detected bounding boxes.
    
    Parameters:
    - ground_truth: List of tuples representing ground truth bounding boxes.
    - detected: List of tuples representing detected bounding boxes.
    - iou_threshold: IoU threshold to determine a true positive match (default is 0.5).
    
    Returns:
    - precision: Precision of the detected bounding boxes.
    - recall: Recall of the detected bounding boxes.
    - f_score: F1 score of the detected bounding boxes.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    matched_gt = set()

    for det in detected:
        match_found = False
        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            iou = calculate_iou(det, gt)
            if iou >= iou_threshold:
                true_positives += 1
                matched_gt.add(i)
                match_found = True
                break
        if not match_found:
            false_positives += 1

    false_negatives = len(ground_truth) - len(matched_gt)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f_score

# Example usage
ground_truth = [(50, 50, 100, 100), (150, 150, 200, 200)]
detected = [(55, 55, 95, 95), (160, 160, 190, 190), (300, 300, 350, 350)]

precision, recall, f_score = calculate_metrics(ground_truth, detected)
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F-score: {f_score:.2f}")
