"""
This module provides functionality for detecting object using YOLOv8.
"""
import numpy as np

from ultralytics import YOLO
from detection.model.detection import DetectionResults
from detection.model.bounding_box import from_boxes


def detect_objects(img: np.ndarray, model: YOLO, conf_threshold: float) -> DetectionResults:
    """
    Given an image, detect objects in it.

    :param img: The image
    :param model: The YOLO model used for object detection
    :param conf_threshold: The confidence score threshold
    :return: List of boxes representing the objects detected
    """
    predictions = model.predict(source=img, conf=conf_threshold)
    boxes = []
    [[boxes.append(bounding_box) for bounding_box in from_boxes(detection.boxes, model.names)]
     for detection in predictions]
    return DetectionResults(boxes, img)
