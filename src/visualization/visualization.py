"""
This module provides functionality for visualizing results of object detection.
"""
import cv2
import numpy as np

from detection.model.detection import DetectionResults
from detection.model.bounding_box import BoundingBox


GREEN = (0, 255, 0)


def visualize_image(img: np.ndarray) -> None:
    """
    Visualize an image.

    :param img: Image to visualize
    :return: None
    """
    cv2.imshow('visualization', img)
    cv2.waitKey(0)


def visualize_boxes(results: DetectionResults) -> None:
    """
    Visualize the boxes detected by the object detection model

    :param results: Results of a prediction, contains original image and bounding boxes
    :return: None
    """
    to_show = results.original_image.copy()
    for box in results.boxes:
        to_show = draw_bounding_box(to_show, box)
    visualize_image(to_show)


def draw_bounding_box(img: np.ndarray, box: BoundingBox) -> np.ndarray:
    """
    Draw a box on an image.

    :param img: The image
    :param box: The box to draw
    :return: The image with the box drawn on it
    """
    shape = img.shape
    top = int((box.center[1] - (box.height / 2)) * shape[1])
    bottom = int((box.center[1] + (box.height / 2)) * shape[1])
    left = int((box.center[0] - (box.width / 2)) * shape[0])
    right = int((box.center[0] + (box.width / 2)) * shape[0])
    start = (left, top)
    end = (right, bottom)

    img = cv2.rectangle(img, start, end, GREEN, 2)
    rounded_conf = round(box.conf, 2)
    text = box.label + ': ' + str(rounded_conf)
    return cv2.putText(img, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2)
