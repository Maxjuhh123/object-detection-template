"""
This module provides functionality for bounding boxes detected by the YOLO model.
"""
from dataclasses import dataclass
from typing import List
from typing import Dict
from ultralytics.yolo.engine.results import Boxes


@dataclass
class BoundingBox:
    """
    Class to represent bounding boxes.
    """
    center: (float, float)
    width: float
    height: float
    label: str
    conf: float


def from_boxes(boxes: Boxes, names: Dict[int, str]) -> List[BoundingBox]:
    """
    Get a list of BoundingBoxes from yolo results.

    :param boxes: The results from yolo
    :param names: Mapping class numbers to labels
    :return: The list of bounding boxes
    """
    coordinates = boxes.xywhn
    confidence = boxes.conf
    labels = [names[int(class_nr.item())] for class_nr in boxes.cls]
    return [BoundingBox((coordinates[i, 0].item(), coordinates[i, 1].item()), coordinates[i, 2].item(),
                        coordinates[i, 3].item(), labels[i], confidence[i].item()) for i in range(len(coordinates))]
