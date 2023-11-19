"""
This module contains functionality for representing object detection results for an image.
"""
from dataclasses import dataclass
from typing import List
from detection.model.bounding_box import BoundingBox

import numpy as np


@dataclass
class DetectionResults:
    """
    A class for storing detection results for a single image.
    """
    boxes: List[BoundingBox]
    original_image: np.ndarray
