"""
This module provides utility functionality for files.
"""
import numpy as np
import cv2


def load_image(file_path: str) -> np.ndarray:
    """
    Load an image given a file path.

    :param file_path: The path to the image
    :return: The loaded image as a numpy array
    """
    return cv2.imread(file_path)
