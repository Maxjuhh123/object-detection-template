"""
This module provides utility for training and loading a yolo model.
"""
from ultralytics import YOLO


TRAINING_CONFIG_FILE = 'resources/training-config.yaml'
TRAINED_WEIGHTS_FILE = 'resources/best.pt'


def train_model(epochs: int) -> YOLO:
    """
    Train the YOLO model on your dataset.

    :param epochs: The amount of epochs for training
    :return: The trained model
    """
    model = YOLO('yolov8n.pt')
    model.train(data=TRAINING_CONFIG_FILE, epochs=epochs)
    return model


def load_model() -> YOLO:
    """
    Try to load the trained YOLO model from a file defining weights.

    :return: The YOLO model
    :exception FileNotFoundError: If weights file not found
    """
    return YOLO(TRAINED_WEIGHTS_FILE)
