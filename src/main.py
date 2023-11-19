"""
This file is meant for loading the YOLO model and using it to detect objects in an image. It requires a trained
model (see train.py).

Instructions on how to run the file are given in the README of this project.
"""
from argparse import ArgumentParser
from argparse import Namespace
from detection.detect_objects import detect_objects
from detection.yolo_setup import load_model
from detection.model.detection import DetectionResults
from visualization.visualization import visualize_boxes
from utils.file_utils import load_image
from ultralytics import YOLO


def get_args() -> Namespace:
    """
    Get arguments from command line.

    :return: The arguments stored in a namespace
    """
    parser = ArgumentParser()
    parser.add_argument('--file_path', type=str, default='resources/example.jpg')
    parser.add_argument('--confidence_threshold', type=float, default=0.6)
    return parser.parse_args()


def load_image_and_extract_objects(file_path: str, model: YOLO, confidence_threshold: float) -> DetectionResults:
    """
    Given a file_path to an image, load it and get the objects detected.

    :param file_path: Path to the image
    :param model: Object detection model used.
    :param confidence_threshold: The threshold for confidence scores
    :return: List of bounding boxes detected.
    """
    image = load_image(file_path)
    return detect_objects(image, model, confidence_threshold)


def main() -> None:
    """
    Main method, detects objects in an image and visualize the bounding boxes.

    :return: None
    """
    args = get_args()
    file_path = args.file_path
    try:
        loaded_model = load_model()
        detected_objects = load_image_and_extract_objects(file_path, loaded_model, args.confidence_threshold)
        visualize_boxes(detected_objects)
    except FileNotFoundError:
        print('File defining weights not found.')


if __name__ == '__main__':
    main()
