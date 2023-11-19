"""
This file is meant to train your object detection model.

Instructions on how to run the file are given in the README.
"""
from argparse import ArgumentParser
from argparse import Namespace

from detection.yolo_setup import train_model


def get_args() -> Namespace:
    """
    Get arguments from command line.

    :return: The arguments stored in a namespace
    """
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    return parser.parse_args()


def main() -> None:
    """
    Main method, trains the model.

    :return: None
    """
    args = get_args()
    train_model(args.epochs)


if __name__ == '__main__':
    main()
