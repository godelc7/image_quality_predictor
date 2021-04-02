#!/usr/bin/python

"""
In aim of this script is to predict the quality of images

Project: Hobby project

File:    predict_quality.py

Author:  Project Owner (fake-name@fake-email.com)

Date:    2020/05/07 (created)
         2021/01/23 (last modification)
"""

from utils_helpers import *


def perform_predictions(path, cnn):
    if os.path.isfile(path):
        predicted = predict_image_quality_cnn(path, cnn)
        print(f"{Fore.GREEN}Score of image", path, f":{Fore.BLUE}   ", predicted, f"{Style.RESET_ALL}")
    elif os.path.isdir(path):
        for file_name in os.listdir(path):
            img_path = os.path.join(path, file_name)
            predicted = predict_image_quality_cnn(img_path, cnn)
            print(f"{Fore.GREEN}Score of image", img_path, f":{Fore.BLUE}   ", predicted, f"{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}[ERROR]   The given path doesn't exist. Please give an absolute path to "
              f"an image or to a folder containing images{Style.RESET_ALL}")
        sys.exit(-2)


if __name__ == "__main__":
    CNN_PATH = ""
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile(os.path.join(BASE_PATH, "image_quality_predictor.h5")):
        CNN_PATH = os.path.join(BASE_PATH, "image_quality_predictor.h5")
    else:
        print(f"{Fore.GREEN}[INFO]   Performing predictions using the pretrained CNN...{Style.RESET_ALL}")
        CNN_PATH = os.path.join(BASE_PATH, "pretrained_cnn", "image_quality_predictor_v1.h5")

    cnn_model = tf.keras.models.load_model(filepath=CNN_PATH)

    path_to_img = ""
    if len(sys.argv) > 1:
        path_to_img = sys.argv[1]
    else:
        print(f"{Fore.RED}[ERROR]   No path to the image where given. Please provide an "
              f"absolute path to the image(s){Style.RESET_ALL}")
        sys.exit(-3)

    perform_predictions(path_to_img, cnn_model)
