#!/usr/bin/python

"""
This python script is a collection of functions and variable definitions. Even though it is possible to directly
run this script, it is not intended to be used like this, but rather to be imported and used by other python scripts

Project: Hobby project

File:    utils_helpers.py

Author:  Project Owner (fake-name@fake-domain.com)

Date:    2020/05/07 (created)
         2021/01/23 (last modification)
"""

import os
import sys
import shutil
import re
import time
import typing
from matplotlib import pyplot as plt
import scipy.io
import numpy as np
import math
from ntpath import basename, split
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from keras_preprocessing.image import load_img, save_img, img_to_array, array_to_img
from numba import jit
from colorama import init, Fore, Style
ACCURACY_THRESHOLD = 0.95
LOSS_THRESHOLD = 0.1
IMG_HEIGHT = 384  # after checking all the image, I found that the smallest height is 384 and the smallest width is 480
IMG_WIDTH = 384
IMAGE_SPLIT_LOCATIONS = (0, 1620, 0, 0, 0, 1040, 580, 0, 55, 220)

init()


def show_images(images: typing.List[tf.Tensor], block_on_plot: bool = False, **kwargs):
    """
    This function plots images and shows them on the screen
    :param images: Images to be plotted on screen. Tensor-like objects
    :param block_on_plot: Whether to stop the running script upon plotting the images and resume the script
    when the plot window is closed
    :param kwargs: Additional plot arguments
    :return:
    """
    fig, axs = plt.subplots(1, len(images), figsize=(19, 10))
    for image, ax in zip(images, axs):
        assert image.get_shape().ndims in (3, 4,), "The tensor must be of dimension 3 or 4"
        if image.get_shape().ndims == 4:
            image = tf.squeeze(image)

        _ = ax.imshow(image, **kwargs)
        ax.axis("off")
    fig.tight_layout()
    plt.show(block=block_on_plot)


def filename_from_path(path):
    """
    From the given path, return only the file name at the end of the path
    :param path: path to file
    :return: file name
    """
    tree, leaf = split(path)
    return leaf or basename(path)


# @jit()
def contrast_normalization(image: tf.Tensor, new_min=0, new_max=255) -> tf.Tensor:
    original_dtype = image.dtype
    new_min = tf.constant(new_min, dtype=tf.float32)
    new_max = tf.constant(new_max, dtype=tf.float32)
    image_min = tf.cast(tf.reduce_min(image), tf.float32)
    image_max = tf.cast(tf.reduce_max(image), tf.float32)
    image = tf.cast(image, tf.float32)
    normalized_image = (new_max - new_min) / (image_max - image_min) * (image - image_min) + new_min
    return tf.cast(normalized_image, original_dtype)


class MyCallback(Callback):
    """
    This class implement a tensorflow callback, which is represent some conditions that are evaluated to check
    whether to stop the training or not
    """
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}  # {'loss': 1.5, 'mean_squared_error': 1.5}
        if logs.get('mean_squared_error') < LOSS_THRESHOLD:
            print(f"\n{Fore.GREEN}[INFO]   Accuracy and loss thresholds reached. Cancelling training!{Style.RESET_ALL}")
            self.model.stop_training = True


def print_elapsed_time(start_time, end_time, message="Time elapsed"):
    elapsed_time = end_time - start_time
    message += " Time elapsed:"
    print("")
    print(f"{Fore.GREEN}[INFO]  ", message, round(elapsed_time) // 60, "min", round(elapsed_time) % 60, "sec",
          f"{Style.RESET_ALL}")
    print("")


def error_map(reference: tf.Tensor, distorted: tf.Tensor, p: float = 0.2):
    diff = tf.cast(reference - distorted, tf.float32)
    return tf.pow(tf.abs(diff), p)


def calculate_subjective_score(features):
    distorted_img_norm_contrast = contrast_normalization(features['distorted_image_path'])
    dmos = features['dmos']
    return distorted_img_norm_contrast, dmos


def reliability_map(distorted: tf.Tensor, alpha: float) -> tf.Tensor:
    distorted = tf.cast(distorted, tf.float32)
    return 2 / (1 + tf.exp(-alpha * tf.abs(distorted))) - 1


def average_reliability_map(distorted: tf.Tensor, alpha: float) -> tf.Tensor:
    reliability = reliability_map(distorted, alpha)
    return reliability / tf.reduce_mean(reliability)


def load_image_from_path(img_filepath, size=None, color='rgb'):
    try:
        img_pil = load_img(img_filepath, target_size=size, color_mode=color)
    except Exception as e:
        print(f"{Fore.RED}[ERROR]  Could not load image", img_filepath, "in PIL format. Error code: ",
              e, f"{Style.RESET_ALL}")
        sys.exit(-2)
    img_np = img_to_array(img_pil)
    return img_np


def loss(model, x, y_true, reliability):
    y_prediction = model(x)
    return tf.reduce_mean(tf.square((y_true - y_prediction) * reliability))


def gradient(model, x, y_true, reliability):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, y_true, reliability)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)


def rescale(image: tf.Tensor, scale: float, dtype=tf.float32, **kwargs) -> tf.Tensor:
    assert image.get_shape().ndims in (3, 4), "The tensor must be of dimension 3 or 4"
    image = tf.cast(image, tf.float32)
    shape = tf.shape(image)
    shape = shape[:2] if image.get_shape().ndims == 3 else shape[1:3]
    shape = tf.cast(shape, dtype)
    shape = tf.math.ceil(shape * scale)
    rescale_size = tf.cast(shape, tf.int32)
    rescaled_image = tf.image.resize(image, size=rescale_size, **kwargs)
    return tf.cast(rescaled_image, dtype)


def calculate_error_map(features):
    distorted_img_norm_contrast = contrast_normalization(features['distorted_image_path'])
    ref_img = contrast_normalization(features['reference_image_path'])
    reliability = rescale(average_reliability_map(distorted_img_norm_contrast, 0.2), 0.25)
    obj_err = rescale(error_map(ref_img, distorted_img_norm_contrast, 0.2), 0.25)
    return distorted_img_norm_contrast, obj_err, reliability


def rescale_image(image_npy, scale_factor=1/255.):
    assert(scale_factor > 0.0)
    ret_img = image_npy * scale_factor
    return ret_img


def import_matlab_file(filepath):
    mat = scipy.io.loadmat(filepath)
    mat = {k: v for k, v in mat.items() if k[0] != '_'}
    data_pd = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
    return data_pd


def append_img_to_list(img_file_path, list2extend, target_height_and_width=None):
    img_npy = load_image_from_path(img_file_path, size=target_height_and_width)
    list2extend.append(img_npy)


def update_progressbar(progress, prefix=""):
    bar_length = round(shutil.get_terminal_size()[0] * 0.7)  # length of the progress bar --> ~60% of terminal width
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Stop...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(bar_length*progress))
    # text = "\r  {0}:  [{1}] {2}% {3}".format(prefix, "#"*block + "-"*(bar_length-block), round(progress*100), status)
    text = "{0}\r  {1}:  [{2}] {3}% {4}{5}".format(Fore.CYAN, prefix, "#" * block + "-" * (bar_length - block),
                                                   round(progress * 100), status, Style.RESET_ALL)
    sys.stdout.write(text)
    sys.stdout.flush()


@jit()
def local_contrast_normalization(img_npy, val_p=3, val_q=3, val_c=10.0):
    img_npy_list = [img_npy]
    img_npy_list = np.asarray(img_npy_list, dtype='f')
    ret_img_list = np.zeros(img_npy_list.shape)

    for k in range(img_npy_list.shape[3]):
        for i in range(val_p, img_npy_list.shape[2] - val_p):
            for j in range(val_q, img_npy_list.shape[1] - val_q):
                u = np.mean(img_npy_list[:, (j - val_q):(j + val_q), (i - val_p):(i + val_p), k])
                s = np.sqrt(np.mean(np.square(img_npy_list[:, (j - val_q):(j + val_q),
                                              (i - val_p):(i + val_p), k] - u)))
                ret_img_list[:, j, i, k] = (img_npy_list[:, j, i, k] - u) / (s + val_c)

    return ret_img_list[0]


def save_image(image_npy, dest_folder_path=os.getcwd(), dest_filename="img.bmp"):
    if image_npy is None:
        raise Exception("The provided numpy Array is None")
    if not len(image_npy.shape) in (2, 3):
        raise Exception("The Number od dimensions of the provided image is neither 2 nor 3.")
    if not os.path.isdir(dest_folder_path):
        raise Exception("The provided folder path is not existent")
    if dest_filename in os.listdir(dest_folder_path):
        raise Exception("A file with the same name is already present in this folder")
    pil_img = array_to_img(image_npy)
    save_img(os.path.join(dest_folder_path, dest_filename), pil_img)


def test_lcn(img_folder, final_size=None):
    time_load, time_lcn, time_save = 0, 0, 0
    img_list = os.listdir(img_folder)
    for img in img_list:
        if img.find("_lcn") == -1:
            img_path = os.path.join(img_folder, img)
            start_load = time.time()
            np_img = load_image_from_path(img_path, final_size)
            end_load = time.time()
            start_lcn = time.time()
            np_img = local_contrast_normalization(np_img, val_p=3, val_q=3, val_c=10.0)
            end_lcn = time.time()
            print("Max Pixel value:", np_img.max())
            start_save = time.time()
            save_image(np_img, img_folder, img[:-4] + "_lcn.bmp")
            end_save = time.time()
            time_load += end_load - start_load
            time_lcn += end_lcn - start_lcn
            time_save += end_save - start_save
    print("Timers:  load, lcn, save =", time_load, time_lcn, time_save)


def split_into_subimages(image, subimage_size):
    """
    Take an image(3D tensor) and the sizes of the sub-images [sub-Height, sub-Width]. It split the image into
    sub-images of the given sub-height and sub-width, and return all the sub-images in a 4D tensor
    of shape [nbr_sub_images, sub-Height, sub-Width, channels]
    :param image:
    :param subimage_size:
    :return: 4D tensor containing all the sub-images
    """
    image_shape = tf.shape(image)
    tile_rows = tf.reshape(image, [image_shape[0], -1, subimage_size[1], image_shape[2]])
    serial_tiles = tf.transpose(tile_rows, [1, 0, 2, 3])
    return tf.reshape(serial_tiles, [-1, subimage_size[1], subimage_size[0], image_shape[2]])


def reassemble_images(splitted_images, image_shape):
    """
    Reassembles splitted images (4D tensor of shape [nbr_sub_images, sub-Height, sub-Width, channels])
    into a single large image representing the original image before splitting
    :param splitted_images:
    :param image_shape: shape of the original image as [Heigth, width, channels]
    :return: original image
    """
    tile_width = tf.shape(splitted_images)[1]
    serialized_tiles = tf.reshape(splitted_images, [-1, image_shape[0], tile_width, image_shape[2]])
    rowwise_tiles = tf.transpose(serialized_tiles, [1, 0, 2, 3])
    return tf.reshape(rowwise_tiles, [image_shape[0], image_shape[1], image_shape[2]])


def pad_image_to_nearest_multiple(image, subimage_size, padding="CONSTANT"):
    """
    Pad a tensor, e.g. it removes borders of the tensor so that many sub-images with given size(subimage_size) can
    perfectly fit into the tensor(image)
    :param image: 3D tensor representing the image to be padded
    :param subimage_size: sizes [sub-height, sub-width] of the sub-images that must fit in the image
    :param padding: padding mode
    :return: padded image
    """
    imagesize = tf.shape(image)[0:2]
    padding_ = tf.cast(tf.math.ceil(imagesize / subimage_size), dtype=tf.int32) * subimage_size - imagesize
    return tf.pad(image, [[0, padding_[0]], [0, padding_[1]], [0, 0]], padding)


def test_image_splitting(img_folder, subimage_size):
    img_name = os.listdir(img_folder)[0]
    img_path = os.path.join(img_folder, img_name)
    img_ = load_image_from_path(img_path)
    img_pad = pad_image_to_nearest_multiple(img_, subimage_size)
    sub_images = split_into_subimages(img_pad, subimage_size)

    for i in range(sub_images.shape[0]):
        sub_img = sub_images[i]
        save_image(sub_img, img_folder, dest_filename="sub_img_" + str(i) + ".bmp")

    reassembled = reassemble_images(sub_images, tf.shape(img_pad))
    save_image(reassembled, img_folder, dest_filename="1_reassembled.bmp")


def preprocess_image(filepath, target_heigth_and_width=None, sub_image_size=(16, 16), rescale_img=True):
    img_np = load_image_from_path(filepath, size=target_heigth_and_width)
    img_lcn = local_contrast_normalization(img_np, P=3, Q=3, C=10.0)
    if rescale_img:
        # max_pixel_value = img_lcn.max()
        img_lcn_scaled = rescale_image(img_lcn)  # , scale_factor=1/max_pixel_value)
    else:
        img_lcn_scaled = img_lcn
    img_pad = pad_image_to_nearest_multiple(img_lcn_scaled, sub_image_size)
    npy_arr_of_sub_images = split_into_subimages(img_pad, sub_image_size)
    return npy_arr_of_sub_images


def preprocess_image_with_timers(filepath, target_heigth_and_width=None, sub_image_size=(16, 16), rescale_img=True):
    start_load = time.time()
    img_np = load_image_from_path(filepath, size=target_heigth_and_width)
    end_load = time.time()
    start_lcn = time.time()
    img_lcn = local_contrast_normalization(img_np, P=3, Q=3, C=10.0)
    end_lcn = time.time()
    start_rescale = time.time()
    if rescale_img:
        max_pixel_value = img_lcn.max()
        img_lcn_scaled = rescale_image(img_lcn, scale_factor=1/max_pixel_value)
    else:
        img_lcn_scaled = img_lcn
    end_rescale = time.time()
    start_pad = time.time()
    img_pad = pad_image_to_nearest_multiple(img_lcn_scaled, sub_image_size)
    end_pad = time.time()
    start_split = time.time()
    npy_arr_of_sub_images = split_into_subimages(img_pad, sub_image_size)
    end_split = time.time()
    timers = [end_load - start_load, end_lcn - start_lcn, end_rescale - start_rescale,
              end_pad - start_pad, end_split - start_split]
    return npy_arr_of_sub_images, timers


def extend_data_and_label_lists(sub_images, label, img_data_list, label_list):
    for i in range(sub_images.shape[0]):
        img_data_list.append(sub_images[i])
        label_list.append(label)


def test_string_list_2_npy_arr():
    string_list = ["Godel", "Carmel", "Kamdoum"]
    arr_string = np.asarray(string_list)
    np.save(os.path.join(os.getcwd(), "1_test_npy_string_list"), arr_string)


def test_process_images(img_folder):
    timers = [0, 0, 0, 0, 0]
    for img in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img)
        _, actual_timers = preprocess_image_with_timers(img_path)
        for i in range(5):
            timers[i] += actual_timers[i]
    print("Timers:  load, lcn, rescale, pad, split =", [timers[i] for i in range(5)])


def spearman_rank_order_correlation_coefficient(y_prediction: list, y_true: list) -> float:
    """
    This Function calculate the Spearman's rank-order correlation coefficient (SROCC)

    :param y_prediction: A list containing the predicted scores
    :param y_true: A list containing the true scores (labels)
    :return: The SROCC value
    """
    assert(len(y_prediction) == len(y_true))
    nbr_items = len(y_prediction)
    nbr_items_sq = float(nbr_items * nbr_items)
    sum_sq = 0.0
    for index in range(nbr_items):
        sum_sq += (y_prediction[index] - y_true[index]) * (y_prediction[index] - y_true[index])

    q = (6.0 * sum_sq) / (float(nbr_items)*(nbr_items_sq - 1.0))
    return 1.0 - q


def pearson_linear_correlation_coefficient(y_prediction: list, y_true: list) -> float:
    """
    This Function calculate the pearson linear correlation coefficient (PLCC)

    :param y_prediction: A list containing the predicted scores
    :param y_true: A list containing the true scores (labels)
    :return: The PLCC value
    """
    assert(len(y_prediction) == len(y_true))
    nbr_items = len(y_prediction)
    mean_prediction, mean_true = 0.0, 0.0

    for index in range(nbr_items):
        mean_prediction += y_prediction[index]
        mean_true += y_true[index]
    mean_prediction /= nbr_items
    mean_true /= nbr_items

    sum_1, sum_2, sum_3 = 0.0, 0.0, 0.0
    for index in range(nbr_items):
        sum_1 += (y_prediction[index] - mean_prediction) * (y_true[index] - mean_true)
        sum_2 += (y_prediction[index] - mean_prediction) * (y_prediction[index] - mean_prediction)
        sum_3 += (y_true[index] - mean_true) * (y_true[index] - mean_true)

    q = math.sqrt(sum_2) * math.sqrt(sum_3)

    return sum_1 / q


def predict_image_quality_cnn(image_filepath: str, cnn_model) -> float:
    """
    Predict the image quality using a neural network
    :param image_filepath: Path to image file to be predicted
    :param cnn_model: A neural network that can be used to perform predictions
    :return: The predicted value representing the image quality
    """
    img = load_image_from_path(image_filepath)
    img = img.reshape((1,) + img.shape)
    normalized_img = contrast_normalization(img)
    prediction = cnn_model.predict(normalized_img)[0][0]
    return prediction


def split_image(image_filepath: str, split_locations=IMAGE_SPLIT_LOCATIONS):
    # print(image_filepath)
    img = load_image_from_path(image_filepath)
    img_height, img_width, nbr_channels = img.shape[0], img.shape[1], img.shape[2]
    img_filename = filename_from_path(image_filepath)
    assert(os.path.isfile(image_filepath) and img_filename.startswith("capture_"))
    image_group = int(img_filename[12])

    a, b = split_locations[0], split_locations[1]
    c, d = split_locations[2], split_locations[3]
    e, f = split_locations[4], split_locations[5]
    g, h = split_locations[6], split_locations[7]
    # i, j = split_locations[8], split_locations[9]

    if image_group == 1:
        ret_img = (np.array_split(img, [g, img_width-h], axis=1))[1]
    elif image_group == 2:
        ret_img = (np.array_split(img, [e, img_width-f], axis=1))[1]
    elif image_group == 3:
        ret_img = (np.array_split(img, [a, img_width-b], axis=1))[1]
    elif image_group == 4:
        ret_img = (np.array_split(img, [c, img_width-d], axis=1))[1]
    else:
        print(f"{Fore.RED}[ERROR]  Wrong image group given. "
              f"The image group should be '1' or '2' or '3' or '4'{Style.RESET_ALL}")
        sys.exit(-2)

    return ret_img, image_group


def stitch_images_and_save(filepath_list: list, stitched_folder: str,
                           split_locations: tuple = IMAGE_SPLIT_LOCATIONS):
    print("\n\nNew Stitch:")
    if not os.path.isdir(stitched_folder):
        os.makedirs(stitched_folder)

    if re.search("Alge", filepath_list[0]):
        stitched_filename = "stitched_Alge"
    elif re.search("Zwiebelzelle", filepath_list[0]):
        stitched_filename = "stitched_Zwiebelzelle"
    else:
        stitched_filename = "stitched"

    split_images_list = []

    for filepath in filepath_list:
        print(filepath)
        split_images_list.append(split_image(filepath, split_locations))
        filename = filename_from_path(filepath)
        stitched_filename += "_" + filename.split(".")[0][8:]
    stitched_filename += ".bmp"
    stitched_filepath = os.path.join(stitched_folder, stitched_filename)

    rearranged_images = [[], [], [], []]  # [image3, image4, image2, image1]
    for tup in split_images_list:
        if tup[1] == 1:
            rearranged_images[3] = tup[0]
        elif tup[1] == 2:
            rearranged_images[2] = tup[0]
        elif tup[1] == 3:
            rearranged_images[0] = tup[0]
        elif tup[1] == 4:
            rearranged_images[1] = tup[0]
        else:
            print(f"{Fore.RED}[ERROR]  Wrong image group found. "
                  f"The image group should be '1' or '2' or '3' or '4'{Style.RESET_ALL}")
            sys.exit(-2)

    merged_img1 = np.concatenate((rearranged_images[0], rearranged_images[1]), axis=1)
    merged_img2 = np.concatenate((rearranged_images[2], rearranged_images[3]), axis=1)
    img_height1, img_height2 = merged_img1.shape[0], merged_img2.shape[0]
    i, j = split_locations[8], split_locations[9]

    split_merged_img1 = (np.array_split(merged_img1, [img_height1-i], axis=0))[0]
    split_merged_img2 = (np.array_split(merged_img2, [220], axis=0))[1]
    merged_total = np.concatenate((split_merged_img1, split_merged_img2), axis=0)
    save_img(stitched_filepath, merged_total)


if __name__ == "__main__":
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    test_folder = os.path.join(BASE_PATH, "local_contrast_norm_test")
    # print("testing lcn for pictures in folder", test_folder, "...")
    test_lcn(test_folder)
    # print("testing split images for pictures in folder", test_folder, "...")
    # test_image_splitting(test_folder, [256, 256])
    # test_string_list_2_npy_arr()
    # print("testing image processing for pictures in folder", test_folder, "...")
    # test_process_images(test_folder)
