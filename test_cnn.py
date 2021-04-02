#!/usr/bin/python

"""
In this python script the following is done:

    1 - Predict the quality scores for all images in the TID2013 dataset
    2 - Compute the Spearman and Person's correlation coefficients
    3 - Stores the results in a file named "predictions_tid2013.csv" to disk

Project: Hobby project

File:    test_cnn.py

Author:  Project Owner (fake-name@fake-domain.com)

Date:    2020/05/07 (created)
         2021/01/23 (last modification)
"""

from build_cnn import *


BASE_PATH = os.path.dirname(os.path.abspath(__file__))

if os.path.isfile(os.path.join(BASE_PATH, "image_quality_predictor.h5")):
    CNN_PATH = os.path.join(BASE_PATH, "image_quality_predictor.h5")
else:
    CNN_PATH = os.path.join(BASE_PATH, "pretrained_cnn", "image_quality_predictor_v1.h5")


def load_cnn_and_dataset():
    return tf.keras.models.load_model(filepath=CNN_PATH), build_dataset()


def preform_predictions_tid2013(neural_network, dataset):
    nbr_data = len(list(dataset))
    prediction_list = []
    true_list = []
    index = 0
    summary_file = open(os.path.join(BASE_PATH, "predictions_tid2013.csv"), "w+")
    summary_file.write("Image_name   Original_dmos   Predicted_dmos\n")

    for item in dataset.take(nbr_data):
        distorted_image = item["distorted_image_path"]
        contrast_normalized_img = contrast_normalization(distorted_image)
        distorted_image_name = (item["distorted_image_name"][0].numpy()).decode()
        orig_dmos = item['dmos'][0].numpy()
        prediction = neural_network.predict(contrast_normalized_img)[0][0]
        prediction_list.append(prediction)
        true_list.append(orig_dmos)
        summary_file.write(distorted_image_name + "    " + str(format(orig_dmos, '.7f')) + "    " +
                           str(format(prediction, '.7f')) + "\n")
        index += 1
        update_progressbar(index/nbr_data, "Predicting scores for the TID2013 dataset")

    srocc_value = spearman_rank_order_correlation_coefficient(prediction_list, true_list)
    plcc_value = pearson_linear_correlation_coefficient(prediction_list, true_list)

    summary_file.write("\n\nSpearman's rank-order correlation coefficient (SROCC): " + str(srocc_value))
    summary_file.write("\nPearson's linear correlation coefficient (PLCC):       " + str(plcc_value))
    print("Spearman's rank-order correlation coefficient (SROCC):", srocc_value)
    print("Pearson's linear correlation coefficient (PLCC):", plcc_value)
    summary_file.close()


def test_4_groups_of_cell_images(neural_network):
    img_file_folders = ["group_" + str(i) for i in range(1, 5)]
    logs = ""
    for folder_name in img_file_folders:
        logs += "\n======== Predictions for images in " + folder_name + " of 4 ========\n"
        folder_path = os.path.join(BASE_PATH, "3_groups_of_images", folder_name)
        all_predictions = []
        for ind in range(1, 5):
            img_name1 = "Image_" + str(ind) + ".jpg"
            img_name2 = "Image_" + str(ind) + ".bmp"
            if os.path.isfile(os.path.join(folder_path, img_name1)):
                img_path = os.path.join(folder_path, img_name1)
                img_name = img_name1
            elif os.path.isfile(os.path.join(folder_path, img_name2)):
                img_path = os.path.join(folder_path, img_name2)
                img_name = img_name2
            else:
                continue
            img = load_image_from_path(img_path)
            img = img.reshape((1,) + img.shape)
            normalized_img = contrast_normalization(img)
            prediction = neural_network.predict(normalized_img)[0][0]
            all_predictions.append(prediction)
            logs += "       " + img_name + "    Prediction: " + str(prediction) + "\n"
            print("**", sep='', end='', flush=True)
        if all_predictions[0] >= all_predictions[1] >= all_predictions[2] >= all_predictions[3]:
            logs += f"{Fore.GREEN}       WRIGHT PREDICTED ORDER\n{Style.RESET_ALL}"
        else:
            logs += f"{Fore.RED}       WRONG PREDICTED ORDER\n{Style.RESET_ALL}"
    print(f"\n\n{Fore.GREEN}Results for 3 groups of images:{Style.RESET_ALL}", end="")
    print(logs)


def test_15_groups_of_cell_images(neural_network):
    img_file_folders = ["Gruppe_" + str(i) for i in range(1, 16)]
    logs = ""
    for folder_name in img_file_folders:
        logs += "\n======== Predictions for images in " + folder_name + " of 15 ========\n"
        folder_path = os.path.join(BASE_PATH, "15_groups_of_images", folder_name)
        all_predictions = []
        for ind in range(1, 5):
            img_name1 = "Image_" + str(ind) + ".jpg"
            img_name2 = "Image_" + str(ind) + ".bmp"
            if os.path.isfile(os.path.join(folder_path, img_name1)):
                img_path = os.path.join(folder_path, img_name1)
                img_name = img_name1
            elif os.path.isfile(os.path.join(folder_path, img_name2)):
                img_path = os.path.join(folder_path, img_name2)
                img_name = img_name2
            else:
                continue
            img = load_image_from_path(img_path)
            img = img.reshape((1,) + img.shape)
            normalized_img = contrast_normalization(img)
            prediction = neural_network.predict(normalized_img)[0][0]
            all_predictions.append(prediction)
            logs += "       " + img_name + "    Prediction: " + str(prediction) + "\n"
            print("**", sep='', end='', flush=True)
        if all_predictions[0] >= all_predictions[1] >= all_predictions[2]:
            logs += f"{Fore.GREEN}       WRIGHT PREDICTED ORDER\n{Style.RESET_ALL}"
        else:
            logs += f"{Fore.RED}       WRONG PREDICTED ORDER\n{Style.RESET_ALL}"
    print(f"\n\n{Fore.GREEN}Results for 15 groups of images:{Style.RESET_ALL}", end="")
    print(logs)


if __name__ == "__main__":
    cnn, all_data = load_cnn_and_dataset()
    # test_4_groups_of_cell_images(cnn)
    # test_15_groups_of_cell_images(cnn)
    preform_predictions_tid2013(cnn, all_data)
