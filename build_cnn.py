#!/usr/bin/python

"""
In this python script the following is done:

    1 - The TID2013 dataset is transformed in such a way that it can be efficiently loaded and used by tensorFlow
    2 - Tree randomly chosen images from the dataset are plotted on screen to check if the transformation of
        the dataset was successful. The script execution stops on plots, so the plot windows need to be closed
        by the user for the execution to continue
    3 - The neural network is defined, compiled and trained using the previously transformed dataset.
        After teh training, the trained neural network is saved in a .h5 file to disk.
        Afterward, the history of mean squared error for each epoch is saved in a .csv file to disk
        and plotted on screen

    In summary, two files are created by this script in the same directory:
        - The pretrained neural network: "image_quality_predictor.h5"
        - The mean squared error per epoch: "mse_per_epoch.csv"

Project: Hobby project

File:    build_cnn.py

Author:  Project Owner (fake-name@fake-email.com)

Date:    2020/05/07 (created)
         2021/01/23 (last modification)
"""

import tensorflow_datasets.public_api as tfds
from utils_helpers import *


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKSUMS_PATH = os.path.join(BASE_PATH, "checksums")
CNN_FILE_PATH = os.path.join(BASE_PATH, "image_quality_predictor.h5")

tfds.download.add_checksums_dir(CHECKSUMS_PATH)


CITATION = r"""
@article{ponomarenko2015image,
  title={Image database TID2013: Peculiarities, results and perspectives},
  author={Ponomarenko, Nikolay and Jin, Lina and Ieremeiev, Oleg and Lukin, Vladimir and Egiazarian, 
  Karen and Astola, Jaakko and Vozel, Benoit and Chehdi, Kacem and Carli, Marco and Battisti, Federica and others},
  journal={Signal Processing: Image Communication},
  volume={30},
  pages={57--77},
  year={2015},
  publisher={Elsevier}
}
"""
DESCRIPTION = """
The TID2013 contains 25 reference images and 3000 distorted images 
(25 reference images x 24 types of distortions x 5 levels of distortions). 
Reference images are obtained by cropping from Kodak Lossless True Color Image Suite. 
All images are saved in database in Bitmap format without any compression.
File names are organized in such a manner that they indicate a number of the reference image, 
then a number of distortion's type, and, finally, a number of distortion's level: "iXX_YY_Z.bmp".
"""
URL = u"http://www.ponomarenko.info/tid2013.htm"
SUPERVISED_KEYS = ("distorted_image", "mos")


class Tid2013(tfds.core.GeneratorBasedBuilder):
    name = "tid2013"
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=DESCRIPTION,
            features=tfds.features.FeaturesDict({
                    "distorted_image_path": tfds.features.Image(),
                    "reference_image_path": tfds.features.Image(),
                    "distorted_image_name": tf.string,
                    "distortion_type": tf.string,
                    "distortion_level": tf.string,
                    "dmos": tf.float32,
                }),
            supervised_keys=SUPERVISED_KEYS,
            urls=URL,
            citation=CITATION,
        )

    def _split_generators(self, manager):
        tid2013 = os.path.join(BASE_PATH, "tid2013.tar.gz")
        extracted_path = manager.download_and_extract([tid2013])
        images_path = os.path.join(extracted_path[0], "tid2013")

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"images_path": images_path, "labels": os.path.join(images_path, "mos.csv"), },
            )
        ]

    def _generate_examples(self, images_path, labels):
        with tf.io.gfile.GFile(labels) as f:
            lines = f.readlines()

        for image_id, line in enumerate(lines[1:]):
            values = line.split(", ")
            yield image_id, {
                "distorted_image_path": os.path.join(images_path, values[0]),
                "reference_image_path": os.path.join(images_path, values[1]),
                "distorted_image_name": values[0].split("/")[1],
                "distortion_type": values[2],
                "distortion_level": values[3],
                "dmos": values[4],
            }


def build_dataset():
    builder = Tid2013()
    dowld_config = tfds.download.DownloadConfig(register_checksums=True)
    builder.download_and_prepare(download_config=dowld_config)
    data_bld = builder.as_dataset(shuffle_files=True)['train']
    data_bld = data_bld.shuffle(1024).batch(1)
    return data_bld


def build_neural_network(dataset):
    for features in dataset.take(3):
        distorted_image = features["distorted_image_path"]
        reference_image = features["reference_image_path"]
        distorted_image_name = (features["distorted_image_name"][0].numpy()).decode()
        distortion_type = (features["distortion_type"][0].numpy()).decode()
        distortion_level = (features["distortion_level"][0].numpy()).decode()
        dmos = tf.round(features["dmos"][0], 2)
        print(f'Image name: {distorted_image_name}\tImage shape: {distorted_image.shape}'
              f'\tDistortion type: {distortion_type}\tDistortion level: {distortion_level}'
              f'\tImage quality score: {dmos}')
        show_images([reference_image, distorted_image], block_on_plot=True)

    training_data = dataset.map(calculate_subjective_score)

    neural_network = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(None, None, 3), batch_size=1, name='input_layer'),
        tf.keras.layers.Conv2D(48,  (3, 3),  name='conv_1', activation='relu', padding='same'),
        tf.keras.layers.Conv2D(48,  (3, 3),  name='conv_2', activation='relu', padding='same', strides=(2, 2)),
        tf.keras.layers.Conv2D(64,  (3, 3),  name='conv_3', activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64,  (3, 3),  name='conv_4', activation='relu', padding='same', strides=(2, 2)),
        tf.keras.layers.Conv2D(64,  (3, 3),  name='conv_5', activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64,  (3, 3),  name='conv_6', activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3),  name='conv_7', activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3),  name='conv_8', activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last', name='glo_avg_pl2D'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
        ],
        name='image_quality_predictor'
    )

    n_adam = tf.optimizers.Nadam(learning_rate=0.0002)

    neural_network.compile(optimizer=n_adam,
                           loss=tf.losses.MeanSquaredError(),
                           metrics=[tf.metrics.MeanSquaredError()])
    neural_network.summary()

    print("\n\n")
    print(f"{Fore.GREEN}[INFO]   Compiling and training the neural network...{Style.RESET_ALL}")
    print(f"{Fore.GREEN}[INFO]   Please grab a coffee and relax, this may take a while ;-) :-){Style.RESET_ALL}")
    print("")

    training_start = time.time()
    callbacks = MyCallback()
    history = neural_network.fit(training_data,
                                 epochs=200,
                                 callbacks=[callbacks])
    neural_network.save(CNN_FILE_PATH)
    training_end = time.time()
    print_elapsed_time(training_start, training_end, message="Training the neural network successfully terminated.")

    for sample in dataset.take(10):
        distorted_image = sample["distorted_image_path"]
        img = contrast_normalization(distorted_image)
        orig_dmos = round(sample['dmos'][0].numpy(), 2)
        prediction = round(neural_network.predict(img)[0][0], 2)
        print("    Original dmos value:", orig_dmos, "\tPredicted value:", prediction)

    mse = history.history['mean_squared_error']
    epochs = range(len(mse))
    mse_filepath = os.path.join(BASE_PATH, "mse_per_epoch.csv")
    mse_file = open(mse_filepath, "w+")
    mse_file.write("Epoch, Mean Squared Error\n")
    for e in epochs:
        mse_file.write(str(e+1) + ", " + str(mse[e]) + "\n")
    mse_file.close()

    plt.plot(epochs, mse, 'r', label='Mean squared error')
    plt.legend(loc=0)
    plt.show(block=True)


if __name__ == "__main__":
    data = build_dataset()
    build_neural_network(data)
