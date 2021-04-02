# image_quality_predictor

This repository provides an TensorFlow implementation of the CNN architecture proposed [here](https://ieeexplore.ieee.org/document/8383698). Its purpose is to predict the subjective quality of images. 
For Training, the well known [TAMPERE IMAGE DATABASE 2013](http://www.ponomarenko.info/tid2013.htm) is used. Compared to the original CNN from the research paper, some hyper parameters were modified to enhance the prediction accuracy leading to correlation coefficients of:

* Spearman Rank-Order Korrelationskoeffizient: **0.999**
* Pearson Linear Korrelationskoeffizient: **0.9727**



## Getting started

1. Install [Anaconconda3](https://www.anaconda.com/) (this code were tested only with python 3)
2. Install tensorflow or tensorflow-GPU and activate the new created environment
    ```bash
    conda create -n tf tensorflow
    conda activate tf
    ```
	or 
    ```bash
    conda create -n tf-gpu tensorflow-gpu
    conda activate tf-gpu
    ```
3. Install additional python modules:
    ```bash
    conda install tensorflow-datasets pandas colorama matplotlib pyyaml numba 
    conda install -c conda-forge imutils hickle cudatoolkit
    ```
4. Training:
    ```bash
    python build_cnn.py
    ```
5. Test CNN:
    ```bash
    python test_cnn.py
    ```
	Note: if there is no CNN file in the actual folder ("image_quality_predictor.h5"), this script will use the pretrained network located in "./pretrained_cnn/image_quality_predictor_v1.h5"


6. Predict the image quality for:
    ```bash
    python predict_quality.py <ABSOLUTE_PATH_TO_IMAGE_OR_FOLDER_CONTAINING_IMAGES>
    ```