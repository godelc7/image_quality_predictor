# image_quality_predictor

This repository provides an TensorFlow implementation of the CNN architecture proposed [here](https://ieeexplore.ieee.org/document/8383698). Its purpose is to predict the subjective quality of images. 
For Training, the well known [TAMPERE IMAGE DATABASE 2013 TID2013](http://www.ponomarenko.info/tid2013.htm) is used. Compared to the original CNN from the research paper, some hyper parameter were modified to enhance the prediction accuracy leading to correlation coefficients of:

* Spearman Rank-Order Korrelationskoeffizient: 0.999
* Pearson Linear Korrelationskoeffizient: 0.9727



## Getting started

* Install Anaconconda