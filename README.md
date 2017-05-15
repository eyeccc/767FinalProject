# Lung Nodule Detection in X-ray Image
CS767 Final Project

## Dataset for Lung Nodule
JSRT: http://www.jsrt.or.jp/jsrt-db/eng.php

## Aproach
1. Using simple image processing methods (HoG)
2. Using deep learning with pre-trained models

## Dependency (Deep Learning Python Library)

1. http://deeplearning.net/software/theano/
2. https://github.com/fchollet/keras
3. https://github.com/fchollet/deep-learning-models

## Pre-trained Model
ResNet: http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006

(Get from: https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py)

### Test layers for deep learning method
1. last average relu layer
2. Only the first 2 layers

## Issue I faced
1. How to get hidden layer feature: https://github.com/fchollet/keras/issues/3166
   1. http://blog.christianperone.com/tag/feature-extraction/
2. Visualize feature extracted from hidden layer: https://github.com/fchollet/keras/issues/431#issuecomment-124175958
3. Object localization using Keras: https://blog.heuritech.com/2016/04/26/pre-trained-convnets-and-object-localisation-in-keras/

## Preprocessing
1. Image Registration (rotation to make every figure be at the same angle): https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
2. Remove bone shadow: https://www.mit.bme.hu/node/9150
3. data augmentation: Resize each image from 2048x2048 to 256x256 then random cropping to 224x224 to fit the model
   1. http://machinelearningmastery.com/image-augmentation-deep-learning-keras/
   2. http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
   3. https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
4. contrast stretching: https://arxiv.org/pdf/1603.08486.pdf
5. http://www.isi.uu.nl/Gallery/noduledetection/


## More Info about Lung Nodule
The doctor will look at the X-ray to evaluate the size and shape of the nodule, its location, and its general appearance. Single pulmonary nodules seen on chest x-rays are generally at least 8 to 10 millimeters in diameter. If they are smaller than that, they are unlikely to be visible on a chest X-ray. The larger the nodule is, and the more irregularly shaped it is, the more likely it is to be cancerous. Those located in the upper portions of the lung are also more likely to be cancerous.

## Method or Algo
1. classify nodule vs non-nodule
   1. using cnn
   2. slice picture to 20 mm suqare, do something on those squares
2. detect position, same as 1.
3. classify benign vs malicious, given position, crop the nodule area, do classification on sub-image

## Source
1. https://www.cs.tau.ac.il/~wolf/papers/SPIE15chest.pdf
2. http://my.clevelandclinic.org/health/diseases_conditions/hic_Pulmonary_Nodules
3. http://cis-linux1.temple.edu/~wangp/5603-AI/Project/2016S/Transfer_ImageNet.pdf

## Slides
https://docs.google.com/presentation/d/1sBLeiPUyPBZl__jUt5zuR4w3zGU0fA9u9FBkoZ-kAys/edit#slide=id.gcb9a0b074_1_0

