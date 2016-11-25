# 767FinalProject
CS767 Final Project

## Dataset for Lung Nodule
1. X-ray: http://www.jsrt.or.jp/jsrt-db/eng.php
2. CT: http://www.via.cornell.edu/databases/crpf.html

## Dependency (Deep Learning Python Library)

1. http://deeplearning.net/tutorial/lenet.html
2. http://deeplearning.net/software/theano/
3. https://github.com/fchollet/keras
4. https://github.com/fchollet/deep-learning-models

#### might not be used
1. https://code.google.com/archive/p/neurolab/
2. https://www.tensorflow.org/
3. https://github.com/kitofans/caffe-theano-conversion

## Model I use
ResNet: http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006

(Get from: https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py)

### layers
pool1, res2c, res3d, res4f, and pool5

## Issue I faced
1. How to get hidden layer feature: https://github.com/fchollet/keras/issues/3166
   1. http://blog.christianperone.com/tag/feature-extraction/
2. Visualize feature extracted from hidden layer: https://github.com/fchollet/keras/issues/431#issuecomment-124175958
3. Object localization using Keras: https://blog.heuritech.com/2016/04/26/pre-trained-convnets-and-object-localisation-in-keras/

## Preprocessing
1. Image Registration (rotation to make every figure be at the same angle): https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
2. Remove bone shadow: https://www.mit.bme.hu/node/9150
3. data augmentation: 
   1. http://machinelearningmastery.com/image-augmentation-deep-learning-keras/
   2. http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html


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

## Disclaimer
I used my friends laptop for part of the implementation and forgot to change git config file. Thus, it will show other contributor, but it's actually me...
