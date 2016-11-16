import numpy as np

from matplotlib import pylab as plt
#from scipy.misc import imread, imresize, imsave
# this part is just for bypassing the authorization
# of installing new python package in lab machine
import sys
#sys.path.append("/u/c/h/chih-ching/Theano")
import theano
import tensorflow
#sys.path.append("/u/c/h/chih-ching/keras-master")
import keras
#sys.path.append("/u/c/h/chih-ching/deep-learning-models-master")
sys.path.append("/Users/waster/deep-learning-models")
from resnet50 import ResNet50
#sys.path.append("/u/c/h/chih-ching/h5py-master/h5py")
import h5py
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
# path for images 
# /Users/waster/Downloads/All247images
def readimg(prestr, minNum, maxNum):
  filename = prestr
  out = []
  for i in range(minNum, maxNum):
    name = filename + str(maxNum).zfill(3) + ".IMG"
    A = np.fromfile(name, dtype='int16', sep="")
    #A = A.reshape([2048, 2048])
    A = np.resize(A,[224,224])
    B = np.repeat(A[:,:,np.newaxis],3,axis=2)
    #A = A.reshape([1,3,1182,1182])
    out.append(B)

  return out

def main():
  # if read everything at once, might run out of memory?
  #nodule_img = readimg("JPCLN", 0, 154)
  #non_nodule_img = readimg("JPCNN", 0, 93)
  model = ResNet50(weights='imagenet')
  nodule_img = readimg("JPCLN",137,138)
  #plt.imshow(nodule_img[0],cmap='Greys_r')
  #plt.show()
  #print (nodule_img)
  #print ("test")
  #img_path = 'elephant.jpg'
  #img = image.load_img(img_path, target_size=(224, 224))
  img = nodule_img[0]
  print(img.shape)
  x = image.img_to_array(img)
  print(x.shape)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  preds = model.predict(x)
  print('Predicted:', decode_predictions(preds))

if __name__ == '__main__':
  main()

# ------------------UNUSED PART BELOW----------------------------

'''
sys.path.append("/u/c/h/chih-ching/Lasagne-master")
import lasagne

sys.path.append("/u/c/h/chih-ching/pylearn2-master")
import pylearn2

sys.path.append("/u/c/h/chih-ching/protobuf-master/")
#from google.protobuf import symbol_database as _symbol_database
#from google import protobuf
#print (protobuf.version)

sys.path.append("/u/c/h/chih-ching/caffe-theano-conversion-master/caffe2theano")
from conversion import convert
from models import *
'''

'''
# I hate dependency and being unable to have authorization to install package on lab machine!!!
import argparse
import numpy as np
import chainer
#from scipy.misc import imread, imresize, imsave
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F
from chainer.functions import caffe
import math
import os

def readimg(prestr, minNum, maxNum):
  filename = prestr
  out = []
  for i in range(minNum, maxNum):
    name = filename + str(maxNum).zfill(3) + ".IMG"
    A = np.fromfile(name, dtype='int16', sep="")
    A = A.reshape([2048, 2048])
    out.append(A)

  return out

def main():
  # if read everything at once, might run out of memory?
  #nodule_img = readimg("JPCLN", 1, 154)
  #non_nodule_img = readimg("JPCNN", 1, 93)
  print "Start loading model..."
  model = caffe.CaffeFunction('VGG_ILSVRC_19_layers.caffemodel')
  print "Finish loading model..."

if __name__ == '__main__':
  main()
'''
