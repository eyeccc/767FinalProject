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
from keras.layers.core import Activation
from keras import backend as K
#sys.path.append("/u/c/h/chih-ching/h5py-master/h5py")
import h5py
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions
# path for images 
# /Users/waster/Downloads/All247images
def getFeatures(path, prestr, idx, model):
  img = readimg(path + prestr, idx)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  layer_list = [5,31,34,70,73,132,135,174]
  feat_list = []
  for i in range(0, len(layer_list)):
    feat = get_activations(model, i, x)
    feat_list.append(feat)
  return feat_list

def readimg(prestr, idx):
  filename = prestr
  name = filename + str(idx).zfill(3) + ".IMG"
  A = np.fromfile(name, dtype='int16', sep="")
    #A = A.reshape([2048, 2048])
  A = np.resize(A,[224,224])
  B = np.repeat(A[:,:,np.newaxis],3,axis=2)
    #A = A.reshape([1,3,1182,1182])
  return B

def get_activations(model, layer_idx, X_batch):
#  print(model.layers[layer_idx].get_config())
  get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output,])
  activations = get_activations([X_batch,0])
  return activations

def main():
  # if read everything at once, might run out of memory?
  #nodule_img = readimg("JPCLN", 0, 154)
  #non_nodule_img = readimg("JPCNN", 0, 93)
  model = ResNet50(weights='imagenet')

  pos_feat = []
  path = "/Users/waster/Downloads/All247images/"
  prestr = "JPCLN"
  for i in range(1,10):
    f = getFeatures(path, prestr, i, model)
    pos_feat.append(f)

  neg_feat = []
  prestr = "JPCNN"
  for i in range(1,10):
    f = getFeatures(path, prestr, i, model)
    neg_feat.append(f)

  #plt.imshow(nodule_img[0],cmap='Greys_r')
  #plt.show()
  #print (nodule_img)
  #print ("test")
  #img_path = 'elephant.jpg'
  #img = image.load_img(img_path, target_size=(224, 224))
  
  
#theano.function([model.layers[0].input], convout1.get_output(train=False), allow_input_downcast=False)
#feat = get_feature(x)
#plt.imshow(feat)
# max: four dim array
#  print(len(feat))
#  print(len(feat[0]))
#  print(len(feat[0][0]))
#  print(len(feat[0][0][0]))
#  print(len(feat[0][0][0][0]))
#preds = model.predict(x)
# print('Predicted:', decode_predictions(preds))

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
