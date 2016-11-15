import numpy as np

#from matplotlib import pylab as plt

# this part is just for bypassing the authorization
# of installing new python package in lab machine
import sys
sys.path.append("/u/c/h/chih-ching/Theano")
import theano
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
sys.path.append("/u/c/h/chih-ching/keras-master")
import keras
sys.path.append("/u/c/h/chih-ching/deep-learning-models-master")
from resnet50 import ResNet50
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
  print ("test")

if __name__ == '__main__':
  main()

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
