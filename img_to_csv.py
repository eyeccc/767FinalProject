import numpy as np
from matplotlib import pylab as plt
import sys
import theano
import keras
from PIL import Image
from scipy.misc import imread, imresize, imsave
import csv
import cv2
# just to locate the resnet file in local machine
sys.path.append("/Users/waster/deep-learning-models")
from resnet50 import ResNet50
from keras.layers.core import Activation
from keras import backend as K
import h5py
from keras.preprocessing import image
from keras.models import Model
from imagenet_utils import preprocess_input, decode_predictions
# path for images 
# /Users/waster/Downloads/All247images
# /Users/waster/Downloads/bone_shadow_eliminated_JSRT_2013-04-19
def getFeatures(img,  model):
#img = readimg(path + prestr, idx)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  
  feat = model.predict(x)

  return feat

def readimg(prestr, idx): # for binary image
  filename = prestr
  name = filename + str(idx).zfill(3) + ".IMG"

  A = np.fromfile(name, dtype='int16', sep="")
  A = A.reshape([2048,2048])

  B = Image.fromarray(A)
  B = imresize(B,[224,224])
  B = np.repeat(B[:,:,np.newaxis],3,axis=2)

  return B
def readpng(prestr, idx):
  filename = prestr
  name = filename + str(idx).zfill(3) + ".png"
  A = Image.open(name)
  B = imresize(A,[224,224])
  B = np.repeat(B[:,:,np.newaxis],3,axis=2)
  return B

def sift_feat(img):
  sift = cv2.xfeatures2d.SIFT_create()
  kp, d = sift.detectAndCompute(img,None)
#print(kp)
#print(d)
  return kp

def get_activations(model, layer_idx, X_batch):
  get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output,])
  activations = get_activations([X_batch,0])
  return activations

def main():
  base_model = ResNet50(weights='imagenet')
  model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
  pos_feat = []
  path = "/Users/waster/Downloads/All247images/"
  path2 = "/Users/waster/Downloads/bone_shadow_eliminated_JSRT_2013-04-19/"
  prestr = "JPCLN"
#im = readimg(path+prestr,1)
# kp = sift_feat(im)
# print(np.asarray(kp).shape)
# im = readimg(path+prestr,2)
# kp = sift_feat(im)
#print(np.asarray(kp).shape)
  p_feat_png = []
  for i in range(1,154+1):
    img = readpng(path2+prestr,i)
    f = getFeatures(img, model)
    tp = np.asarray(f)
    tp = np.squeeze(tp)
    p_feat_png.append(tp)
  a = np.asarray(p_feat_png)
  np.savetxt("pos_png_feat.csv", a, delimiter=",")

  '''  
  for i in range(1,154+1):
    img = readimg(path+prestr,i)
    f = getFeatures(img, model)
    tp = np.asarray(f)
    tp = np.squeeze(tp)
    pos_feat.append(tp)

  a = np.asarray(pos_feat)
  np.savetxt("pos_feat.csv", a, delimiter=",")
  '''
  neg_feat = []
  prestr = "JPCNN"
  '''
  for i in range(1,93+1):
    img = readimg(path+prestr, i)
    f = getFeatures(img, model)
    tp = np.asarray(f)
    tp = np.squeeze(tp)
    neg_feat.append(tp)
  a = np.asarray(neg_feat)
  np.savetxt("neg_feat.csv", a, delimiter=",")
  '''
  n_feat_png = []
  for i in range(1,93+1):
    img = readpng(path2+prestr, i)
    f = getFeatures(img, model)
    tp = np.asarray(f)
    tp = np.squeeze(tp)
    n_feat_png.append(tp)
  a = np.asarray(n_feat_png)
  np.savetxt("neg_feat_png.csv",a,delimiter=",")

if __name__ == '__main__':
  main()
