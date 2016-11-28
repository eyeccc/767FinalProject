import numpy as np
from matplotlib import pylab as plt
import sys
import theano
import keras
from PIL import Image
from scipy.misc import imread, imresize, imsave
import csv
import cv2
import random
# just to locate the resnet file in local machine
sys.path.append("/Users/waster/deep-learning-models")
from resnet50 import ResNet50
from keras.layers.core import Activation
from keras import backend as K
import h5py
from keras.preprocessing import image
from keras.models import Model
from imagenet_utils import preprocess_input, decode_predictions
from skimage.feature import hog
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

def readimg(prestr, idx, rx, ry): # for binary image
  filename = prestr
  name = filename + str(idx).zfill(3) + ".IMG"

  A = np.fromfile(name, dtype='int16', sep="")
  A = A.reshape([2048,2048])

  B = Image.fromarray(A)
#B = imresize(B,[256,256])
  B = B.resize((256,256), Image.ANTIALIAS)
  B = B.crop((rx,ry,rx+224,ry+224))
  #B = np.asarray(B)
  #B = np.repeat(B[:,:,np.newaxis],3,axis=2)

  return B
def readpng(prestr, idx, rx, ry):
  filename = prestr
  name = filename + str(idx).zfill(3) + ".png"
  A = Image.open(name)
  B = A.resize((256,256), Image.ANTIALIAS)
#  B = imresize(A,[224,224])
  B = B.crop((rx,ry,rx+224,ry+224))
  #B = np.asarray(B)
  #B = np.repeat(B[:,:,np.newaxis],3,axis=2)
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

def writeFeat(imgPath, imgType, maxidx, model, outname):
  arr_out = []
  for i in range(1,maxidx+1):
    for j in range(0,10):
      rx = random.randint(0,31)
      ry = random.randint(0,31)
      if (imgType == 1):
        img = readpng(imgPath, i, rx, ry)
      else:
        img = readimg(imgPath, i, rx, ry)
      B = np.asarray(img)
      B = np.repeat(B[:,:,np.newaxis],3,axis=2)
      B = np.divide(B, 3.)
      f = getFeatures(B, model)
      tp = np.asarray(f)
      tp = np.squeeze(tp)
      tp = tp.flatten()
      #tp = np.append(tp, [i])
      #(fd, hog_image) = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
      #tp2 = np.asarray(hog_image)
      #tpa = np.concatenate((tp, tp2), axis=0)
      #tpa = tp + tp2
      arr_out.append(tp)
  a = np.asarray(arr_out)
  np.savetxt(outname, a, delimiter=",")

def main():
  base_model = ResNet50(weights='imagenet')
  model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
  pos_feat = []
  path = "/Users/waster/Downloads/All247images/"
  path2 = "/Users/waster/Downloads/bone_shadow_eliminated_JSRT_2013-04-19/"
  prestr = "JPCLN"
  #writeFeat(path2+prestr, 1, 1, model, "test.csv")
  writeFeat(path2+prestr, 1, 154, model,  "pos_png_feat1.csv")
  writeFeat(path+prestr, 0, 154, model, "pos_feat1.csv")

  prestr = "JPCNN"
  writeFeat(path2+prestr, 1, 93, model, "neg_png_feat1.csv")
  writeFeat(path+prestr, 0, 93, model, "neg_feat1.csv")
  

if __name__ == '__main__':
  main()
