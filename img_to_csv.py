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
from keras.layers import Dense, GlobalAveragePooling2D
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
def readpngnr(prestr, idx):
  filename = prestr
  name = filename + str(idx).zfill(3) + ".png"
  A = Image.open(name)
  B = A.resize((224,224), Image.ANTIALIAS)
#  B = imresize(A,[224,224])

  B = np.asarray(B)
  B = np.repeat(B[:,:,np.newaxis],3,axis=2)
  B = np.divide(B, 3.)
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
  for j in range(0,20):#20
    for i in range(1,maxidx+1):
      #print(outname + str(i))
    
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
    arr_out.append(tp)
  a = np.asarray(arr_out)
  np.savetxt(outname, a, delimiter=",")
def writePatchFeat(imgPath, minidx, maxidx, model, outname):
  arr_out = []

  for i in range(minidx,maxidx+1):
    img = readpngnr(imgPath, i)

    f = getFeatures(img, model)
    tp = np.asarray(f)
    tp = np.squeeze(tp)
    tp = tp.flatten()
    arr_out.append(tp)
  a = np.asarray(arr_out)
  np.savetxt(outname, a, delimiter=",")

def main():
  base_model = ResNet50(weights='imagenet')
  x = base_model.get_layer('res2a_branch2a').output
  x = GlobalAveragePooling2D()(x)
  predictions = Dense(1, activation='softmax')(x)
  model = Model(input=base_model.input, output=predictions)
  model.compile(optimizer='rmsprop', loss='binary_crossentropy')
  # img patch with nodule
  path = "cropped_img/"
  prep = "cnp"
  # img patch without nodule
  pren = "n"

  data = []
  labels = []
  
  for idx in range(1,135+1):#total 154, 135 as training and 
    im = readpngnr(path+prep, idx)
    data.append(im)
    labels.append(1)

  for i = range(1,2+1): #total 192, 128 as training
    for idx in range(1,64+1):
      s = str(i).zfill(3)
      im = readpngnr(path+pren+s, idx)
      data.append(im)
      labels.append(0)

  data = np.asarray(data)
  labels = np.asarray(labels)

  l = labels.reshape((-1, 1))
  
  model.fit(data, l, nb_epoch=10)#only train the layer i add

  model.compile(optimizer='rmsprop', loss='binary_crossentropy')
  model.fit(data, l, nb_epoch=10)
  #model.save('fineTuneModelp.h5')

  model1 = Model(input=model.input, output=model.get_layer('res2a_branch2a').output)

  writePatchFeat(path+prep,1,135,model1,"patch_train_p.csv")
  writePatchFeat(path+prep,136,154,model1,"patch_test_p.csv")
  s = "001"
  writePatchFeat(path+pren+s,1,64,model1,"patch_train_n1.csv")
  s = "002"
  writePatchFeat(path+pren+s,1,64,model1,"patch_train_n2.csv")
  s = "003"
  writePatchFeat(path+pren+s,1,64,model1,"patch_test_n.csv")

if __name__ == '__main__':
  main()
