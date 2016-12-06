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
#from keras.layers.core import Activation
from keras import backend as K
import h5py
from keras.preprocessing import image
from keras.models import Model
from imagenet_utils import preprocess_input, decode_predictions
from skimage.feature import hog
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import SGD
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
  B = B.resize((20,20), Image.ANTIALIAS)
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
  B = A.resize((50,50), Image.ANTIALIAS) # resize to very small patch
#  B = imresize(A,[224,224])

  B = np.asarray(B)
  #B = B.flatten()
  #B = np.reshape(B,(1,20,20))
  B = np.repeat(B[:,:,np.newaxis],3,axis=2)
  B = np.divide(B, 3.)
  B = np.reshape(B,(3,50,50))
  #B = np.expand_dims(B, axis=0)
  #B = preprocess_input(B)
  return B

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
  # img patch with nodule
  path = "cropped_img/"
  prep = "cnp"
  # img patch without nodule
  pren = "n"

  data = []
  labels = []
  test_data = []
  test_y = []
  
  for idx in range(1,135+1):#total 154, 135 as training and 
    im = readpngnr(path+prep, idx)
    data.append(im)
    labels.append(1)
  for idx in range(136,154+1):#total 154, 135 as training and 
    im = readpngnr(path+prep, idx)
    test_data.append(im)
    test_y.append(1)

  for i in range(1,2+1): #total 192, 128 as training
    for idx in range(1,64+1):
      s = str(i).zfill(3)
      im = readpngnr(path+pren+s, idx)
      data.append(im)
      labels.append(0)
  for idx in range(1,64+1):
    s = str(3).zfill(3)
    im = readpngnr(path+pren+s, idx)
    test_data.append(im)
    test_y.append(0)
  dimfeat = len(data[0])
  data = np.asarray(data)
  #data = data.reshape((1,-1))
  labels = np.asarray(labels)
  test_data = np.asarray(test_data)
  #test_data = test_data.reshape((1,-1))
  test_y = np.asarray(test_y)

  l = labels.reshape((-1, 1))
  test_y = test_y.reshape((-1,1))
  
  #dimfeat = len(P1[0])
  model = Sequential()
  model.add(Convolution2D(10, 3, 3,input_shape=data.shape[1:]))
  model.add(Activation('relu'))
  #model.add(MaxPooling2D(pool_size=(2, 2)))
  #model.add(GlobalAveragePooling2D())
  #model.add(Dense(60, input_dim=dimfeat, activation='relu'))
  #model.add(Dense(1, activation='softmax'))
  model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
  #model.add(Dense(64))
  #model.add(Activation('relu'))
  #model.add(Dropout(0.5))
  model.add(Dense(1))
  model.add(Activation('softmax'))
  sgd = SGD(lr=0.1, decay=1e-6, momentum=0.5)
  model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])

  #l = labels.reshape((-1, 1))
  #model.fit(data, l, nb_epoch=100)
  #for j in range(0,100):
    #for i in range(0,len(data)):
  #print(data.shape)
      #data = data.reshape((263,1,3,20,20))
  model.fit(data.reshape((263,3,50,50)), l, nb_epoch=20, batch_size=263)
  num = 0.
  for i in range(0,len(test_y)):
    score = model.evaluate(test_data[i].reshape((1,3,50,50)), test_y[i], batch_size=263)
    print("score")
    print(score[1])
    num = num + score[1]

  print(num / len(test_y))
  '''
  writePatchFeat(path+prep,1,135,model1,"patch_train_p.csv")
  writePatchFeat(path+prep,136,154,model1,"patch_test_p.csv")
  s = "001"
  writePatchFeat(path+pren+s,1,64,model1,"patch_train_n1.csv")
  s = "002"
  writePatchFeat(path+pren+s,1,64,model1,"patch_train_n2.csv")
  s = "003"
  writePatchFeat(path+pren+s,1,64,model1,"patch_test_n.csv")
  '''

if __name__ == '__main__':
  main()
