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

def main():
  base_model = ResNet50(weights='imagenet')
  #model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
  #model = base_model
  #base_model.layers.pop()
  #base_model.add(Dense(2, activation='softmax', name='fc8'))

  x = base_model.get_layer('avg_pool').output
  #print(x.shape)
  x = GlobalAveragePooling2D()(x)
  #x = Dense(1024, activation='relu')(x)
  predictions = Dense(1, activation='softmax')(x)
  model = Model(input=base_model.input, output=predictions)
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='rmsprop', loss='binary_crossentropy')


  pos_feat = []
  path = "/Users/waster/Downloads/All247images/"
  path2 = "/Users/waster/Downloads/bone_shadow_eliminated_JSRT_2013-04-19/"
  path3 = "/Users/waster/Desktop/767img/"
  prestr1 = "pp"
  prestr = "JPCLN"

  data = []
  labels = []
  '''
  for idx in range(1,154+1):
    rx = random.randint(0,31)
    ry = random.randint(0,31)
    im = readpng(path2+prestr, idx, rx, ry)
    B = np.asarray(im)
    B = np.repeat(B[:,:,np.newaxis],3,axis=2)
    B = np.divide(B, 3.)
    data.append(B)
    labels.append(1)
  for idx in range(1,154+1):
    rx = random.randint(0,31)
    ry = random.randint(0,31)
    im = readimg(path+prestr, idx, rx, ry)
    B = np.asarray(im)
    B = np.repeat(B[:,:,np.newaxis],3,axis=2)
    B = np.divide(B, 3.)
    data.append(B)
    labels.append(1)
  
  for idx in range(1,154+1):
    rx = random.randint(0,31)
    ry = random.randint(0,31)
    im = readpng(path3+prestr1, idx, rx, ry)
    B = np.asarray(im)
    B = np.repeat(B[:,:,np.newaxis],3,axis=2)
    B = np.divide(B, 3.)
    data.append(B)
    labels.append(1)
  '''
  
  
  #writeFeat(path2+prestr, 1, 1, model, "test.csv")
  #154
  #writeFeat(path2+prestr, 1, 10, model,  "pos_png_featf.csv")
  #writeFeat(path+prestr, 0, 10, model, "pos_featf.csv")
  #93
  prestr = "JPCNN"
  '''
  for idx in range(1,93+1):
    rx = random.randint(0,31)
    ry = random.randint(0,31)
    im = readpng(path2+prestr, idx, rx, ry)
    B = np.asarray(im)
    B = np.repeat(B[:,:,np.newaxis],3,axis=2)
    B = np.divide(B, 3.)
    #B = np.transpose()
    data.append(B)
    labels.append(0)
  for idx in range(1,93+1):
    rx = random.randint(0,31)
    ry = random.randint(0,31)
    im = readimg(path+prestr, idx, rx, ry)
    B = np.asarray(im)
    B = np.repeat(B[:,:,np.newaxis],3,axis=2)
    B = np.divide(B, 3.)
    data.append(B)
    labels.append(0)
  
  for idx in range(1, 93+1):
    rx = random.randint(0,31)
    ry = random.randint(0,31)
    im = readpng(path3+"pn", idx, rx, ry)
    B = np.asarray(im)
    B = np.repeat(B[:,:,np.newaxis],3,axis=2)
    B = np.divide(B, 3.)
    data.append(B)
    labels.append(0)
  #writeFeat(path2+prestr, 1, 10, model, "neg_png_featf.csv")
  #writeFeat(path+prestr, 0, 10, model, "neg_featf.csv")
  data = np.asarray(data)
  labels = np.asarray(labels)
  l = labels.reshape((-1, 1))
  '''
  #model = base_model
  #for i in range(0,len(data)):
    #d = data[i]
    #d = np.transpose(d,(2,0,1))
    #d = np.expand_dims(d, axis=0)
    #l = labels[i]
  info = []
  with open('pos_and_bm.csv', 'r') as f:
    reader = csv.reader(f)
    info = list(reader)
  info = [[int(j) for j in i] for i in info]
  path = "cropped_img/c"
  imglist = []
  y = []
  for j in range(0,20):
    for i in range(1,154+1):
      rx = random.randint(0,31)
      ry = random.randint(0,31)
      im = readpng(path, i, rx, ry)
      B = np.asarray(im)
      B = np.repeat(B[:,:,np.newaxis],3,axis=2)
      B = np.divide(B, 3.)
      imglist.append(B)
      y.append(info[i-1][2])

  X = np.asarray(imglist)
  y = np.asarray(y)
  y = y.reshape((-1, 1))
  model.fit(X, y, nb_epoch=10)#only train the layer i add
  for layer in model.layers[:170]:
    layer.trainable = False
  for layer in model.layers[170:]:
    layer.trainable = True

  model.compile(optimizer='rmsprop', loss='binary_crossentropy')
  model.fit(data, l, nb_epoch=10)
  model.save('fineTuneModelp.h5')

  model1 = Model(input=model.input, output=model.get_layer('avg_pool').output)
  #writeFeat(path2+prestr, 1, 93, model1, "neg_png_featf1.csv")
  #writeFeat(path+prestr, 0, 93, model1, "neg_featf1.csv")
  prestr = "JPCLN"
  writeFeat(path,1,154,model1,"bm.csv")
  #writeFeat(path3+"pp",1,154,model1,"pp_feat.csv")
  #writeFeat(path3+"pn",1,93,model1,"pn_feat.csv")
  #writeFeat(path2+prestr, 1, 154, model1,  "pos_png_featf1.csv")
  #writeFeat(path+prestr, 0, 154, model1, "pos_featf1.csv")
if __name__ == '__main__':
  main()
