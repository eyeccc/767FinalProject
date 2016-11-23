import numpy as np
from matplotlib import pylab as plt
import sys
import theano
import keras
from PIL import Image
from scipy.misc import imread, imresize, imsave
#import cv2
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
def getFeatures(path, prestr, idx, model):
  img = readimg(path + prestr, idx)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
#i = 174 # best layer for feature extraction
#i = 5
  # option 1, extract feature from layer 174
 #feat = get_activations(model, i, x)
  
  # option 2, extract feature from final result
  feat = model.predict(x)
#test = np.asarray(feat)
#print(test.shape)
  return feat
  # option 3, use all feature maps
  '''
  layer_list = [5,31,34,70,73,132,135,174]
  feat_list = []
  for i in range(0, len(layer_list)):
    feat = get_activations(model, i, x)
    feat_list.append(feat)
  return feat_list
  '''

def readimg(prestr, idx):
  filename = prestr
  name = filename + str(idx).zfill(3) + ".IMG"

  A = np.fromfile(name, dtype='int16', sep="")
  A = A.reshape([2048,2048])

  B = Image.fromarray(A)
  
# B = imresize(B,[2048,2048])
  B = imresize(B,[224,224])
  B = np.repeat(B[:,:,np.newaxis],3,axis=2)
#plt.imshow(B)
#  plt.plot([1634],[692],"r*")
#  plt.show()
  return B

def get_activations(model, layer_idx, X_batch):
  get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer_idx].output,])
  activations = get_activations([X_batch,0])
  return activations

def main():
  # if read everything at once, might run out of memory?
  #nodule_img = readimg("JPCLN", 0, 154)
  #non_nodule_img = readimg("JPCNN", 0, 93)
  base_model = ResNet50(weights='imagenet')
  model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
  pos_feat = []
  path = "/Users/waster/Downloads/All247images/"
  prestr = "JPCLN"
  test = getFeatures(path, prestr, 9, model)
  '''
  for i in range(1,10):
    f = getFeatures(path, prestr, i, model)
    pos_feat.append(f)'''
  tp = np.asarray(test)
  print(tp.shape)
#t = np.transpose(tp,(2,3,4,0,1))
# print(t.shape)
# t = np.squeeze(t)
# print(t.shape)
# plt.imshow(t)
# plt.show()
  
'''
  neg_feat = []
  prestr = "JPCNN"
  for i in range(1,10):
    f = getFeatures(path, prestr, i, model)
    neg_feat.append(f)
'''
#preds = model.predict(x)
# print('Predicted:', decode_predictions(preds))

if __name__ == '__main__':
  main()
