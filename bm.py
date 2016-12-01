import numpy as np
import sys
from PIL import Image
from scipy.misc import imread, imresize, imsave
import csv
import cv2
import random
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def readpng(prestr, idx, rx, ry):
  filename = prestr
  name = filename + str(idx).zfill(3) + ".png"
  A = Image.open(name)
  #B = A.resize((256,256), Image.ANTIALIAS)
#  B = imresize(A,[224,224])
  B = B.crop((rx,ry,rx+224,ry+224))
  B = np.asarray(B)
  #B = np.repeat(B[:,:,np.newaxis],3,axis=2)
  return B
  
def create_smaller():
	# create model
	model = Sequential()
	model.add(Dense(30, input_dim=(224, 224), init='normal', activation='relu'))
	model.add(Dense(1, init='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def main():
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
	  imglist.append(im)
	  y.append(info[i][2])

  X = np.asarray(imglist)
  y = np.asarray(y)
  l = y.reshape((-1, 1))
  
  seed = 7
  numpy.random.seed(seed)
  estimators = []
  estimators.append(('standardize', StandardScaler()))
  estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, nb_epoch=100, batch_size=5, verbose=0)))
  pipeline = Pipeline(estimators)
  kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
  results = cross_val_score(pipeline, X, l, cv=kfold)
  print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
  '''
  dimfeat = len(P1[0])
  model = Sequential()
  model.add(Dense(60, input_dim=(224,224,) activation='relu'))
  model.add(Dense(1, activation='softmax'))
  model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

  
  model.fit(X, l, validation_split=0.1, nb_epoch=150, batch_size=10)
  '''
  
if __name__ == '__main__':
  main()
