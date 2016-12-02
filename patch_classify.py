import numpy as np
from matplotlib import pylab as plt
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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

def main():
  #base_model = ResNet50(weights='imagenet')
  #model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
  P = []
  N = []
  filepath = '/Users/waster/767csv/'
  with open('patchfeatp.csv', 'r') as f:
    reader = csv.reader(f)
    P = list(reader)
  with open('patchfeatn.csv', 'r') as f:
    reader = csv.reader(f)
    N = list(reader)

  P1 = [[float(j) for j in i] for i in P]
  N1 = [[float(j) for j in i] for i in N]
  
  random.shuffle(P1)
  random.shuffle(N1)

  loo = LeaveOneOut()
  X = N1 + P1
  y = [0]*len(N1) + [1]*len(P1)
  X = np.asarray(X)
  y = np.asarray(y)
  
  scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
  print("accuracy")
  print(scores)
  print(scores.mean())
  scores = cross_val_score(LogisticRegression(), X, y, scoring='roc_auc', cv=10)
  print("roc_auc")
  print(scores)
  print(scores.mean())
  
  '''
  clfsvm = svm.SVC()
  clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
  
  predicted = cross_val_predict(clf, X, y, cv=10)
  result = metrics.accuracy_score(y, predicted) 
  print(result)
  predicted = cross_val_predict(clfsvm, X, y, cv=10)
  result = metrics.accuracy_score(y, predicted) 
  print(result)
  
  dimfeat = len(P1[0])
  model = Sequential()
  model.add(Dense(60, input_dim=dimfeat, activation='relu'))
  model.add(Dense(1, activation='softmax'))
  model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

  l = y.reshape((-1, 1))
  #model.fit(data, l, nb_epoch=100)
  model.fit(X, l, validation_split=0.1, nb_epoch=150, batch_size=10)
  '''
  
if __name__ == '__main__':
  main()
