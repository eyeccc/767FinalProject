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

def main():
  #base_model = ResNet50(weights='imagenet')
  #model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
  P = []
  N = []

  with open('pos_png_feat.csv', 'r') as f:
    reader = csv.reader(f)
    P = list(reader)
  with open('neg_feat_png.csv', 'r') as f:
    reader = csv.reader(f)
    N = list(reader)
  for ele in P:
    ele.pop()
  for ele in N:
    ele.pop()
  #print(len(P[0]))
  P1 = [[float(j) for j in i] for i in P]
  N1 = [[float(j) for j in i] for i in N]

  with open('pos_feat.csv', 'r') as f:
    reader = csv.reader(f)
    P = list(reader)
  with open('neg_feat.csv', 'r') as f:
    reader = csv.reader(f)
    N = list(reader)
  for ele in P:
    ele.pop()
  for ele in N:
    ele.pop()

  P2 = [[float(j) for j in i] for i in P]
  N2 = [[float(j) for j in i] for i in N]
  
  random.shuffle(P1)
  random.shuffle(N1)
  random.shuffle(P2)
  random.shuffle(N2)
  pos_num = int(len(P) / 10 * 9)
  neg_num = int(len(N) / 10 * 9)
  # Train NN with 9/10 and do cross-validation
  P_train = P1[:pos_num] + P2[:pos_num]
  P_class = [1] * pos_num * 2
  N_train = N1[:neg_num] + N2[:neg_num]
  N_class = [0] * neg_num * 2
  train = P_train + N_train
  y = P_class + N_class
  i = 0
  j = 0
  train1 = []
  y1 = []
  while (i < len(P_train) and j < len(N_train) ):
    r = random.randint(0,1)
    if (r > 0):
      train1.append(P_train[i])
      y1.append(1)
      i = i + 1
    else:
      train1.append(N_train[j])
      y1.append(0)
      j = j + 1
  #print(len(train))
  #print(len(y))
  #print(type(train[0][0]))

  clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
  clfsvm = svm.SVC()
  clfsvm.fit(train1, y1) 
  clf.fit(train1, y1)
  
  P_test = len(P) - pos_num
  N_test = len(N) - neg_num
  test = P1[-P_test:] + P2[-P_test:] + N1[-N_test:] + N2[-N_test:]
  test_y = [1]*P_test*2 + [0]*N_test*2
  result = clf.predict(test)
  r1 = 0.
  for i in range(0,len(test_y)):
    if test_y[i] == result[i]:
      r1 = r1 + 1.
  print(r1/len(test_y))
  r2 = 0.
  resultsvm = clfsvm.predict(test)
  for i in range(0,len(test_y)):
    if test_y[i] == resultsvm[i]:
      r2 = r2 + 1.
  print(r2/len(test_y))
  '''
  model = Sequential()
  model.add(Dense(32, input_shape=(2048,)))
  model.add(Dense(10, activation='softmax'))
  model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
  '''
  data = np.asarray(train1)
  labels = np.asarray(y1)
  model = Sequential()
  model.add(Dense(1, input_dim=2048, activation='softmax'))
  model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
  model.fit(data, labels, nb_epoch=30, batch_size=32)
  score = model.evaluate(test, test_y, batch_size=32)
  print(score)

if __name__ == '__main__':
  main()
