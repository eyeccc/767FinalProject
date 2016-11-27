import numpy as np
from matplotlib import pylab as plt
import sys
from PIL import Image
from scipy.misc import imread, imresize, imsave
import csv
import cv2
import random
from sklearn.neural_network import MLPClassifier

def main():
  #base_model = ResNet50(weights='imagenet')
  #model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
  P = []
  N = []

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

  P1 = [[float(j) for j in i] for i in P]
  N1 = [[float(j) for j in i] for i in N]
  #random.shuffle(P)
  #random.shuffle(N)
  pos_num = int(len(P) / 10 * 9)
  neg_num = int(len(N) / 10 * 9)
  # Train NN with 9/10 and do cross-validation
  P_train = P1[:pos_num]
  P_class = [1] * pos_num
  N_train = N1[:neg_num]
  N_class = [0] * neg_num
  train = P_train + N_train
  y = P_class + N_class
  #print(len(train))
  #print(len(y))
  #print(type(train[0][0]))

  clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
  
  clf.fit(train, y)
  
  P_test = len(P) - pos_num
  N_test = len(N) - neg_num
  test = P1[-P_test:] + N1[-N_test:]
  result = clf.predict(test)
  print(result)

if __name__ == '__main__':
  main()
