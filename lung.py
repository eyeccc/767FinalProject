import numpy as np
from matplotlib import pylab as plt
# this part is just for bypassing the authorization
# of installing new python package in lab machine
import sys
sys.path.append("/u/c/h/chih-ching/Theano")

import theano
A = np.fromfile("JPCLN138.IMG", dtype='int16', sep="")

#with open('JPCLN138.IMG', 'rb') as f:
#  data = f.read(16)
#  text = data.decode('hex')
#  print text
A = A.reshape([2048, 2048])
plt.imshow(A)
plt.show()
