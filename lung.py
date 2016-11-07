import numpy as np
from matplotlib import pylab as plt

A = np.fromfile("JPCLN138.IMG", dtype='int16', sep="")

#with open('JPCLN138.IMG', 'rb') as f:
#  data = f.read(16)
#  text = data.decode('hex')
#  print text
A = A.reshape([2048, 2048])
plt.imshow(A)
plt.show()
