import numpy as np
from matplotlib import pylab as plt

A = np.fromfile("JPCLN138.img", dtype='int16', sep="")
A = A.reshape([2048, 2048])
plt.imshow(A)