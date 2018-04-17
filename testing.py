# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:48:07 2018

@author: suzanne
"""

import numpy
import matplotlib.pyplot as plt

x=numpy.random.randn(20,20)
plt.figure()
plt.imshow(x)


x[5]=0
x[12,:]=0
plt.figure()
plt.imshow(x)
plt.colorbar()
plt.show()

plt.figure()
plt.spy(x)
