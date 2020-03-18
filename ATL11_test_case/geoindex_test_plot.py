#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:56:39 2020

@author: ben
"""

import pointCollection as pc
import numpy as np
import sys
import matplotlib.pyplot as plt

plt.figure()
for index_file in sys.argv[1:]:
    xy=pc.geoIndex().from_file(index_file).bins_as_array()
    plt.plot(xy[0], xy[1], 'o',  label=index_file)
    D=pc.geoIndex().from_file(index_file).query_xy([[xy[0][0]], [xy[1][0]]], get_data=True, fields=['x','y','h_li','pair','cycle_number'])
    D=pc.data().from_list(D)
    plt.plot(D.x, D.y, '.', label='first bin of '+index_file)

plt.axis('equal')
plt.legend()
plt.show()
