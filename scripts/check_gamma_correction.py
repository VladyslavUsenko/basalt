#!/usr/bin/env python3

import sys
import math
import os
import webp

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

dataset_path = sys.argv[1]

print(dataset_path)

timestamps = np.loadtxt(dataset_path + '/mav0/cam0/data.csv', usecols=[0], delimiter=',', dtype=np.int64)
exposures = np.loadtxt(dataset_path + '/mav0/cam0/exposure.csv', usecols=[1], delimiter=',', dtype=np.int64)
pixel_avgs = list()


# check image data.
img_extensions = ['.png', '.jpg', '.webp']
for timestamp in timestamps:
    path = dataset_path + '/mav0/cam0/data/' + str(timestamp)
    img = webp.imread(dataset_path + '/mav0/cam0/data/' + str(timestamp) + '.webp')
    pixel_avgs.append(np.mean(img))

plt.plot(exposures, pixel_avgs)
plt.ylabel('Img Mean')
plt.xlabel('Exposure')
plt.show()


