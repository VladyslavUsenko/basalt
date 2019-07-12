#!/usr/bin/env python3

import sys
import math
import os
import cv2

import numpy as np
from matplotlib import pyplot as plt

dataset_path = sys.argv[1]

print(dataset_path)

timestamps = np.loadtxt(dataset_path + '/mav0/cam0/data.csv', usecols=[0], delimiter=',', dtype=np.int64)
exposures = np.loadtxt(dataset_path + '/mav0/cam0/exposure.csv', usecols=[1], delimiter=',', dtype=np.int64).astype(np.float64) * 1e-6
pixel_avgs = list()

if timestamps.shape[0] != exposures.shape[0]: print("timestamps and exposures do not match")

imgs = []

# check image data.
for timestamp in timestamps:
    path = dataset_path + '/mav0/cam0/data/' + str(timestamp)
    img = cv2.imread(dataset_path + '/mav0/cam0/data/' + str(timestamp) + '.webp', cv2.IMREAD_GRAYSCALE)[:,:,0]
    imgs.append(img)
    pixel_avgs.append(np.mean(img))

imgs = np.array(imgs)
print(imgs.shape)
print(imgs.dtype)



num_pixels_by_intensity = np.bincount(imgs.flat)
print('num_pixels_by_intensity', num_pixels_by_intensity)

inv_resp = np.arange(num_pixels_by_intensity.shape[0], dtype=np.float64)
inv_resp[-1] = -1.0 # Use negative numbers to detect saturation 


def opt_irradiance():
    corrected_imgs = inv_resp[imgs] * exposures[:, np.newaxis, np.newaxis]
    times = np.ones_like(corrected_imgs) * (exposures**2)[:, np.newaxis, np.newaxis]

    times[corrected_imgs < 0] = 0
    corrected_imgs[corrected_imgs < 0] = 0

    denom = np.sum(times, axis=0)
    idx = (denom != 0)
    irr = np.sum(corrected_imgs, axis=0)
    irr[idx] /= denom[idx]
    irr[denom == 0] = -1.0
    return irr

def opt_inv_resp():
    generated_imgs = irradiance[np.newaxis, :, :] * exposures[:, np.newaxis, np.newaxis]
    
    num_pixels_by_intensity = np.bincount(imgs.flat, generated_imgs.flat >= 0)
    
    generated_imgs[generated_imgs < 0] = 0
    sum_by_intensity = np.bincount(imgs.flat, generated_imgs.flat)
    
    new_inv_resp = inv_resp

    idx = np.nonzero(num_pixels_by_intensity > 0)
    new_inv_resp[idx] = sum_by_intensity[idx] / num_pixels_by_intensity[idx]
    new_inv_resp[-1] = -1.0 # Use negative numbers to detect saturation 
    return new_inv_resp 

def print_error():
    generated_imgs = irradiance[np.newaxis, :, :] * exposures[:, np.newaxis, np.newaxis]
    generated_imgs -= inv_resp[imgs]
    generated_imgs[imgs == 255] = 0
    print(np.sum(generated_imgs**2))

for iter in range(5):
    irradiance = opt_irradiance()
    print_error()
    inv_resp = opt_inv_resp()
    print_error()



plt.figure()
plt.plot(inv_resp[:-1])
plt.ylabel('Irradiance Value')
plt.xlabel('Image Intensity')
plt.title('Inverse Responce Function')

plt.figure()
plt.imshow(irradiance)
plt.title('Irradiance Image')
plt.show()


