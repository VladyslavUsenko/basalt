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
exposures = np.loadtxt(dataset_path + '/mav0/cam0/exposure.csv', usecols=[1], delimiter=',', dtype=np.int64).astype(np.float64) * 1e-9
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

inv_resp = np.arange(256, dtype=np.float64)
inv_resp[250:] = -1.0 # Use negative numbers to detect oversaturation 

irradiance = imgs[0] / exposures[0]

def opt_irradiance():
    corrected_imgs = inv_resp[imgs] * exposures[:, np.newaxis, np.newaxis]
    times = np.ones_like(corrected_imgs) * (exposures**2)[:, np.newaxis, np.newaxis]
    times[corrected_imgs < 0] == 0
    corrected_imgs[corrected_imgs < 0] == 0

    irr = np.sum(corrected_imgs, axis=0) / np.sum(times, axis=0)
    return irr

def opt_inv_resp():
    generated_imgs = irradiance[np.newaxis, :, :] * exposures[:, np.newaxis, np.newaxis]
    
    num_pixels_by_intensity = np.bincount(imgs.flat)
    sum_by_intensity = np.bincount(imgs.flat, generated_imgs.flat)
    
    new_inv_resp = inv_resp

    idx = np.nonzero(num_pixels_by_intensity > 0)
    new_inv_resp[idx] = sum_by_intensity[idx] / num_pixels_by_intensity[idx]
    new_inv_resp[250:] = -1.0
    return new_inv_resp

def print_error():
    generated_imgs = irradiance[np.newaxis, :, :] * exposures[:, np.newaxis, np.newaxis]
    generated_imgs -= inv_resp[imgs]
    generated_imgs[imgs == 255] = 0
    print(np.sum(generated_imgs**2))

print_error()
for iter in range(3):
    irradiance = opt_irradiance()
    print_error()
    inv_resp = opt_inv_resp()
    print_error()


plt.figure()
plt.plot(inv_resp)
plt.ylabel('Img Mean')
plt.xlabel('Exposure')

plt.figure()
plt.imshow(irradiance)
plt.show()


