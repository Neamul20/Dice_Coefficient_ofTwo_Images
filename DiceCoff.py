#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:40:25 2020

@author: neamul
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#import cv2
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import nibabel as nib
import os

def dice_coef(img, img2):
    if img.shape != img2.shape:
        raise ValueError("Shape mismatch: img and img2 must have to be of the same shape.")
    else:
        intersection = np.logical_and(img, img2)
        value = (2. * intersection.sum())  / (img.sum() + img2.sum())
    return value 
#===========================img1======================================
filename = os.path.join("/home/neamul/thesis_git/Dice_coff/zlst/new/Unetmask_Zstl_Nonzero.nii")
#filename = os.path.join("/home/neamul/thesis_git/Dice_coff/wast/UNETmaskWastNonzero.nii")
img1= nib.load(filename)
DataCT = img1.get_fdata(dtype='float32')
img1 = img1.get_fdata().squeeze()
plt.imshow (img1[50])
img1= np.asarray(img1).astype(np.bool)
#==================================================================

#===========================img2======================================
#filename = os.path.join("/home/neamul/thesis_git/testresult/combination1/wast14/wast_pred_mask.nii")
filename = os.path.join("/home/neamul/thesis_git/testresult/New/Combination4/Zlst/zlst_pred4Mask.nii")
img2= nib.load(filename)
DataCT = img2.get_fdata(dtype='float32')
img2 = img2.get_fdata().squeeze()
img2= np.asarray(img2).astype(np.bool)
#==================================================================
value = dice_coef(img1, img2)
print (value)
