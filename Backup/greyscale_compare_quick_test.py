import sys
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import axes
import matplotlib.image as mpimg
import math
import cv2
import re
import skimage
import skimage.feature
import skimage.viewer
import imutils
from skimage.measure import label, regionprops
from scipy.ndimage import label
from skimage import measure
import tkinter.filedialog



root = tkinter.Tk()
path = tkinter.filedialog.askdirectory(parent=root, initialdir="/", title='Select Folder')
root.withdraw()
allfiles = [f for f in listdir(path) if isfile(join(path,f))]
# imgfiles = [f for f in allfiles  in f]
imgfiles = allfiles
for i in range(0, len(imgfiles)):
    filename = imgfiles[i]
    # Load image
    # img = cv2.imread(path+'/'+filename)
    img_original = cv2.imread(path+'/'+filename)
    img_grey = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1,2, figsize=(15,15), dpi = 500) #, figsize=(15,15)) #figsize 15 15 to save dpi 500
    ax[0].imshow(img_original)
    ax[0].set_axis_off()
    ax[0].set_title('Original')
    ax[1].imshow(img_grey)
    ax[1].set_axis_off()
    ax[1].set_title('Grey')

plt.show()
