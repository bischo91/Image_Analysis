import sys
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import axes
import matplotlib.image as mpimg
from PIL import Image
import PIL
import math
import cv2
import re
import skimage
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import skimage.feature
import skimage.viewer
import imutils
from skimage.measure import label, regionprops
from scipy.ndimage import label
from skimage import measure
import openpyxl
from openpyxl.drawing.image import Image
import tkinter.filedialog


pathstr = r"C:\Users\bisch\Desktop\Mattrix\QVGA Panel\test images\pixel".replace("\\","/")
path = os.path.abspath(pathstr)


allfiles = [f for f in listdir(path) if isfile(join(path,f))]
imgfiles = [f for f in allfiles if f.upper().endswith('.BMP')]

for i in range(0, len(imgfiles)):
    filename = imgfiles[i]
    # Load image
    img = mpimg.imread(path+'/'+filename)
    img_original = mpimg.imread(path+'/'+filename)
    # Pixel Array
    arr = np.array(img)
    img = cv2.imread(path+'/'+filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)
    img = cv2.GaussianBlur(img, (25,25),0)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    img = img==255
    pixel_value=[]
    for n in range(0, np.shape(img)[0]):
        for m in range(0, np.shape(img)[1]):
            if img[n,m] == True:
                pixel_value.append(img_gray[n,m])

    # print(pixel_value)

    img_original = mpimg.imread(path+'/'+filename)
    fig, ax = plt.subplots(1,3, figsize=(5,5), dpi=100)
    ax[0].imshow(img_original)
    ax[1].imshow(img)
    print(np.shape(img_original[:,:,0]))
    print(np.shape(img_original[:,:,1]))
    print(np.shape(img_original[:,:,2]))
    print(np.shape(img_gray))
    hist = cv2.calcHist([img_gray], [0], None, [250], [0,250])
    hist_R = cv2.calcHist([img_original[:,:,0]], [0], None, [250], [0,250])
    hist_G = cv2.calcHist([img_original[:,:,1]], [0], None, [250], [0,250])
    hist_B = cv2.calcHist([img_original[:,:,2]], [0], None, [250], [0,250])
    # plt.plot(hist)
    ax[2].plot(hist)
    ax[2].plot(hist_R)
    ax[2].plot(hist_G)
    ax[2].plot(hist_B)
    fig.suptitle(filename)
plt.show()
