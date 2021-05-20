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
import openpyxl
from openpyxl.drawing.image import Image
import tkinter.filedialog


# Pixel-to-pixel
# Where are low values at? because pixel values around the edges are lower,
# and G has relatively high value which creates larger difference between center and edge of pixel

def uniformity_cal(values):
    avg = np.average(values)
    std = np.std(values)
    uniformity = 100*(1-std/avg)
    return uniformity

def detect_and_binarize_pixels_R(img):
    img_R = img[:,:,0]
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img_R = cv2.GaussianBlur(img_R, (25, 25), 0)

    # img_R = cv2.adaptiveThreshold(img_R, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
    img_R = cv2.adaptiveThreshold(img_R, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)
    img_R = cv2.GaussianBlur(img_R, (9, 9), 0)

    ret, img_R_th = cv2.threshold(img_R, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_R_th = (img_R_th==255)
    img_R_co = np.zeros(np.shape(img_R_th))
    for n in range(0, np.shape(img_R)[0]):
        for m in range(0, np.shape(img_R)[1]):
            img_R_co[n,m] = (max(img[n,m,:]) == img[n,m,0])
    img_R_co = (img_R_co==1)
    img_R = np.bitwise_and(img_R_co, img_R_th)

    labeled_array, num_features = label(img_R)
    properties = measure.regionprops(labeled_array)
    bw = np.zeros(np.shape(img), dtype = bool)
    valid_label = set()
    for prop in properties:
        if prop.area>500:
            valid_label.add(prop.label)
    img_R = np.in1d(labeled_array, list(valid_label)).reshape(np.shape(labeled_array))

    return img_R



pathstr = r"C:\Users\bisch\Desktop\Mattrix\QVGA Panel\JSR QVGA Panel\JSR QVGA #12_sprayed\after encap_photos\microscope pixels\test".replace("\\","/")
# pathstr = r"C:\Users\bisch\Desktop\Mattrix\image_process\RGB\Microscopic".replace("\\","/")

path = os.path.abspath(pathstr)


allfiles = [f for f in listdir(path) if isfile(join(path,f))]
# imgfiles = [f for f in allfiles if f.upper().endswith('.PNG')]
imgfiles = [f for f in allfiles if f.upper().endswith('.BMP')]


for i in range(0, len(imgfiles)):
    filename = imgfiles[i]
    # Load image
    # img = mpimg.imread(path+'/'+filename)
    img = cv2.imread(path+'/'+filename)
    # img_original = mpimg.imread(path+'/'+filename)
    img_original = cv2.imread(path+'/'+filename)
    # Pixel Array
    arr = np.array(img)
    img = cv2.imread(path+'/'+filename)
    print(np.shape(img))
    img_R = img[:,:,0]
    print(np.shape(img_R))
    img_G = img[:,:,1]
    img_B = img[:,:,2]

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img_B
    # img = cv2.GaussianBlur(img, (9,9),0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
    # img = cv2.GaussianBlur(img, (25,25),0)
    img = cv2.blur(img, (9,9))
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    img = img==255
    pixel_value = []
    R_pixel_value = []
    G_pixel_value = []
    B_pixel_value = []


    labeled_array, num_features = label(img)
    properties = measure.regionprops(labeled_array)
    bw = np.zeros(np.shape(img), dtype = bool)
    valid_label = set()
    for prop in properties:
        if prop.area>250:
            valid_label.add(prop.label)
    current_bw = np.in1d(labeled_array, list(valid_label)).reshape(np.shape(labeled_array))
    img = current_bw



    for n in range(0, np.shape(img)[0]):
        for m in range(0, np.shape(img)[1]):
            if img[n,m] == True:
                pixel_value.append(img_gray[n,m])
                max_value = max(img_original[n,m])
                if img_original[n,m][0] == max_value:
                    R_pixel_value.append(img_gray[n,m])
                elif img_original[n,m][1] == max_value:
                    G_pixel_value.append(img_gray[n,m])
                elif img_original[n,m][2] == max_value:
                    B_pixel_value.append(img_gray[n,m])



    R_pixel_value = np.array(R_pixel_value)
    G_pixel_value = np.array(G_pixel_value)
    B_pixel_value = np.array(B_pixel_value)
    pixel_value = np.array(pixel_value)

    norm_R_pixel_value = R_pixel_value/np.linalg.norm(R_pixel_value)
    norm_G_pixel_value = G_pixel_value/np.linalg.norm(G_pixel_value)
    norm_B_pixel_value = B_pixel_value/np.linalg.norm(B_pixel_value)
    norm_pixel_value = []
    norm_pixel_all = []
    norm_pixel_value.append(norm_R_pixel_value)
    norm_pixel_value.append(norm_G_pixel_value)
    norm_pixel_value.append(norm_B_pixel_value)
    norm_pixel_all = np.append(norm_R_pixel_value, norm_pixel_all)
    norm_pixel_all = np.append(norm_G_pixel_value, norm_pixel_all)
    norm_pixel_all = np.append(norm_B_pixel_value, norm_pixel_all)


    img_original = cv2.imread(path+'/'+filename)

    fig, ax = plt.subplots(2,3, figsize=(5,5), dpi=100) #figsize 15 15 to save dpi 500
    fig.suptitle(filename)
    ax[0,0].imshow(img_original)
    ax[0,0].set_axis_off()
    ax[0,1].imshow(img)
    ax[0,1].set_axis_off()
    # ax[0,2].hist(pixel_value, bins=255)
    # ax[0,2].hist(norm_pixel_value, bins=255)
    # ax[0,2].set_title('Gray, Total Count: ' + str(np.shape(pixel_value)[0]))
    # counter = np.shape(norm_pixel_value[0])[0] + np.shape(norm_pixel_value[1])[0] + np.shape(norm_pixel_value[2])[0]
    # print(counter)
    ax[0,2].imshow(detect_and_binarize_pixels_R(img_original))
    ax[0,2].set_axis_off()
    # ax[0,2].set_title('Gray, Total Count: ' + str(counter) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(norm_pixel_all)))

    # fig_RGB, ax_RGB = plt.subplots(1,3)

    ax[1,0].hist(R_pixel_value, bins=255, color = 'r')
    ax[1,1].hist(G_pixel_value, bins=255, color = 'g')
    ax[1,2].hist(B_pixel_value, bins=255, color = 'b')
    ax[1,0].set_xlim([0, 255])
    ax[1,1].set_xlim([0, 255])
    ax[1,2].set_xlim([0, 255])
    ax[1,0].set_title('Red, Count: ' + str(np.shape(R_pixel_value)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(R_pixel_value)))
    ax[1,1].set_title('Green, Count: ' + str(np.shape(G_pixel_value)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(G_pixel_value)))
    ax[1,2].set_title('Blue, Count: ' + str(np.shape(B_pixel_value)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(B_pixel_value)))

    # fig.savefig(pathstr+'\Figure ' + filename.replace('bmp','png'))

    # hist = cv2.calcHist([img_gray], [0], None, [250], [30,150])
    # hist_R = cv2.calcHist([img_original[:,:,0]], [0], None, [250], [0,250])
    # hist_G = cv2.calcHist([img_original[:,:,1]], [0], None, [250], [0,250])
    # hist_B = cv2.calcHist([img_original[:,:,2]], [0], None, [250], [0,250])
    # plt.plot(hist)

    # ax[2].plot(hist_R)
    # ax[2].plot(hist_G)
    # ax[2].plot(hist_B)



plt.show()
