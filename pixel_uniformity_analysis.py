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
    uniformity = 100*(1-(std/avg))
    return uniformity

def detect_calculate_pixel(img, i):
    # Box shape
    img_i = img[:,:,i]
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img_R = cv2.GaussianBlur(img_R, (25, 25), 0)
    # img_R = cv2.adaptiveThreshold(img_R, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
    img_i = cv2.adaptiveThreshold(img_i, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)
    img_i = cv2.GaussianBlur(img_i, (9, 9), 0)
    ret, img_i_th = cv2.threshold(img_i, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_i_th = (img_i_th==255)
    img_i_co = np.zeros(np.shape(img_i_th))
    for n in range(0, np.shape(img_i)[0]):
        for m in range(0, np.shape(img_i)[1]):
            img_i_co[n,m] = (max(img[n,m,:]) == img[n,m,i])
    img_i_co = (img_i_co==1)
    img_i = np.bitwise_and(img_i_co, img_i_th)

    labeled_array, num_features = label(img_i)
    properties = measure.regionprops(labeled_array)
    img_i = np.zeros(np.shape(img_i), dtype = bool)
    valid_label = set()
    bbox_list = []
    for prop in properties:
        if prop.area>750:
            valid_label.add(prop.label)
            bbox_list.append(prop.bbox)
    img_i = np.in1d(labeled_array, list(valid_label)).reshape(np.shape(labeled_array))

    img_o = img.copy()
    pixel_value = np.zeros(len(bbox_list))
    k=0
    for bbox in bbox_list:
        h = (bbox[2] - bbox[0])*0.25
        w = (bbox[3] - bbox[1])*0.3
        min_row = round(bbox[0]+h)
        min_col = round(bbox[1]+w)
        max_row = round(bbox[2]-h)
        max_col = round(bbox[3]-w)
        pixel_value[k] = np.average(img_gray[min_row:max_row, min_col:max_col])
        for row in range(min_row, max_row):
            for col in range(min_col, max_col):
                img_o[row, col] = [0,0,0]
        k += 1
    return img_o, pixel_value

def detect_calculate_pixel_2(img, i):
    # All pixels
    img_i = img[:,:,i]
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img_R = cv2.GaussianBlur(img_R, (25, 25), 0)

    # img_R = cv2.adaptiveThreshold(img_R, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
    img_i = cv2.adaptiveThreshold(img_i, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)
    img_i = cv2.GaussianBlur(img_i, (9, 9), 0)
    ret, img_i_th = cv2.threshold(img_i, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # kernel = np.ones((5,5), np.uint8)
    # img_i_th = cv2.erode(img_i_th, kernel)
    img_i_th = (img_i_th==255)
    img_i_co = np.zeros(np.shape(img_i_th))
    for n in range(0, np.shape(img_i)[0]):
        for m in range(0, np.shape(img_i)[1]):
            img_i_co[n,m] = (max(img[n,m,:]) == img[n,m,i])
    img_i_co = (img_i_co==1)
    img_i = np.bitwise_and(img_i_co, img_i_th)

    labeled_array, num_features = label(img_i)
    properties = measure.regionprops(labeled_array)
    img_i = np.zeros(np.shape(img_i), dtype = bool)
    valid_label = set()
    bbox_list = []
    coords_list = []
    for prop in properties:
        if prop.area>750:
            valid_label.add(prop.label)
            bbox_list.append(prop.bbox)
            coords_list.append(prop.coords)
    img_i = np.in1d(labeled_array, list(valid_label)).reshape(np.shape(labeled_array))
    img_o = img.copy()
    pixel_value = np.zeros(len(coords_list))
    p=0
    for coords in coords_list:
        pixel_ind_value = np.zeros(len(coords))
        k=0
        for coord in coords:
            img_o[coord[0], coord[1]] = [0,0,0]
            pixel_ind_value[k] = img_gray[coord[0], coord[1]]
            k+=1
        pixel_value[p] = np.average(pixel_ind_value)
        p+=1
    return img_o, pixel_value

# pathstr = r"C:\Users\bisch\Desktop\Mattrix\QVGA Panel\JSR QVGA Panel\JSR QVGA #12_sprayed\after encap_photos\microscope pixels\test".replace("\\","/")
# pathstr = r"C:\Users\bisch\Desktop\Mattrix\QVGA Panel\JSR QVGA Panel\JSR QVGA #14_JSR sprayed\after encap microscopic images Vg -8V, Vd -17V".replace("\\","/")
# path = os.path.abspath(pathstr)

root = tkinter.Tk()
path = tkinter.filedialog.askdirectory(parent=root, initialdir="/", title='Select Folder')
root.withdraw()
# pathstr = r"C:\Users\bisch\Desktop\Mattrix\QVGA Panel\test images".replace("\\","/")
# path = os.path.abspath(pathstr)


# Read all files in the folder
# allfiles = [f for f in listdir(path) if isfile(join(path,f))]
# allfiles = [f for f in allfiles if 'cropped' not in f and 'grid' not in f]
# imgfiles = [f for f in allfiles if f.upper().endswith('.JPG')]

allfiles = [f for f in listdir(path) if isfile(join(path,f))]
# imgfiles = [f for f in allfiles if f.upper().endswith('.PNG')]
imgfiles = [f for f in allfiles if (f.upper().endswith('.BMP') or f.upper().endswith('.PNG')) and 'Uniformity' not in f]


for i in range(0, len(imgfiles)):
    filename = imgfiles[i]
    # Load image
    img = cv2.imread(path+'/'+filename)
    img_original = cv2.imread(path+'/'+filename)
    img_R, R_pixel_value = detect_calculate_pixel(img_original, 0)
    img_G, G_pixel_value = detect_calculate_pixel(img_original, 1)
    img_B, B_pixel_value = detect_calculate_pixel(img_original, 2)

    fig, ax = plt.subplots(2,3, figsize=(10,10), dpi=100) #figsize 15 15 to save dpi 500
    fig.suptitle(filename)
    ax[0,0].imshow(img_R)
    ax[0,0].set_axis_off()
    ax[0,1].imshow(img_G)
    ax[0,1].set_axis_off()
    ax[0,2].imshow(img_B)
    ax[0,2].set_axis_off()

    ax[1,0].hist(R_pixel_value, bins=255, color = 'r')
    ax[1,1].hist(G_pixel_value, bins=255, color = 'g')
    ax[1,2].hist(B_pixel_value, bins=255, color = 'b')
    # ax[1,0].set_xlim([0, 255])
    # ax[1,1].set_xlim([0, 255])
    # ax[1,2].set_xlim([0, 255])
    ax[1,0].set_title('R, Count: ' + str(np.shape(R_pixel_value)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(R_pixel_value)))
    ax[1,1].set_title('G, Count: ' + str(np.shape(G_pixel_value)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(G_pixel_value)))
    ax[1,2].set_title('B, Count: ' + str(np.shape(B_pixel_value)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(B_pixel_value)))


    img_R, R_pixel_value = detect_calculate_pixel_2(img_original, 0)
    img_G, G_pixel_value = detect_calculate_pixel_2(img_original, 1)
    img_B, B_pixel_value = detect_calculate_pixel_2(img_original, 2)

    fig2, ax2 = plt.subplots(2,3, figsize=(15,15), dpi=100) #figsize 15 15 to save dpi 500
    fig2.suptitle(filename)
    ax2[0,0].imshow(img_R)
    ax2[0,0].set_axis_off()
    ax2[0,1].imshow(img_G)
    ax2[0,1].set_axis_off()
    ax2[0,2].imshow(img_B)
    ax2[0,2].set_axis_off()
    ax2[1,0].hist(R_pixel_value, bins=255, color = 'r')
    ax2[1,1].hist(G_pixel_value, bins=255, color = 'g')
    ax2[1,2].hist(B_pixel_value, bins=255, color = 'b')
    # ax[1,0].set_xlim([0, 255])
    # ax[1,1].set_xlim([0, 255])
    # ax[1,2].set_xlim([0, 255])
    ax2[1,0].set_title('R, Count: ' + str(np.shape(R_pixel_value)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(R_pixel_value)))
    ax2[1,1].set_title('G, Count: ' + str(np.shape(G_pixel_value)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(G_pixel_value)))
    ax2[1,2].set_title('B, Count: ' + str(np.shape(B_pixel_value)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(B_pixel_value)))

    fig.savefig(path + '/Uniformity_method_1' + filename.replace('bmp','png'))
    fig2.savefig(path+ '/Uniformity_method_2' + filename.replace('bmp','png'))


plt.show()
