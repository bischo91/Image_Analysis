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


class ImageProcess:
    def image_select_by_threshold(img):
        # Take one of R/G/B pixel values and binarize by thresholding
        # img = cv2.bilateralFilter(img, 15, 75, 75)
        img = cv2.GaussianBlur(img, (3, 3), 0) # was (9,9) before
        img_o = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-5)
        # img_o = cv2.threshold(img, round(ret*1.1), 255, cv2.THRESH_BINARY)
        # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)
        kernel = np.ones((5,5), np.uint8)
        img = cv2.erode(img, kernel, iterations=3)
        # img = cv2.filter2D(img, -1, kernel)
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = (img==255) # Change img_i to True/False arrays
        return img_o

    def image_select_by_color(img, img_original_RGB, i):
        img = np.zeros(np.shape(img))
        for n in range(0, np.shape(img)[0]):
            for m in range(0, np.shape(img)[1]):
                img[n,m] = (max(img_original_RGB[n,m,:]) == img_original_RGB[n,m,i])
        img = (img==1)
        return img_o

    def detect_pixel_boxes(img):
        labeled_array, num_features = label(img)
        properties = measure.regionprops(labeled_array)
        valid_label = set()
        bbox_list = []
        for prop in properties:
            if prop.area>750:
                valid_label.add(prop.label)
                bbox_list.append(prop.bbox)
        return bbox_list


def uniformity_cal(values):
    # Calculate and return uniformity from list of pixel values
    avg = np.average(values)
    std = np.std(values)
    uniformity = 100*(1-(std/avg))
    return uniformity


def detect_calculate_pixel(img_i):
    img_o = img_i.copy()
    img_gray = cv2.cvtColor(img_i, cv2.COLOR_RGB2GRAY)
    # Seleting from image thresholding
    img_i_th = ImageProcess.image_select_by_threshold(img_gray)

    bbox_list = ImageProcess.detect_pixel_boxes(img_i_th)
    pixel_value = np.zeros(len(bbox_list))
    k=0

    for bbox in bbox_list:
        h = (bbox[2] - bbox[0])*0.25
        w = (bbox[3] - bbox[1])*0.3
        min_row = round(bbox[0]+h)
        min_col = round(bbox[1]+w)
        max_row = round(bbox[2]-h)
        max_col = round(bbox[3]-w)
        img_ind = img_gray[min_row:max_row, min_col:max_col]
        circle_detect = cv2.HoughCircles(cv2.blur(img_ind,(3,3)), \
        cv2.HOUGH_GRADIENT, 1.5, 1, param1 = 100, param2 = 0.5, minRadius = 1, maxRadius =-1)
        if circle_detect is not None:
            ret, img_ind = cv2.threshold(img_ind, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            ind_pixel_value=[]
            for row in range(min_row, max_row):
                for col in range(min_col, max_col):
                    if img_ind[row-min_row,col-min_col] == 255:
                        img_o[row, col] = [0,0,0]
                        ind_pixel_value.append(img_gray[row,col])
            pixel_value[k] = round(np.average(ind_pixel_value))
        else:
            pixel_value[k] = round(np.average(img_gray[min_row:max_row, min_col:max_col]))
            for row in range(min_row, max_row):
                for col in range(min_col, max_col):
                    img_o[row, col] = [0,0,0]
        k += 1
    plt.imshow(img_i_th)
    # fig_2, ax_tut_1 = plt.subplots(1,3, figsize=(15,15), dpi = 500) #, figsize=(15,15)) #figsize 15 15 to save dpi 500
    # fig_2.suptitle(filename)
    # ax_tut_1[0].set_title('img_i_th')
    # ax_tut_1[1].set_title('img_gray')
    # ax_tut_1[2].set_title('img_gray')
    # ax_tut_1[0].imshow(img_i_th)
    # ax_tut_1[1].imshow(img_i_th)
    # ax_tut_1[2].imshow(img_i_th)
    # ax_tut_1[0].set_axis_off()
    # ax_tut_1[1].set_axis_off()
    # ax_tut_1[2].set_axis_off()

    return img_o, pixel_value


def detect_pixel_boxes(img):
    labeled_array, num_features = label(img)
    properties = measure.regionprops(labeled_array)
    valid_label = set()
    bbox_list = []
    for prop in properties:
        if prop.area>750:
            valid_label.add(prop.label)
            bbox_list.append(prop.bbox)
    return bbox_list


def detect_calculate_pixel_2(img, i):
    # All pixels
    img_i = img[:,:,i]
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img_R = cv2.GaussianBlur(img_R, (25, 25), 0)
    img_i_th = ImageProcess.image_select_by_threshold(img_i)
    img_i_co = ImageProcess.image_select_by_color(img_i, img, i)
    img_i = np.bitwise_and(img_i_co, img_i_th)
    bbox_list = ImageProcess.detect_pixel_boxes(img_i)
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
        pixel_value[p] = round(np.average(pixel_ind_value))
        p+=1
    return img_o, pixel_value

def flatten(list_in):
    return [item for elem in list_in for item in elem]


root = tkinter.Tk()
path = tkinter.filedialog.askdirectory(parent=root, initialdir="/", title='Select Folder')
root.withdraw()
# pathstr = r"C:\Users\bisch\Desktop\Mattrix\QVGA Panel\JSR QVGA Panel\JSR QVGA #12_sprayed\after encap_photos\microscope pixels\test".replace("\\","/")
# path = os.path.abspath(pathstr)

# Read all files in the folder
allfiles = [f for f in listdir(path) if isfile(join(path,f))]
# imgfiles = [f for f in allfiles if f.upper().endswith('.PNG')]
imgfiles = [f for f in allfiles if (f.upper().endswith('.BMP') or f.upper().endswith('.PNG')) and 'Uniformity' not in f]


pixel_total = []

for i in range(0, len(imgfiles)):
    filename = imgfiles[i]
    # Load image
    img = cv2.imread(path+'/'+filename)
    img_original = cv2.imread(path+'/'+filename)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_o, pixel_value = detect_calculate_pixel(img_original)

    pixel_total.append(pixel_value.tolist())

    # fig, ax = plt.subplots(1,2, figsize=(15,15), dpi = 500) #, figsize=(15,15)) #figsize 15 15 to save dpi 500
    # fig.suptitle(filename)
    # ax[0].imshow(img_o)
    # ax[0].set_axis_off()
    #
    # ax[1].hist(pixel_value)
    # ax[1].set_xlim([0, 255])
    # ax[1].set_title('Count: ' + str(np.shape(pixel_value)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(pixel_value)))
    # fig.savefig(path + '/Uniformity_method_1' + filename.replace('bmp','png'))

plt.show()
