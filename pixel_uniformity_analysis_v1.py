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
import skimage
import skimage.feature
import skimage.viewer
from skimage.measure import label, regionprops
from skimage import measure
import imutils
from scipy.ndimage import label
import openpyxl
from openpyxl.drawing.image import Image
import tkinter.filedialog


class ImageProcess:
    def image_select_by_threshold(img):
        # Take one of R/G/B pixel values and binarize by thresholding

        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 0)
        img = cv2.GaussianBlur(img, (3, 3), 0) # was (9,9) before

        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        img = (img==255) # Change img_i to True/False arrays
        return img

    def image_select_by_color(img, img_original_RGB, i):
        # Find which color is maximum on each pixel, and if the color matches the selected color, return True on that pixel.
        # Otherwise False on all the pixels.
        # Output is the True/False arrays
        img = np.zeros(np.shape(img))
        for n in range(0, np.shape(img)[0]):
            for m in range(0, np.shape(img)[1]):
                img[n,m] = (max(img_original_RGB[n,m,:]) == img_original_RGB[n,m,i])
        img = (img==1)

        return img

    def detect_pixel_boxes(img):
        # Find all bouding box of all features on the image
        # Return list of locations of the bouding boxes
        labeled_array, num_features = label(img)
        properties = measure.regionprops(labeled_array)
        # valid_label = set()
        bbox_list = []
        for prop in properties:
            if prop.area>700:
                # valid_label.add(prop.label)
                bbox_list.append(prop.bbox)
        return bbox_list


def uniformity_cal(values):
    # Calculate and return uniformity from list of pixel values
    avg = np.average(values)
    std = np.std(values)
    uniformity = 100*(1-(std/avg))
    return uniformity

def detect_calculate_pixel(img, i):
    # Takes image and color (R/G/B) and returns filtered image and average pixel values on the box
    # i = 0, 1, 2 for R, G, B respectively
    img_i = img[:,:,i]
    img_o = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Seleting from image thresholding
    img_i_th = ImageProcess.image_select_by_threshold(img_gray)
    # Selecting from maximum color (R/G/B)
    img_i_co = ImageProcess.image_select_by_color(img_i, img, i)
    img_i = np.bitwise_and(img_i_co, img_i_th)

    # Detect bounding boxes of displayed pixels
    bbox_list = ImageProcess.detect_pixel_boxes(img_i)
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

    return img_o, pixel_value

def flatten(list_in):
    return [item for elem in list_in for item in elem]

root = tkinter.Tk()
path = tkinter.filedialog.askdirectory(parent=root, initialdir="/", title='Select Folder')
root.withdraw()

# Read all files in the folder
allfiles = [f for f in listdir(path) if isfile(join(path,f))]
# imgfiles = [f for f in allfiles if f.upper().endswith('.PNG')]
imgfiles = [f for f in allfiles if (f.upper().endswith('.BMP') or f.upper().endswith('.PNG')) and 'Uniformity' not in f]

pixel_total_R = []
pixel_total_G = []
pixel_total_B = []
pixel_total_R_2 = []
pixel_total_G_2 = []
pixel_total_B_2 = []

for i in range(0, len(imgfiles)):
    filename = imgfiles[i]
    # Load image
    img_original = cv2.imread(path+'/'+filename)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_R, R_pixel_value = detect_calculate_pixel(img_original, 0)
    img_G, G_pixel_value = detect_calculate_pixel(img_original, 1)
    img_B, B_pixel_value = detect_calculate_pixel(img_original, 2)

    pixel_total_R.append(R_pixel_value.tolist())
    pixel_total_G.append(G_pixel_value.tolist())
    pixel_total_B.append(B_pixel_value.tolist())

    fig, ax = plt.subplots(2,3, figsize=(15,15), dpi = 500) #, figsize=(15,15)) #figsize 15 15 to save dpi 500
    fig.suptitle(filename)
    ax[0,0].imshow(img_R)
    ax[0,0].set_axis_off()
    ax[0,1].imshow(img_G)
    ax[0,1].set_axis_off()
    ax[0,2].imshow(img_B)
    ax[0,2].set_axis_off()

    ax[1,0].hist(R_pixel_value, color = 'r')
    ax[1,1].hist(G_pixel_value, color = 'g')
    ax[1,2].hist(B_pixel_value, color = 'b') # bins = 255
    ax[1,0].set_xlim([0, 255])
    ax[1,1].set_xlim([0, 255])
    ax[1,2].set_xlim([0, 255])
    ax[1,0].set_title('R, Count: ' + str(np.shape(R_pixel_value)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(R_pixel_value)))
    ax[1,1].set_title('G, Count: ' + str(np.shape(G_pixel_value)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(G_pixel_value)))
    ax[1,2].set_title('B, Count: ' + str(np.shape(B_pixel_value)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(B_pixel_value)))
    fig.savefig(path + '/Uniformity_method_1' + filename.replace('bmp','png'))


fig_all, ax_all = plt.subplots(3,1, figsize=(10,10), dpi=100)
fig_all.suptitle('RGB Histogram')
ax_all[0].hist(flatten(pixel_total_R), color='r', bins = math.ceil(255/5))
ax_all[1].hist(flatten(pixel_total_G), color='g', bins =math.ceil(255/5))
ax_all[2].hist(flatten(pixel_total_B), color='b', bins =math.ceil(255/5))
ax_all[0].set_xlim([0,255])
ax_all[1].set_xlim([0,255])
ax_all[2].set_xlim([0,255])
ax_all[0].set_title('R, Count: ' + str(len(flatten(pixel_total_R))) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(flatten(pixel_total_R))))
ax_all[1].set_title('G, Count: ' + str(len(flatten(pixel_total_G))) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(flatten(pixel_total_G))))
ax_all[2].set_title('B, Count: ' + str(len(flatten(pixel_total_B))) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(flatten(pixel_total_B))))
fig_all.savefig(path+ '/Uniformity_all.png')
