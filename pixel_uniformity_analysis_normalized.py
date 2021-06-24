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
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)
        img = cv2.GaussianBlur(img, (3, 3), 0) # was (9,9) before
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = (img==255) # Change img_i to True/False arrays
        return img

    def image_select_by_color(img, img_original_RGB, i):
        img = np.zeros(np.shape(img))
        for n in range(0, np.shape(img)[0]):
            for m in range(0, np.shape(img)[1]):
                img[n,m] = (max(img_original_RGB[n,m,:]) == img_original_RGB[n,m,i])
        img = (img==1)
        return img

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


def detect_calculate_pixel(img, i):
    # Takes image and color (R/G/B) and returns filtered image and average pixel values on the box
    # i = 0, 1, 2 for R, G, B respectively
    img_i = img[:,:,i]
    img_o = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray_weight = img_gray
    # rgb_weights = [0.33,0.33,0.33]
    # img_gray_weight = np.dot(img[...,:3], rgb_weights)
    # img_gray_weight = np.uint8(np.round(img_gray_weight, 0).astype(int))

    # Seleting from image thresholding
    img_i_th = ImageProcess.image_select_by_threshold(img_i)

    # Selecting from maximum color (R/G/B)
    img_i_co = ImageProcess.image_select_by_color(img_i, img, i)

    # Combine Selected Image (Selection from threshold AND Selection from maximum color)
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
            # circle_detect=np.uint16(np.around(circle_detect))
            ret, img_ind = cv2.threshold(img_ind, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            ind_pixel_value=[]
            for row in range(min_row, max_row):
                for col in range(min_col, max_col):
                    if img_ind[row-min_row,col-min_col] == 255:
                        img_o[row, col] = [0,0,0]
                        ind_pixel_value.append(img_gray_weight[row,col])
            pixel_value[k] = round(np.average(ind_pixel_value))
        else:
            pixel_value[k] = round(np.average(img_gray_weight[min_row:max_row, min_col:max_col]))
            for row in range(min_row, max_row):
                for col in range(min_col, max_col):
                    img_o[row, col] = [0,0,0]
        k += 1

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


pixel_total_R = []
pixel_total_G = []
pixel_total_B = []
pixel_total_R_2 = []
pixel_total_G_2 = []
pixel_total_B_2 = []

for i in range(0, len(imgfiles)):
    filename = imgfiles[i]
    # Load image
    # img = cv2.imread(path+'/'+filename)
    img_original = cv2.imread(path+'/'+filename)
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_R, R_pixel_value = detect_calculate_pixel(img_original, 0)
    img_G, G_pixel_value = detect_calculate_pixel(img_original, 1)
    img_B, B_pixel_value = detect_calculate_pixel(img_original, 2)

    # print(np.shape(R_pixel_value))

    # print(np.linalg.norm(R_pixel_value))
    # print(np.linalg.norm(G_pixel_value))
    # print(np.linalg.norm(B_pixel_value))

    R_pixel_value = R_pixel_value/np.linalg.norm(R_pixel_value)
    G_pixel_value = G_pixel_value/np.linalg.norm(G_pixel_value)
    B_pixel_value = B_pixel_value/np.linalg.norm(B_pixel_value)

    # R_pixel_value = (R_pixel_value-np.min(R_pixel_value))/(np.max(R_pixel_value)-np.min(R_pixel_value))
    # G_pixel_value = (G_pixel_value-np.min(G_pixel_value))/(np.max(G_pixel_value)-np.min(G_pixel_value))
    # B_pixel_value = (B_pixel_value-np.min(B_pixel_value))/(np.max(B_pixel_value)-np.min(B_pixel_value))



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
    # ax[1,0].set_xlim([0, 255])
    # ax[1,1].set_xlim([0, 255])
    # ax[1,2].set_xlim([0, 255])
    ax[1,0].set_title('R, Count: ' + str(np.shape(R_pixel_value)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(R_pixel_value)))
    ax[1,1].set_title('G, Count: ' + str(np.shape(G_pixel_value)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(G_pixel_value)))
    ax[1,2].set_title('B, Count: ' + str(np.shape(B_pixel_value)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(B_pixel_value)))
    # img_R, R_pixel_value_2 = detect_calculate_pixel_2(img_original, 0)
    # img_G, G_pixel_value_2 = detect_calculate_pixel_2(img_original, 1)
    # img_B, B_pixel_value_2 = detect_calculate_pixel_2(img_original, 2)
    # pixel_total_R_2.append(R_pixel_value_2.tolist())
    # pixel_total_G_2.append(G_pixel_value_2.tolist())
    # pixel_total_B_2.append(B_pixel_value_2.tolist())
    # fig2, ax2 = plt.subplots(2,3, figsize=(15,15), dpi = 500) #figsize 15 15 to save dpi 500
    # fig2.suptitle(filename)
    # ax2[0,0].imshow(img_R)
    # ax2[0,0].set_axis_off()
    # ax2[0,1].imshow(img_G)
    # ax2[0,1].set_axis_off()
    # ax2[0,2].imshow(img_B)
    # ax2[0,2].set_axis_off()
    # ax2[1,0].hist(R_pixel_value_2, color = 'r')
    # ax2[1,1].hist(G_pixel_value_2, color = 'g')
    # ax2[1,2].hist(B_pixel_value_2, color = 'b')
    # ax2[1,0].set_xlim([0, 255])
    # ax2[1,1].set_xlim([0, 255])
    # ax2[1,2].set_xlim([0, 255])
    # ax2[1,0].set_title('R, Count: ' + str(np.shape(R_pixel_value_2)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(R_pixel_value_2)))
    # ax2[1,1].set_title('G, Count: ' + str(np.shape(G_pixel_value_2)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(G_pixel_value_2)))
    # ax2[1,2].set_title('B, Count: ' + str(np.shape(B_pixel_value_2)[0]) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(B_pixel_value_2)))
    fig.savefig(path + '/Uniformity_method_1' + filename.replace('bmp','png'))
    # fig2.savefig(path+ '/Uniformity_method_2' + filename.replace('bmp','png'))


fig_all, ax_all = plt.subplots(3,1, figsize=(10,10), dpi=100)
fig_all.suptitle('RGB Histogram')
ax_all[0].hist(flatten(pixel_total_R), color='r')
ax_all[1].hist(flatten(pixel_total_G), color='g')
ax_all[2].hist(flatten(pixel_total_B), color='b')
# ax_all[0].set_xlim([0,255])
# ax_all[1].set_xlim([0,255])
# ax_all[2].set_xlim([0,255])
ax_all[0].set_title('R, Count: ' + str(len(flatten(pixel_total_R))) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(flatten(pixel_total_R))))
ax_all[1].set_title('G, Count: ' + str(len(flatten(pixel_total_G))) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(flatten(pixel_total_G))))
ax_all[2].set_title('B, Count: ' + str(len(flatten(pixel_total_B))) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(flatten(pixel_total_B))))
fig_all.savefig(path+ '/Uniformity_all.png')

# fig_all_2, ax_all_2 = plt.subplots(3,1, figsize=(10,10), dpi=100)
# fig_all_2.suptitle('RGB Histogram_2')
# ax_all_2[0].hist(flatten(pixel_total_R_2), color='r')
# ax_all_2[1].hist(flatten(pixel_total_G_2), color='g')
# ax_all_2[2].hist(flatten(pixel_total_B_2), color='b')
# ax_all_2[0].set_xlim([0,255])
# ax_all_2[1].set_xlim([0,255])
# ax_all_2[2].set_xlim([0,255])
# ax_all_2[0].set_title('R, Count: ' + str(len(flatten(pixel_total_R_2))) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(flatten(pixel_total_R_2))))
# ax_all_2[1].set_title('G, Count: ' + str(len(flatten(pixel_total_G_2))) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(flatten(pixel_total_G_2))))
# ax_all_2[2].set_title('B, Count: ' + str(len(flatten(pixel_total_B_2))) + ', Uniformity: ' + "{:.2f}".format(uniformity_cal(flatten(pixel_total_B_2))))
# fig_all_2.savefig(path+ '/Uniformity_all_2.png')
# plt.show()
