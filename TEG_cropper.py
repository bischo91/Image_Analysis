import sys
import os
from os import listdir
from os.path import isfile, join
import cv2
import openpyxl
from openpyxl.drawing.image import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import axes
import matplotlib.image as mpimg
from PIL import Image
import PIL
import math
import cv2
import skimage
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import skimage.feature
import skimage.viewer
from skimage.measure import regionprops
from scipy.ndimage import label
from skimage import measure
from skimage.morphology import skeletonize, medial_axis
import imutils
import math
import tkinter.filedialog
import re


def find_vg_from_filename(filename):
    plus = re.findall(r'%s(\d*\.?\d+)' % 'VG_\+', filename.upper())
    minus = re.findall(r'%s(\d*\.?\d+)' % 'VG_-', filename.upper())
    zero = re.findall(r'%s(\d*\.?\d+)' % 'VG_', filename.upper())
    if plus != []:
        return '+'+plus[0]
    elif minus != []:
        return '-'+minus[0]
    elif zero != []:
        return 0
    else:
        plus = re.findall(r'%s(\d*\.?\d+)' % 'VG \+', filename.upper())
        minus = re.findall(r'%s(\d*\.?\d+)' % 'VG -', filename.upper())
        zero = re.findall(r'%s(\d*\.?\d+)' % 'VG ', filename.upper())
        if plus != []:
            return '+'+plus[0]
        elif minus != []:
            return '-'+minus[0]
        elif zero != []:
            return 0
        else:
            return None

def rearrange_files_by_vg(fileinput):
    Vg_list = []
    for filename in fileinput:
        Vg_list.append(float(find_vg_from_filename(filename)))
    Vg_list_sorted = sorted(Vg_list)
    fileoutput = []
    for Vg in Vg_list_sorted:
        fileoutput.append(fileinput[Vg_list.index(Vg)])
    return fileoutput

def detect_TEG(img, img_original, dim):
    # Takes grey and original input and returns resized and rotated panel image and boolean whether panel is detected or not
    img = cv2.GaussianBlur(img, (25,25),0)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, h = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x_max = 0
    y_max = 0
    x_min = math.inf
    y_min = math.inf
    counter = 0
    for count in contours:
        area = cv2.contourArea(count)
        if 5000>area>1000:
            counter += 1
            approx = cv2.approxPolyDP(count, 0.09 * cv2.arcLength(count, True), True)
            x1, y1 = approx[0][0]
            x2, y2 = approx[1][0]
            if max(x1, x2)>x_max:
                x_max = max(x1, x2)
                if x1>x2:
                    y_x_max = y1
                else:
                    y_x_max = y2
            if min(x1, x2)<x_min:
                x_min = min(x1, x2)
                if x1<x2:
                    y_x_min = y1
                else:
                    y_x_min = y2
            if max(y1, y2)>y_max:
                y_max = max(y1, y2)
                if y1>y2:
                    x_y_max = x1
                else:
                    x_y_max = x2
            if min(y1, y2)<y_min:
                y_min = min(y1, y2)
                if y1<y2:
                    x_y_min = x1
                else:
                    x_y_min = x2
    if 20<counter<26:
        x2 = x_max
        y2 = y_x_max
        x3 = x_y_max
        y3 = y_max
        x4 = x_min
        y4 = y_x_min
        x1 = x_y_min
        y1 = y_min
        hxy_1 = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        hxy_2 = np.sqrt((x2-x3)**2 + (y2-y3)**2)
        hx = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        if hxy_1>hxy_2:
            hx = hxy_2
            hy = hxy_1
            dh = hxy_1
            d = x1-x2
        else:
            hx = hxy_1
            hy = hxy_2
            dh = hxy_2
            d = y1-y2
        theta = np.arcsin(d/dh)*180/np.pi
        top_left_x = min([x1,x2,x3,x4])
        top_left_y = min([y1,y2,y3,y4])
        bot_right_x = max([x1,x2,x3,x4])
        bot_right_y = max([y1,y2,y3,y4])
    else:
        theta = dim[0]
        top_left_x = dim[1]
        top_left_y = dim[2]
        bot_right_x = dim[3]
        bot_right_y = dim[4]

    dy = math.ceil(abs(top_left_y-bot_right_y)*0.01)
    dx = math.ceil(abs(top_left_x-bot_right_x)*1)
    img_original = img_original[top_left_y-dy:bot_right_y+dy, top_left_x-dx:bot_right_x+dx]
    img_resized = imutils.rotate(img_original, theta)

    return img_resized, [theta, top_left_x, top_left_y, bot_right_x, bot_right_y]


# Set current directory as path (where the py file is is the directory)
path = os.getcwd()
path = os.path.abspath("C:/Users/bisch/Desktop/Mattrix/QVGA Panel/SAIT QVGA Panel/Panel 28/after encap V_SWT_TFT_+5V_exp_time_1_50s/TEG")

root = tkinter.Tk()
path = tkinter.filedialog.askdirectory(parent=root, initialdir="/", title='Select Folder')
root.withdraw()

allfiles = [f for f in listdir(path) if isfile(join(path,f))]
allfiles = [f for f in allfiles if 'cropped' not in f and 'grid' not in f]
imgfiles = [f for f in allfiles if f.upper().endswith('.JPG')]
imgfiles = rearrange_files_by_vg(imgfiles)
prev_dim = [0, 0, 0, 0, 0]
for j in range(0, len(imgfiles)):
    filename = imgfiles[j]
    print(filename)
    # Load image
    img = mpimg.imread(path+'/'+filename)
    # Pixel Array
    arr = np.array(img)
    img = cv2.imread(path+'/'+filename)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_original = mpimg.imread(path+'/'+filename)

    [img_resized, dim] = detect_TEG(img, img_original, prev_dim)
    prev_dim = dim
    # fig, ax = plt.subplots(1,1)
    # ax.set_axis_off()
    # ax.set_title(filename)
    # ax.imshow(img_resized)
    new_img_file = path+'/' + filename.replace('.jpg','').replace('.JPG', '') + '_cropped.jpg'
    cv2.imwrite(new_img_file, cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
plt.show()
