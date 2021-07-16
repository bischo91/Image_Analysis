import sys
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import axes
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import imutils
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage import measure
from skimage.filters import meijering, sato, frangi, hessian
from PIL import Image
import PIL
from scipy.ndimage import label
import math
from skimage.morphology import skeletonize, medial_axis
from numpy import loadtxt

def overlay(binary_img, img_original):
# Take binary image array and original image
# and return the unprocessed image with green marks (binary image as mask)
    for x in range(binary_img.shape[0]):
        for y in range(binary_img.shape[1]):
            if binary_img[x,y] != 0:
                img_original[x,y,0] = 0
                img_original[x,y,1] = 255
                img_original[x,y,2] = 0
                img_original[x,y,3] = 255
    return img_original

def overlay_grey(binary_img, img_original):
# Take binary image array and original image
# and return the unprocessed image with green marks (binary image as mask)
    for x in range(binary_img.shape[0]):
        for y in range(binary_img.shape[1]):
            if binary_img[x,y] != 0:
                img_original[x,y] = 255
    return img_original

def surface_coverage(img):
    x, counts = np.unique(img, return_counts=True)
    coverage=100*counts[1]/(counts[0]+counts[1])
    return coverage

def plot_all(img_1, img_2, img_3, img_4):
    fig, ax = plt.subplots(1,4, figsize=(15,5), dpi=100)
    # Original image
    ax[0].imshow(img_1)
    ax[0].set_axis_off()
    ax[0].set_title(onlyfiles[n], fontsize=10)

    ax[1].imshow(img_2)
    ax[1].set_axis_off()
    x, counts = np.unique(img_2, return_counts=True)
    if counts != []:
        coverage=100*counts[1]/(counts[0]+counts[1])
    ax[1].set_title('1. Adaptive Threshold(Gaussian)\n2. Regionprops\nSurface coverage: '+ '{:.2f}'.format(coverage) + '%', fontsize=10)

    ax[2].imshow(img_3)
    ax[2].set_axis_off()
    x, counts = np.unique(img_3, return_counts=True)
    if counts != []:
        coverage=100*counts[1]/(counts[0]+counts[1])
    ax[2].set_title('3. Detect Ridge\n4. Otsu Threshold\n5. Regionprops \nSurface coverage: '+ '{:.2f}'.format(coverage) + '%', fontsize=10)

    ax[3].imshow(img_4)
    ax[3].set_axis_off()
    ax[3].set_title('Overlay', fontsize=10)


def modified_otsu(img_in):
    # OTSU -------------- only counting up to 250
    hist = cv2.calcHist([img_in],[0],None,[201],[0,201])
    blur = cv2.GaussianBlur(img_in,(5,5),0)
    hist_norm = hist.ravel()/hist.max()
    Q = hist_norm.cumsum()
    bins = np.arange(201)
    fn_min = np.inf
    for i in range(1,201):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[200]-Q[i] # cum sum of classes
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    ret, th1 = cv2.threshold(img_RGB2GRAY, thresh, 255, cv2.THRESH_BINARY)
    return ret, th1

def sobel(img_in, *args):
    if args == ():
        degrees = 180
    else:
        degrees = int(args[0])
    alltheta=[]
    for i in range(0,degrees):
        theta = (i)*180/degrees
        alltheta.append(theta)
    i=0
    sum = np.zeros((512,512))
    for theta in alltheta:
        # rotate
        img_rot = imutils.rotate_bound(img_in, angle = theta)
        # sobel at theata
        sobely = cv2.Sobel(img_rot, cv2.CV_64F, 0, 1, ksize=5)
        # rotate back
        sobely = imutils.rotate_bound(sobely, angle = -theta)
        # resize
        cols = sobely.shape[0]
        rows = sobely.shape[1]
        mincol = round((cols-512)/2)
        maxcol = round((cols+512)/2)
        sobely = sobely[mincol:maxcol, mincol:maxcol]
        # normalize and merge
        i = i+1
        min = np.min(sobely)
        sobely = sobely - min
        max = np.max(sobely)
        div = max/255
        sobely = np.uint8(sobely/div)
        sum = sum + sobely
    # take average of normalized sobel images
    average = sum/len(alltheta)
    min = np.min(average)
    average = average - min
    max = np.max(average)
    div = max/255
    average = np.uint8(average/div)
    return average

def feature_removal(img_in, area_min, area_large, circularity_min, ecce_min):
    labeled_array, num_features = label(img_in)
    properties = measure.regionprops(labeled_array)
    bw = np.zeros(np.shape(img_in), dtype = bool)
    valid_label = set()
    ecce = []
    area = []
    cir = []
    for prop in properties:
        circularity = (prop.perimeter)**2/(4*np.pi*prop.area)
        if area_min < prop.area:
            if circularity_min < circularity or ecce_min < prop.eccentricity:
                valid_label.add(prop.label)
                ecce.append(prop.eccentricity)
                area.append(prop.area)
                cir.append(circularity)
        elif area_large < prop.area:
                valid_label.add(prop.label)
                ecce.append(prop.eccentricity)
                area.append(prop.area)
                cir.append(circularity)
    current_bw = np.in1d(labeled_array, list(valid_label)).reshape(np.shape(labeled_array))
    return current_bw

def detect_ridges(gray, sigma=1.0):
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges

def uniformity_meas(img, box_size):
    box_avg = []
    uniformity_array = np.zeros([512,512])
    box_size = box_size # 4 X
    for i in range(0, math.floor(img.shape[0]/box_size)):
        ii = box_size*i
        for j in range(0, math.floor(img.shape[1]/box_size)):
            jj = box_size*j
            box = img[ii:ii+box_size, jj:jj+box_size]
            box_avg.append(np.mean(box))
            uniformity_array[ii:ii+box_size, jj:jj+box_size] = np.ones(box.shape)*np.mean(box)
    return uniformity_array, np.std(box_avg)


# Set current directory as path (where the py file is is the directory)
path = os.getcwd()
path = os.path.abspath("C:/CS/python_ruby/image_process/CNT/test_img_nanoscope/20210315/t032521001f/run/")


onlyfiles = [ f for f in listdir(path) if isfile(join(path,f)) ]
i=0
for n in range(0, len(onlyfiles)):
    # Condition for image files, jpg, tif, png
    if onlyfiles[n].split('.')[1] in ['txt']:

        # img_tif = cv2.imread(join(os.path.abspath("C:/CS/python_ruby/image_process/CNT/test_img_nanoscope/20210315/t032521001f/run/t030521001f.tif")))
        # 
        # img = img_tif[80:592, 48:560]
        # 
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # img = cv2.GaussianBlur(img, (3,3), 0)
        # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
        # img = feature_removal(img, 50, 1000, 2, 0.90)
        # img = meijering(img, sigmas=1, mode='reflect', black_ridges=False)
        # min = np.min(img)
        # img = img - min
        # max = np.max(img)
        # div = max/255
        # img = np.uint8(img/div)
        # ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # img_2 = feature_removal(img, 75, 1000, 5, 0.95)



        # Load images
        img = loadtxt(join(path,onlyfiles[n]))

        img_height = img
        
        img_height = img_height-np.min(img_height)
        
        
        min = np.min(img)
        img = img - min
        max = np.max(img)
        div = max/255
        img = np.uint8(img/div)
        
        img_original = img
        # Image RGB to Grayf
        # img_RGB2GRAY = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Guassian Blur [(5,5),0]
        img_RGB2GRAY = cv2.GaussianBlur(img, (3,3), 0)
        # Adaptive Gaussian Threshold [(5,0)]
        img_1 = cv2.adaptiveThreshold(img_RGB2GRAY, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
        img_1 = feature_removal(img_1, 50, 1000, 2, 0.90)
        # Detect Ridges [Hessian, sigma = 0.75]
        img_2 = meijering(img_1, sigmas=1, mode='reflect', black_ridges=False)
        min = np.min(img_2)
        img_2 = img_2 - min
        max = np.max(img_2)
        div = max/255
        img_2 = np.uint8(img_2/div)
        # Otsu Threshold
        ret, img_2 = cv2.threshold(img_2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Reverse Image
        # img_2 = 255-img_2
        # Regionprops [keep (ecc.>0.98 AND area>25) OR (area>1000)]
        img_2 = feature_removal(img_2, 75, 1000, 5, 0.95)
        # 
        # # Plot
        img_copy = img_original.copy()
        img_overlay = overlay_grey(img_2, img_copy)
        
        
        img_copy = img_original.copy()
        fig1, ax1 = plt.subplots(1,2, figsize=(10,5), dpi=100)
        # 
        ax1[0].imshow(img)
        ax1[0].set_axis_off()
        ax1[0].set_title(onlyfiles[n], fontsize=10)
        
        img_2, distance = medial_axis(img_2, return_distance=True)
        ax1[1].imshow(img_2)
        ax1[1].set_axis_off()
        x, counts = np.unique(img_2, return_counts=True)
        if counts != []:
            coverage=100*counts[1]/(counts[0]+counts[1])
        ax1[1].set_title('Processed Image\nSurface coverage: '+ '{:.2f}'.format(coverage) + '%', fontsize=10)


        area = []
        # if 'height' in onlyfiles[n]:
        print(onlyfiles[n])
        for x in range(img_2.shape[0]):
            for y in range(img_2.shape[1]):
                if img_2[x,y] != 0:
                    height = img_height[x,y]
                    area.append(height*19.53*10**-9)
    #                 if height<0:
    #                     neg+=1
    #                 else: 
    #                     pos+=1
    #     print('neg: '+str(neg))
    #     print('pos: '+str(pos))
    #
    #
        coverage = sum(area)/((10*10**-6)*(10*10**-6))
        print(coverage*100)




plt.show()
