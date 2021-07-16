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
import seaborn as sb

class ImageProcess:
    def __init__(self, img):
        self.img = img

    def crop_512(img):
        # Resize AFM image to 512x512
        return img[80:592, 48:560]

    def feature_removal(img, area_min, area_large, circularity_min, ecce_min):
        # Filter features defined using scipy.ndimage.label and skimage.measure.regionprops
        # Will keep regions that are:
        # 1. Area > area_min & (Circularity > circularity_min or Eccentricity > ecce_min)
        # 2. Area > area_large

        # Define/Label regions
        labeled_array, num_features = label(img)
        properties = measure.regionprops(labeled_array)
        # Initialize (Define variables)
        bw = np.zeros(np.shape(img), dtype = bool)
        valid_label = set()
        ecce = []
        area = []
        cir = []
        for prop in properties:
            circularity = (prop.perimeter)**2/(4*np.pi*prop.area)
            if area_min < prop.area:
                if circularity_min < circularity or ecce_min < prop.eccentricity:
                    # 1. Area > area_min & (Circularity > circularity_min or Eccentricity > ecce_min)
                    valid_label.add(prop.label)
                    ecce.append(prop.eccentricity)
                    area.append(prop.area)
                    cir.append(circularity)
            elif area_large < prop.area:
                    # 2. Area > area_large (for large network of CNT)
                    valid_label.add(prop.label)
                    ecce.append(prop.eccentricity)
                    area.append(prop.area)
                    cir.append(circularity)
        current_bw = np.in1d(labeled_array, list(valid_label)).reshape(np.shape(labeled_array))
        return current_bw

    def detect_ridges_meijering(img):
        # Apply Meijdering filter and convert back to grey scale image
        img = meijering(img, sigmas=1, mode='reflect', black_ridges=False)
        # Convert the image to grayscale
        img = img - np.min(img)
        max = np.max(img)
        div = max/255
        return np.uint8(img/div)

class PostImageProcess:
    def __init__(self, img):
        self.img = img

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

    def surface_coverage(img):
        # Takes binary image and returns surface coverage
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

    def uniformity(img):
        N = 500 #dimensionality of the image matrix
        z = []
        for x in np.nditer(img):
            if x ==255:
                z.append(x)
        ones = np.where(img ==255, 1, img)
        dim_col = 500 #image matrix column size in px
        dim_row = 500 #image matrix row size in px
        s = 50 #size of the submatrices (they'll be dim = sxs PIXELS)
        col_end = int(dim_col/s) #num of matrix splits per col
        row_end = int(dim_row/s) #num of matrix splits per row
        M_sum = np.empty(shape=(col_end,row_end),dtype='object') #empty matrix
        for i in range(col_end):
            for j in range(row_end):
                temp = ones[(i*s):(i*s)+s,(j*s):(j*s)+s]
                z = np.sum(temp)
                M_sum[i,j] = z
        ones = np.where(M_sum==255, 1, M_sum)
        # below produces the spatially-appropriate density matrix
        # values, reported as % coverage inside the sub-matrix segments:
        M_d = (M_sum/(col_end*row_end))
        # --- below this we begin the statistical analysis and reporting ---
        vari = np.var(M_d, ddof=1)
        mean = np.mean(M_d)
        stdev = np.std(M_d)
        Cv = (stdev/mean)*100
        # round the var, mean, stddev to two or three sigfigs for plot labeling:
        rmean = '%.3g' % mean
        rstdev = '%.4g' % stdev
        rvar = '%.4g' % vari
        return col_end, row_end, rvar, rmean, rstdev, Cv, M_d


# Set current directory as path (where the py file is is the directory)
path = os.getcwd()
path = os.path.abspath("C:/CS/python_ruby/image_process/CNT/test_img/test")
onlyfiles = [ f for f in listdir(path) if isfile(join(path,f)) ]
i=0

for n in range(0, len(onlyfiles)):
    # Condition for image files, jpg, tif, png
    if onlyfiles[n].split('.')[1] in ['jpg','tif','png']:
        # Load images
        img = cv2.imread(join(path,onlyfiles[n]))
        img_original = mpimg.imread(join(path,onlyfiles[n]))

        # Resize images / Image crop
        img_original = ImageProcess.crop_512(img_original)
        img_original = img_original[6:506, 6:506]
        img = ImageProcess.crop_512(img)
        img = img[6:506, 6:506]

        # Image RGB to Gray
        img_RGB2GRAY = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Guassian Blur [(5,5),0]
        img_RGB2GRAY = cv2.GaussianBlur(img_RGB2GRAY, (3 ,3), 0)

        # Adaptive Gaussian Threshold [(5,0)]
        img_1 = cv2.adaptiveThreshold(img_RGB2GRAY, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)

        # Regionprops [keep (ecc.>0.95 AND area>25) OR (area>1000)]
        img_1 = ImageProcess.feature_removal(img_1, 50, 1000, 2, 0.90)

        # Detect Ridges [meijering, sigma = 1]
        img_2 = ImageProcess.detect_ridges_meijering(img_1)

        # Otsu Threshold
        ret, img_2 = cv2.threshold(img_2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Regionprops [keep (ecc.>0.98 AND area>25) OR (area>1000)]
        img_2 = ImageProcess.feature_removal(img_2, 75, 1000, 5, 0.95)

        # Plot
        img_copy = img_original.copy()
        img_overlay = PostImageProcess.overlay(img_2, img_copy)
        # PostImageProcess.plot_all(img, img_1, img_2, img_overlay)
        # Image.fromarray(img_overlay).save(str(onlyfiles[n].split('.')[0]) + "_overlay.tif")
        Image.fromarray(img_overlay).show()

        img_otsu = img_2

        N = 500 #dimensionality of the image matrix
        z = []

        # for x in np.nditer(img_otsu):
        #     if x ==255:
        #         z.append(x)
        #
        # coverage = (len(z)/N**2)*100
        coverage = PostImageProcess.surface_coverage(img_2)
        print("Obtained % Surface Coverage: " + str(coverage) + "%")
        # print("BLF Parameters = " + f_param)

        # replace all the 255 pixels with 1's in otsu binarized image:

        ones = np.where(img_otsu==255, 1, img_otsu)


        # --- below this we create the image slicing and density matrix ---


        dim_col = 500 #image matrix column size in px
        dim_row = 500 #image matrix row size in px
        s = 50 #size of the submatrices (they'll be dim = sxs PIXELS)

        col_end = int(dim_col/s) #num of matrix splits per col
        row_end = int(dim_row/s) #num of matrix splits per row

        M_sum = np.empty(shape=(col_end,row_end),dtype='object') #empty matrix

        for i in range(col_end):
            for j in range(row_end):
                temp = ones[(i*s):(i*s)+s,(j*s):(j*s)+s]
                z = np.sum(temp)
                M_sum[i,j] = z

        ones = np.where(M_sum==255, 1, M_sum)


        # below produces the spatially-appropriate density matrix
        # values, reported as % coverage inside the sub-matrix segments:

        M_d = (M_sum/(col_end*row_end))

        # print(M_d)

        # --- below this we begin the statistical analysis and reporting ---

        vari = np.var(M_d, ddof=1)
        mean = np.mean(M_d)
        stdev = np.std(M_d)
        Cv = (stdev/mean)*100

        # round the var, mean, stddev to two or three sigfigs for plot labeling:
        rmean = '%.3g' % mean
        rstdev = '%.4g' % stdev
        rvar = '%.4g' % vari

        print("Uniformity Block Size: "+str(col_end)+"x"+str(row_end))
        print("Variance: " + str(rvar))
        print("Mean: " + str(rmean))
        print("StdDev: " + str(rstdev))
        print("Coeff. Variation: " + str('%.4g' % Cv) +"%")

        # plot the density histogram:

        plottitle ="Cv =  " + str('%.4g' % Cv) +"%"
        plotlabel = r'$\mu$'+"="+str(rmean)+', '+r'$\sigma$'+'='+str(rstdev)

        hist = plt.hist(M_d, bins='auto', label=plotlabel, edgecolor='black')

        plt.xlabel('% Coverage Per Block')
        plt.ylabel('Counts')
        plt.title(plottitle)
        plt.legend(loc='upper right', frameon=True)

        # --- plot a heat map of the densities with numpy, seaborn ---

        #seaborn heatmap hates data type ndarray, so recast the density matrix first:

        M_float = np.vstack(M_d[:,:]).astype(np.float)

        # now seaborn can plot the heatmap:

        plt.figure()
        heat_map = sb.heatmap(M_float,cmap="YlGnBu")
        # plt.title(filename + '         ' + plottitle )


        # show heatmap and histogram side-by-side :
        '''
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].hist(M_d, bins='auto', label=plotlabel, edgecolor='black')
        ax[0].set_xlabel('% Coverage Per Block')
        ax[0].set_ylabel('Counts')

        ax[1].hist(M_d, bins='auto', label=plotlabel, edgecolor='black')
        ax[1].set_xlabel('% Coverage Per Block')
        ax[1].set_ylabel('Counts')

        plt.title(plottitle)
        plt.legend(loc='upper right', frameon=True)
        '''
plt.show()
