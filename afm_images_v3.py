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
from sklearn.cluster import KMeans

class ImageProcess:
    def __init__(self, img):
        self.img = img

    def crop_512(img):
        return img[80:592, 48:560]

    def feature_removal(img, area_min, area_large, circularity_min, ecce_min):
        labeled_array, num_features = label(img)
        properties = measure.regionprops(labeled_array)
        bw = np.zeros(np.shape(img), dtype = bool)
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

    def detect_ridges_meijering(img):
        img = meijering(img, sigmas=1, mode='reflect', black_ridges=False)
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
                    img_original[x,y,1] = 0
                    img_original[x,y,2] = 255
                    img_original[x,y,3] = 255
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

    def num_of_interaction_in_radius(radius, img_skel, i, j):
        img_cir_temp = cv2.circle(np.zeros(img_skel.shape), (i,j), radius, (255,0,0), 1)
        num_of_interaction = np.sum(np.logical_and(img_cir_temp, img_skel))
        return num_of_interaction



def line_intersection(line1, line2, polar=False):
    if not polar:
        line1 = line1.reshape(2, 2)
        line2 = line2.reshape(2, 2)
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
           return -1, -1
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y
    else:
        rho1, theta1 = line1[0]
        rho2, theta2 = line2[0]
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        try:
            x0, y0 = np.linalg.solve(A, b)
        except:
            return [-1, -1]
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return [x0, y0]

# Set current directory as path (where the py file is is the directory)
path = os.getcwd()

path = os.path.abspath(r"C:\Users\bisch\Desktop\Mattrix\image_process\CNT\test_img\test\test1".replace("\\","/"))
# path = os.path.abspath("G:\Shared drives\MTTX_Team Drive\R&D Projects\Image Analysis\Test Images for Analysis\Standard Images for Recipe Testing")
onlyfiles = [ f for f in listdir(path) if isfile(join(path,f)) ]
i=0


for n in range(0, len(onlyfiles)):
# for n in range(0, 1):
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
        img_skel, distance = medial_axis(img_2, return_distance=True)
        img_otsu = img_2
        # N = 500 #dimensionality of the image matrix
        # z = []
        # for x in np.nditer(img_otsu):
        #     if x ==255:
        #         z.append(x)
        # coverage = (len(z)/N**2)*100
        # coverage = PostImageProcess.surface_coverage(img_2)
        # print("Obtained % Surface Coverage: " + str(coverage) + "%")
        # print("BLF Parameters = " + f_param)
        # replace all the 255 pixels with 1's in otsu binarized image:
        # ones = np.where(img_otsu==255, 1, img_otsu)
        #
        # # --- below this we create the image slicing and density matrix ---
        #
        #
        # dim_col = 500 #image matrix column size in px
        # dim_row = 500 #image matrix row size in px
        # s = 50 #size of the submatrices (they'll be dim = sxs PIXELS)
        #
        # col_end = int(dim_col/s) #num of matrix splits per col
        # row_end = int(dim_row/s) #num of matrix splits per row
        #
        # M_sum = np.empty(shape=(col_end,row_end),dtype='object') #empty matrix
        #
        # for i in range(col_end):
        #     for j in range(row_end):
        #         temp = ones[(i*s):(i*s)+s,(j*s):(j*s)+s]
        #         z = np.sum(temp)
        #         M_sum[i,j] = z
        # ones = np.where(M_sum==255, 1, M_sum)
        # below produces the spatially-appropriate density matrix
        # values, reported as % coverage inside the sub-matrix segments:
        # M_d = (M_sum/(col_end*row_end))
        # print(M_d)
        # --- below this we begin the statistical analysis and reporting ---
        # vari = np.var(M_d, ddof=1)
        # mean = np.mean(M_d)
        # stdev = np.std(M_d)
        # Cv = (stdev/mean)*100
        # round the var, mean, stddev to two or three sigfigs for plot labeling:
        # rmean = '%.3g' % mean
        # rstdev = '%.4g' % stdev
        # rvar = '%.4g' % vari
        # print("Uniformity Block Size: "+str(col_end)+"x"+str(row_end))
        # print("Variance: " + str(rvar))
        # print("Mean: " + str(rmean))
        # print("StdDev: " + str(rstdev))
        # print("Coeff. Variation: " + str('%.4g' % Cv) +"%")
        # np.savetxt('test2.txt', img_2, fmt='%1i')
        # plot the density histogram:
        # plottitle ="Cv =  " + str('%.4g' % Cv) +"%"
        # plotlabel = r'$\mu$'+"="+str(rmean)+', '+r'$\sigma$'+'='+str(rstdev)
        # hist = plt.hist(M_d, bins='auto', label=plotlabel, edgecolor='black')
        # plt.xlabel('% Coverage Per Block')
        # plt.ylabel('Counts')
        # plt.title(plottitle)
        # plt.legend(loc='upper right', frameon=True)
        # --- plot a heat map of the densities with numpy, seaborn ---
        #seaborn heatmap hates data type ndarray, so recast the density matrix first:
        # M_float = np.vstack(M_d[:,:]).astype(np.float)
        # now seaborn can plot the heatmap:
        # plt.figure()
        # heat_map = sb.heatmap(M_float,cmap="YlGnBu")
        # plt.close('all')
        # plt.figure()
        # plt.imshow(img_original)
        img_con = np.zeros(img_2.shape)
        img_cir = np.zeros(img_2.shape)
        counter =0
        img_2 = img_skel*1

        # img_2 = img_2*1
        counter_cir=0
        short_interaction_list =[]
        long_interaction_list =[]
        Y=[]
        X=[]

        x_y_coord = []
        for i in range(2, img_2.shape[0]-2):
            for j in range(2, img_2.shape[1]-2):
                # if img_2[i,j] and img_2[i-1, j] and img_2[i+1, j] and img_2[i,j+1] and img_2[i,j-1]:
                # and img_2[i+1, j+1] and img_2[i+1,j-1] and img_2[i-1,j+1] and img_2[i-1,j-1]:
                # and img_2[i-2, j] and img_2[i+2, j] and img_2[i,j+2] and img_2[i,j-2]:
                connectivity = img_2[i,j]+img_2[i-1, j]+img_2[i+1, j]+img_2[i,j+1]+img_2[i,j-1]+\
                img_2[i+1, j+1]+img_2[i+1,j-1]+img_2[i-1,j+1]+img_2[i-1,j-1]
                if connectivity>3:
                    img_con[i,j] = 1
                    img_con[i+1,j+1] = 1
                    img_con[i-1,j-1] = 1
                    img_con[i+1,j-1] = 1
                    img_con[i-1,j+1] = 1
                    img_con[i+1,j] = 1
                    img_con[i,j+1] = 1
                    img_con[i-1,j] = 1
                    img_con[i,j-1] = 1
                    # img_cir = cv2.circle(img_cir, (j,i), 5, (255,0,0), 1)
                    counter +=1
                    # img_cir_temp_10 = cv2.circle(np.zeros(img_2.shape), (j,i), 10, (255,0,0), 1)
                    # img_cir_temp_5 = cv2.circle(np.zeros(img_2.shape), (j,i), 5, (255,0,0), 1)
                    # img_cir_temp_3 = cv2.circle(np.zeros(img_2.shape), (j,i), 3, (255,0,0), 1)
                    # num_of_interaction_10 = np.sum(np.logical_and(img_cir_temp_10, img_skel))
                    # num_of_interaction_5 = np.sum(np.logical_and(img_cir_temp_5, img_skel))
                    # num_of_interaction_3 = np.sum(np.logical_and(img_cir_temp_3, img_skel))

                    # if PostImageProcess.num_of_interaction_in_radius(10, img_skel, i,j)>6:
                    #     # img_cir = cv2.circle(img_cir, (j,i), 3, (255,0,0), 1)
                    #     img_cir[i,j] = 1
                    #     counter_cir +=1
                    # elif PostImageProcess.num_of_interaction_in_radius(5, img_skel, i,j)>2 and PostImageProcess.num_of_interaction_in_radius(2, img_skel, i,j)>2:
                    #     img_cir[i,j] = 1
                    #     # img_cir = cv2.circle(img_cir, (j,i), 3, (255,0,0), 1)
                    #     counter_cir +=1

                    sum_of_interaction_long = 0
                    sum_of_interaction_short = 0
                    for m in range (1, 5):
                        short_range = PostImageProcess.num_of_interaction_in_radius(m, img_skel, j, i)
                        sum_of_interaction_short += short_range
                    for k in range(5, 15):
                        long_range = PostImageProcess.num_of_interaction_in_radius(k, img_skel, j, i) #j is x. i is y.
                        sum_of_interaction_long += long_range
                        # X.append(k)
                        # Y.append(num_of_interaction)
                        # print('For Radius: ' + str(k) + ', # of interaction: ' +  str(num_of_interaction))

                    if sum_of_interaction_long>45 and sum_of_interaction_short>5:
                        # img_cir[i,j] = 1
                        x_y_coord.append([j,i])
                        counter_cir +=1
                    long_interaction_list.append(sum_of_interaction_long)
                    short_interaction_list.append(sum_of_interaction_short)
                    # elif num_of_interaction_3>2:
                    #     img_cir[i,j] = 1
                    #     # img_cir = cv2.circle(img_cir, (j,i), 3, (255,0,0), 1)
                    #     counter_cir +=1

        # print(interaction_list)
        for m in range(1,21):
            num = PostImageProcess.num_of_interaction_in_radius(m, img_skel, 369, 197)
            print('For Radius: ' + str(m) + ', # of int: ' + str(num))

        print('Mean # of int_short: ' + str(np.mean(short_interaction_list)))
        print('Median # of int_short: ' + str(np.median(short_interaction_list)))
        print('Max # of int_short: ' + str(np.max(short_interaction_list)))
        print('Min # of int_short: ' + str(np.min(short_interaction_list)))

        print('Mean # of int_long: ' + str(np.mean(long_interaction_list)))
        print('Median # of int_long: ' + str(np.median(long_interaction_list)))
        print('Max # of int_long: ' + str(np.max(long_interaction_list)))
        print('Min # of int_long: ' + str(np.min(long_interaction_list)))
        print(x_y_coord)

        cost = []
        n_clu = range(1, 2)
        for n in n_clu:
            kmeans =KMeans(n_clusters = n, max_iter = 500).fit(x_y_coord)
            cost.append(kmeans.inertia_)


        cost = np.array(cost)
        ind = np.where(cost == np.min(cost))
        ind = ind[0][0]
        n_k = n_clu[ind]
        n_k = 25
        kmeans = KMeans(n_clusters = n_k, max_iter = 500).fit(x_y_coord)
        # kmeans =KMeans(n_clusters = 5, max_iter = 500).fit(x_y_coord)

        print(kmeans.cluster_centers_)
        for coord_center in kmeans.cluster_centers_:
            # img_cir[round(coord_center[0]),round(coord_center[1])]=1
            img_cir = cv2.circle(img_cir, (round(coord_center[0]),round(coord_center[1])), 25, (255,0,0), 2)
        # plt.figure()
        print(len(kmeans.cluster_centers_))
        # plt.scatter(X,Y)


        # plt.figure()
        # x= np.linspace(0,len(short_interaction_list), len(short_interaction_list))
        # plt.plot(x, short_interaction_list)
        # x= np.linspace(0,len(long_interaction_list), len(long_interaction_list))
        # plt.plot(x, long_interaction_list)

        # labeled_array, num_features = label(img_cir)
        # properties = measure.regionprops(labeled_array)
        # bw = np.zeros(np.shape(img), dtype = bool)
        # valid_label = set()
        # area = []
        # for prop in properties:
        #     if prop.area > 1:
        #         valid_label.add(prop.label)
        #         area.append(prop.area)
        # current_bw = np.in1d(labeled_array, list(valid_label)).reshape(np.shape(labeled_array))
        # img_cir = current_bw

        img_cir = img_cir*1
        # img_cir = img_con*1
        print('total # of pixel (connectivity): ' + str(counter))
        print('total # of pixel satisfied: ' + str(counter_cir))
        plt.figure()
        img_copy_temp = img_original.copy()
        img_overlay_con = PostImageProcess.overlay(img_cir, img_copy_temp)
        # img_overlay_con = PostImageProcess.overlay(img_con, img_copy_temp)
        plt.axis('off')
        plt.imshow(img_overlay_con)
        # plt.imshow(img_con)
        plt.figure()
        plt.imshow(np.logical_or(img_cir, img_skel))
        # plt.imshow(img_cir)
        plt.axis('off')
        # plt.imshow(cv2.cornerHarris(np.uint8(img_2), 2, 3, 0.04))




plt.show()
