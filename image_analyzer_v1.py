# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'image_analyzer.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!

import sys
import math
from os import listdir
from os.path import isfile, join
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes
import matplotlib.image as mpimg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import skimage
from skimage.measure import regionprops
from skimage import measure
from skimage.morphology import medial_axis
from skimage.filters import meijering
import scipy
from scipy.ndimage import label


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

def overlay_img(binary_img, img_original):
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
    x, counts = np.unique(img, return_counts=True)
    coverage=100*counts[1]/(counts[0]+counts[1])
    coverage_percentage_str = str("{:.2f}".format(coverage))+ str('%')
    return coverage_percentage_str

def image_filter(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 0)
    img = feature_removal(img, 50, 1000, 2, 0.90)
    img = meijering(img, sigmas=1, mode='reflect', black_ridges=False)
    min = np.min(img)
    img = img - min
    max = np.max(img)
    div = max/255
    img = np.uint8(img/div)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = feature_removal(img, 75, 1000, 5, 0.95)
    # img_skel, distance = medial_axis(img, return_distance=True)
    return img

class Ui_MainWindow(QWidget):
    path = ''
    img_files = []
    def load_folder(self):
        self.path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if self.path != '':
            self.lineEdit.setText(self.path)
            self.lineEdit.setDisabled(True)
            allfiles = [ f for f in listdir(self.path) if isfile(join(self.path,f)) ]
            self.img_files = []
            for n in range(0,len(allfiles)):
                # if allfiles[n].split('.')[1] in ['jpeg','jpg','tif','png']:
                if allfiles[n].split('.')[1] in ['tif']:
                    self.img_files.append(allfiles[n])
            self.i = 0

            if self.img_files != []:
                self.img, self.img_original = self.update_image()


    def update_image(self):
        if self.img_files != []:
            self.img_original = mpimg.imread(self.path+'/'+ self.img_files[self.i], format="tif")
            # self.img_original = mpimg.imread(self.path+'/'+ self.img_files[self.i])
            self.img = cv2.imread(self.path+'/'+ self.img_files[self.i])
            if self.checkBox_2.isChecked() == True:
                # Resize
                self.img = self.img[80:592, 48:560]
                self.img_original = self.img_original[80:592, 48:560]
            if self.checkBox_4.isChecked() == True:
                # No filter
                self.checkBox.setDisabled(True)
                self.checkBox_3.setDisabled(True)
                self.checkBox.setChecked(False)
                self.checkBox_3.setChecked(False)
                self.label_8.setVisible(False)
                self.img = self.img_original
            else:
                self.checkBox.setDisabled(False)
                self.checkBox_3.setDisabled(False)
                self.label_8.setVisible(True)
                self.img = image_filter(self.img)
                if self.checkBox_3.isChecked()==True:
                    # Skeletonize
                    skel, distance = medial_axis(self.img, return_distance=True)
                    self.img = skel
                self.coverage_percentage = surface_coverage(self.img)
                self.label_8.setText('Surface Coverage: ' + self.coverage_percentage)
                if self.checkBox.isChecked() == True:
                    # Overlay
                    self.img_copy = self.img_original.copy()
                    self.img = overlay_img(self.img, self.img_copy)
            self.label_3.setText(self.img_files[self.i])
            self.ax.imshow(self.img_original)
            self.ax_1.imshow(self.img)
            self.canvas.draw()
            self.canvas_1.draw()
            self.ax_hist.clear()
            img_RGB2GRAY = cv2.cvtColor(self.img_original, cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist([img_RGB2GRAY], [0], None, [201], [0,201])
            self.ax_hist.plot(hist)
            self.canvas_hist.draw()
            return self.img, self.img_original


    def next_image(self):
        if self.img_files != []:
            self.i += 1
            if self.i > len(self.img_files)-1:
                self.i= 0
            self.img, self.img_original = self.update_image()

    def prev_image(self):
        if self.img_files != []:
            self.i -= 1
            if self.i < 0:
                self.i = len(self.img_files)-1
            self.img, self.img_original = self.update_image()


    def saveone(self, savingall):

        file_name = self.img_files[self.i].replace('.tif','').replace('.jpg','').replace('.png','').replace('.jpeg','')
        # new_file_name string is added to the original file name.
        if self.img_files != []:
            if self.checkBox_4.isChecked() == False:
                file_name = file_name + '_' + self.coverage_percentage
            if self.checkBox_2.isChecked() == True:
                # Resize
                file_name = file_name + '_resized'
            if self.checkBox_3.isChecked()==True:
                # Skeletonize
                file_name = file_name + '_skeletonized'
            if self.checkBox.isChecked() == True:
                # Overlay
                file_name = file_name + '_overlay'
            if savingall == False:
                try:
                    self.save_path = QFileDialog.getExistingDirectory(self, "Select Folder")
                    self.fig_1.savefig(self.save_path + '/' + file_name + '.tif', pad_inches = 0)
                except:
                    pass
            self.fig_1.savefig(self.save_path + '/' + file_name + '.tif', pad_inches = 0)

    def saveall(self):
        if self.img_files != []:
            temp_ind = self.i
            try:
                self.save_path = QFileDialog.getExistingDirectory(self, "Select Folder")
                for j in range(0,len(self.img_files)):
                    self.i = j
                    self.img, self.img_original = self.update_image()
                    self.saveone(True)
                self.i = temp_ind
                self.img, self.img_original = self.update_image()
            except:
                pass

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1145, 883)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(20, 70, 1111, 771))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setGeometry(QtCore.QRect(30, 30, 512, 512))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.frame_3)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 512, 512))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.fig, self.ax = plt.subplots(1,1)
        pltsize = plt.gcf()
        pltsize.set_size_inches(100,100)
        self.fig.tight_layout(pad=0)
        self.canvas = FigureCanvas(self.fig)
        self.verticalLayout.addWidget(self.canvas)

        self.ax.set_axis_off()
        self.frame_4 = QtWidgets.QFrame(self.frame)
        self.frame_4.setGeometry(QtCore.QRect(570, 30, 512, 512))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayoutWidget_1 = QtWidgets.QWidget(self.frame_4)
        self.verticalLayoutWidget_1.setGeometry(QtCore.QRect(0, 0, 512, 512))
        self.verticalLayoutWidget_1.setObjectName("verticalLayoutWidget_1")
        self.verticalLayout_1 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_1)
        self.verticalLayout_1.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_1.setObjectName("verticalLayout_1")
        self.fig_1, self.ax_1 = plt.subplots(1,1)
        self.ax_1.set_axis_off()
        pltsize = plt.gcf()
        pltsize.set_size_inches(100,100)
        self.fig_1.tight_layout(pad=0)
        self.canvas_1 = FigureCanvas(self.fig_1)
        self.verticalLayout_1.addWidget(self.canvas_1)
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(30, 0, 512, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(570, 0, 512, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(90, 580, 390, 30))
        self.label_3.setText("")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.frame_5 = QtWidgets.QFrame(self.frame)
        self.frame_5.setGeometry(QtCore.QRect(570, 560, 511, 121))
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.checkBox = QtWidgets.QCheckBox(self.frame_5)
        self.checkBox.setGeometry(QtCore.QRect(10, 30, 150, 17))
        self.checkBox.setObjectName("checkBox")
        self.checkBox.stateChanged.connect(self.update_image)
        self.checkBox_3 = QtWidgets.QCheckBox(self.frame_5)
        self.checkBox_3.setGeometry(QtCore.QRect(10, 50, 150, 17))
        self.checkBox_3.setObjectName("checkBox")
        self.checkBox_3.stateChanged.connect(self.update_image)
        self.checkBox_4 = QtWidgets.QCheckBox(self.frame_5)
        self.checkBox_4.setGeometry(QtCore.QRect(10, 10, 150, 17))
        self.checkBox_4.setObjectName("checkBox")
        self.checkBox_4.stateChanged.connect(self.update_image)
        self.label_8 = QtWidgets.QLabel(self.frame_5)
        self.label_8.setGeometry(QtCore.QRect(30, 90, 200, 20))
        self.label_5 = QtWidgets.QLabel(self.frame_5)
        self.label_5.setGeometry(QtCore.QRect(380, 10, 121, 20))
        self.label_5.setObjectName("label_5")
        self.frame_6 = QtWidgets.QFrame(self.frame)
        self.frame_6.setGeometry(QtCore.QRect(570, 690, 512, 30))
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.label_6 = QtWidgets.QLabel(self.frame_6)
        self.label_6.setGeometry(QtCore.QRect(380, 10, 121, 20))
        self.label_6.setObjectName("label_6")
        self.label_4 = QtWidgets.QLabel(self.frame)
        self.label_4.setGeometry(QtCore.QRect(650, 550, 512, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setText("")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.pushButton_1 = QtWidgets.QPushButton(self.frame)
        self.pushButton_1.setGeometry(QtCore.QRect(850, 740, 200, 23))
        self.pushButton_1.setObjectName("pushButton_1")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(600, 740, 200, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_1.clicked.connect(self.saveall)
        self.pushButton_2.clicked.connect(self.saveone)
        self.frame_7 = QtWidgets.QFrame(self.frame)
        self.frame_7.setGeometry(QtCore.QRect(30, 610, 511, 141))
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.frame_7)
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.fig_hist, self.ax_hist = plt.subplots(1,1)
        pltsize=plt.gcf()
        pltsize.set_size_inches(5, 1.5)
        self.canvas_hist = FigureCanvas(self.fig_hist)
        self.verticalLayout_2.addWidget(self.canvas_hist)
        self.fig_hist.tight_layout(pad=2)
        self.label_7 = QtWidgets.QLabel(self.frame)
        self.label_7.setGeometry(QtCore.QRect(30, 750, 512, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.pushButton_3 = QtWidgets.QPushButton(self.frame)
        self.pushButton_3.setGeometry(QtCore.QRect(30, 580, 40, 30))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.prev_image)
        self.pushButton_4 = QtWidgets.QPushButton(self.frame)
        self.pushButton_4.setGeometry(QtCore.QRect(500, 580, 40, 30))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(self.next_image)
        self.checkBox_2 = QtWidgets.QCheckBox(self.frame)
        self.checkBox_2.setGeometry(QtCore.QRect(30, 550, 150, 17))
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_2.stateChanged.connect(self.update_image)
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(10, 10, 1121, 51))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.lineEdit = QtWidgets.QLineEdit(self.frame_2)
        self.lineEdit.setGeometry(QtCore.QRect(130, 10, 941, 31))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton = QtWidgets.QPushButton(self.frame_2)
        self.pushButton.setGeometry(QtCore.QRect(10, 10, 101, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.load_folder)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1145, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Analyzer"))
        self.label.setText(_translate("MainWindow", "Original Image"))
        self.label_2.setText(_translate("MainWindow", "Processed Image"))
        self.pushButton_2.setText(_translate("MainWindow", "Save Image"))
        self.pushButton_1.setText(_translate("MainWindow", "Save All"))
        self.label_7.setText(_translate("MainWindow", "Histogram"))
        self.pushButton_3.setText(_translate("MainWindow", "<<"))
        self.pushButton_4.setText(_translate("MainWindow", ">>"))
        self.checkBox_3.setText(_translate("MainWindow", "Skeletonize"))
        self.checkBox_2.setText(_translate("MainWindow", "Resize"))
        self.checkBox.setText(_translate("MainWindow", "Overlay"))
        self.checkBox_4.setText(_translate("MainWindow", "No Filter"))
        self.pushButton.setText(_translate("MainWindow", "Load Folder"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
