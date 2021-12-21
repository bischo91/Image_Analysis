# Image Analysis

<p align="center">
  <a href="https://github.com/bischo91/Image_Analysis">
  <h3 align="center">Image Analysis</h3>
  </a>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>
<br>

<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

This project is specifically made for images taken in certain methods. Thus, the programs are not likely to execute properly unless the example images are used. There are three Image Analysis programs in this project serving different purposes, and the following is a brief summary of each project:
<ol>
  <li>AFM Image Process (afm_image_process_v1.py)<br><br>
  AFM Image Process application takes tif images of CNT(Carbon Nano Tube) taken by AFM(Atomic Force Microscope) as input, and calculate surface coverage and uniformity of the CNT over the image. The GUI helps the user to navigate through multiple images and save the processed images.
  </li><br><br>
  <li>Pixel Uniformity Analysis (pixel_uniformity_analysis_v1)<br><br>
  Pixel Uniformity Analysis collects all the bmp images of microscopic images of display panels, and detect individual pixels. For each image, uniformity of three colors (R, G, B) is calculated based on the greyscale value of all the pixels. The output, png image file, displays the histogram of greyscale values and the original images marking the area where the greyscale values are accounted.
  </li><br><br>
  <li>Panel Uniformity Analysis (panel_uniformity_analysis_v4)<br><br>
  Panel Uniformity Analysis detects display panels on the jpg images in the loaded folder. Once the display area is detected, the area is partitioned into multiple small squares. The average of each square is calculated from the greyscale value, and it is taken to determine uniformity the greyscale values in that area are taken to calculate uniformity of the panel on each image file. An Excel file is produced including uniformity and other parameters for each file.
  </li>
</ol>
<br>

### Built With
This project is built with Python.
* [Python](https://www.python.org/)
<br><br>

## Getting Started
### Prerequisites
The programs require [Python](https://www.python.org/), and the following Python packages.
* opencv-python (4.5.1.48)
* scikit-image (0.18.1)
* scipy (1.6.3)
* imutils (0.5.4)
* PyQt5 (5.15.4)
* numpy (1.20.2)
* regex (2021.4.4)
* openpyxl (3.0.7)
* matplotlib (3.4.1)
* seaborn (0.11.1)
* tkinter (8.6)

To run the programs require certain files to operate, which are located in the folder, ./examples/['project name'_examples].


### Installation
There is no need for installation.
The required files are:
* afm_image_process_v1.py
* pixel_uniformity_analysis_v1.py
* panel_uniformity_analysis_v4.py
And the examples images in the folder, ./examples/['project name'_examples].
<br><br>

## Usage

1. AFM Image Process <br>
When the program runs, GUI will show. First, click 'Load Folder', locate './examples/image_analyzer_examples/', and click 'Select Folder'. Since the example images need to be cropped, click 'Resize', which crops the image into pre-defined size. The images can be navigated through '<<' and '>>' to see other images in the folder. The right bottom section shows some calculated parameters such as surface coverage (%). There are also several useful features like 'No Filter', 'Overlay', 'Skeletonize'. 'No Filter' will remove any image process, and show the original image resized. 'Overlay' will overlay the processed and binarized image over the original image. 'Skeletonize' will skeletonize the processed image, which will reduce the surface coverage.
![Image Analyzer](./screenshots/imageanalyzer_1.JPG)
Date vs Trend
* Date
  When 'Date' is selected, the plot presents data based on a single date selected.
  Scatter vs Bar
  * 'Scatter' plots all the data points on the selected 'Date'
  * 'Bar' shows the average and standard deviation of Y data
  Y-axis can be either 'Price' [USD] or 'Price/sqft' [USD/sqft]
  X-axis can be 'bds' (=number of bedrooms), 'ba' (=number of bathrooms), 'location' (=NW/SW/NE/SE/Unknown), 'sqft' (=total sqft of the property)

* Trend
  For 'Trend', Y-axis can be the average 'Price' or 'Price/sqft', and X-axis is the range of the selected date.
  The trend has a legend based on the location, or average of all.


<!-- CONTRIBUTING -->
## Contributing

Any suggestion or contributions are greatly appreciated.


<!-- CONTACT -->
<!-- ACKNOWLEDGEMENTS -->
