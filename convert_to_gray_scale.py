import cv2
import csv
import os
import matplotlib.pyplot as plt
import numpy as np

gray_scale_folder = 'equimosisv3/grayscale'
training_data_path = './equimosisv3/Training/'
cropped_folder = 'equimosisv3/cropped'

bruise_width = 224
bruise_height = 224

def convertToGrayScale(input_file, output_file):
    print(input_file)
    print(output_file)
    img = cv2.imread(input_file)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_file,gray)

def calculateHistogram(image):
    img = cv2.imread(image)
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    print(hist)
    plt.hist(img.ravel(),256,[0,256])
    plt.show()

def basicThresholding(image):
    img = cv2.imread(image, 0)
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

def adaptiveThresholding(image):
    img = cv2.imread(image, 0)
    img = cv2.medianBlur(img,5)
    ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
                'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()

def otsuBinarizationThresholding(image):
    img = cv2.imread(image,0)
    # global thresholding
    ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # Otsu's thresholding
    ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # plot all the images and their histograms
    images = [img, 0, th1,
              img, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
              'Original Noisy Image','Histogram',"Otsu's Thresholding",
              'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
    for i in range(3):
        plt.subplot(3,3,i*3+1)
        plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(3,3,i*3+2)
        plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(3,3,i*3+3)
        plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+2])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def generate_file_list(rootdir, output_csv_file_name):
    with open(output_csv_file_name, mode='w', newline='') as training_file:
        training_writer = csv.writer(training_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        training_writer.writerow(["file_name", "file_full_path", "file_class", "grayscale_file", "x", "y","width", "height", "top", "bottom", "left", "right"])
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                file_name = file
                subdir_split = subdir.split("/")
                file_full_path = ''.join([subdir,'/', file])
                gray_scale_filename = ''.join([gray_scale_folder,'/', file])
                file_class = subdir_split[-1]
                x, y, height, width = calcCentroid(gray_scale_filename)
                top, bottom, left, right = calcCropPointsFromCenter(x, y, height, width)
                new_file_name = ''.join([cropped_folder,'/', file_class,'/', file])
                crop_and_save_image_from_center(file_full_path, new_file_name, x, y)
                training_writer.writerow([file_name, file_full_path, file_class, gray_scale_filename, x, y, width, height, top, bottom, left, right])

def calcCentroid(image):
    img = cv2.imread(image)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert the grayscale image to binary image
    ret,thresh = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)
    M = cv2.moments(thresh)
    height, width = gray_image.shape
    # print(width, height)
    if (M["m00"]== 0):
        cX = width / 2
        cY = height / 2
    else:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    # print (cX, cY)
    return cX, cY, height, width

def batch_grayscale_convert(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            file_full_path = ''.join([subdir,'/', file])
            gray_scale_filename = ''.join([gray_scale_folder,'/', file])
            convertToGrayScale(file_full_path, gray_scale_filename)

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def canny_edge_detection(rootdir):
    # loop over the images
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
        	# load the image, convert it to grayscale, and blur it slightly
            file_full_path = ''.join([subdir,'/', file])
            image = cv2.imread(file_full_path)
            # print(file_full_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        	# apply Canny edge detection using a wide threshold, tight
        	# threshold, and automatically determined threshold
            wide = cv2.Canny(blurred, 10, 200)
            tight = cv2.Canny(blurred, 225, 250)
            auto = auto_canny(blurred)
        	# show the images
            # cv2.imshow("Blurred", blurred)
            cv2.imshow("Original", image)
            cv2.imshow("Auto", auto)
            cv2.imshow("Wide", wide)
            cv2.imshow("Tight", tight)
            # cv2.imshow("Edges", np.hstack([wide, tight, auto]))
            cv2.waitKey(0)

def crop_and_save_image_from_center(file_name, new_file_name, x_coord, y_coord):
    im = cv2.imread(file_name)
    height = im.shape[0]
    width = im.shape[1]

    top, bottom, left, right = calcCropPointsFromCenter(x_coord, y_coord, height, width)

    cropped = im[int(top):int(bottom), int(left):int(right)]
    cv2.imwrite(new_file_name, cropped)

def calcCropPointsFromCenter(center_x, center_y, height, width):
    top = center_y - (bruise_height)/2
    bottom = center_y + (bruise_height)/2
    left = center_x - (bruise_width)/2
    right = center_x + (bruise_width)/2

    top = int(top)
    top = max (0, top)
    bottom = int(bottom)
    bottom = min (height, bottom)

    left = int(left)
    left = max (0, left)
    right = int(right)
    right = min (width, right)

    return top, bottom, left, right

# input_file = 'equimosisv3/Training/ThreeDays/editada_20190917_203526.jpg'
input_file = 'equimosisv3/Training/MoreThanSeventeenDays/editada_IMG-20191029-WA0032.jpg'
output_file = 'test.jpg'
# convertToGrayScale(input_file, output_file)
# calculateHistogram(input_file)
# basicThresholding(input_file)
# adaptiveThresholding(input_file)
# otsuBinarizationThresholding(input_file)
# x, y = calcCentroid(input_file)
# print (x, y)
generate_file_list(training_data_path,'grayscale_centroid_file_6.csv')

# batch_grayscale_convert(training_data_path)
# canny_edge_detection(training_data_path)
