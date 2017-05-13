import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# Calibrates the camera
def calibrate(images):
    imgs = glob.glob(images)

    objpoints = [] # Obtained from real undistorted image (x, y, z)
    imgpoints = [] # image points to hold points of distorted images

    # Create a zero filled array to hold object points 
    objp = np.zeros((6*9,3), np.float32)

    # Create object points using numpy's mgrid function
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Iterate through set of images available for calibration
    for fname in imgs:

        # Read image and convert to grayscale
        img = mpimg.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboardcorners, i.e. the imgpoints
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        # If found, build the list of objpoints and imgpoints
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            # img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            # plt.imshow(img)
            # plt.show()

    # Calibrate the camera using opencv to find the camera matrix and distortion coef dist.
    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# # Read any distorted image
# img = mpimg.imread('camera_cal/calibration2.jpg')
# plt.imshow(img)
# plt.show()
# dst = cv2.undistort(img, mtx, dist, None, mtx)
# plt.imshow(dst)
# plt.show()

def abs_sobel_thresh(img, thresh_min=20, thresh_max=100):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    #return scaled_sobel
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary


def warp(img):
    img_size = (img.shape[1], img.shape[0])
    
    imgx = img.shape[1]
    imgy = img.shape[0]

    x1 = 130
    y1 = 660
    x2 = 550
    y2 = 460
    x3 = 730
    y3 = y2
    x4 = (imgx - x1)
    y4 = y1 

    src = np.float32(
        [[x1, y1],
          [x2, y2],
          [x3, y3],
          [x4, y4]])

    # plt.imshow(img)
    # plt.plot(x1, y1, '.')
    # plt.plot(x2, y2, '.')
    # plt.plot(x3, y3, '.')
    # plt.plot(x4, y4, '.')
    # plt.show()

    dst = np.float32(
        [[x1, y1],
          [x1, imgy - y1],
          [x4, imgy - y1],
          [x4, y4]])

    # plt.imshow(img)
    # plt.plot(x1, y1, '.')
    # plt.plot(x1, imgy - y1, '.')
    # plt.plot(x4, imgy - y1, '.')
    # plt.plot(x4, y4, '.')
    # plt.show()

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


ret, mtx, dist, rvecs, tvecs = calibrate('camera_cal/calibration*.jpg')
img = mpimg.imread('test_images/straight_lines1.jpg')
dst = cv2.undistort(img, mtx, dist, None, mtx)
plt.imshow(dst)
plt.show()
dst = abs_sobel_thresh(dst)
plt.imshow(dst,  cmap='gray')
plt.show()
wimg = warp(dst)
plt.imshow(wimg,  cmap='gray')
plt.show()


