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

def combined_sobelx_s_channel(img, thresh_min=20, thresh_max=100, s_thresh_min=170, s_thresh_max=255):
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return combined_binary


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
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

def unwarp(img):
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

    Minv = cv2.getPerspectiveTransform(dst, src)
    return cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)


def fit_poly(binary_warped, left_fit=None, right_fit=None):
    if left_fit == None and right_fit == None:
        print("Using sliding window")
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            # cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            # cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # plt.imshow(out_img)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # plt.show()
    else:
        print("Looking nearby ...")
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    window_img = np.zeros_like(out_img)
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (255,0, 0))
    # print("windowimg: "+str(window_img.shape))
    # print("outimg: "+str(out_img.shape))
    # resultimg = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    # print("resultimg: "+str(resultimg.shape))
    # plt.imshow(result)
    # # plt.plot(left_fitx, ploty, color='yellow')
    # # plt.plot(right_fitx, ploty, color='yellow')
    # # plt.xlim(0, 1280)
    # # plt.ylim(720, 0)
    # plt.show()
    return ploty, leftx, rightx, lefty, righty, left_fit, right_fit, window_img


def calculate_radii(ploty, leftx, rightx, lefty, righty):
    y_eval = np.max(ploty)

    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')
    return left_curverad, right_curverad

ret, mtx, dist, rvecs, tvecs = calibrate('camera_cal/calibration*.jpg')

# img = mpimg.imread('test_images/test2.jpg')
# dst = cv2.undistort(img, mtx, dist, None, mtx)
# plt.imshow(dst)
# plt.show()
# csimg = combined_sobelx_s_channel(dst)
# plt.imshow(csimg,  cmap='gray')
# plt.show()
# binary_warped = warp(csimg)
# plt.imshow(binary_warped,  cmap='gray')
# plt.show()

# ploty, leftx, rightx, lefty, righty, left_fit, right_fit, resultimg = fit_poly(binary_warped)

# left_curverad, right_curverad = calculate_radii(ploty, leftx, rightx, lefty, righty)

# binary_unwarped = unwarp(resultimg)
# plt.imshow(binary_unwarped)
# plt.show()

# # print("dst: "+str(dst.shape)+str(dst.shape[1])+" "+str(dst.shape[0]))
# # print("resultimg: "+str(resultimg.shape)+str(resultimg.shape[1])+" "+str(resultimg.shape[0]))
# finalimg = cv2.addWeighted(dst, 1, binary_unwarped, 0.8, 0)

# plt.imshow(finalimg)
# plt.show()

videofile = cv2.VideoCapture('project_video.mp4')

left_fit = None 
right_fit = None 

while(videofile.isOpened()):
    ret, frame = videofile.read()

    dst = cv2.undistort(frame, mtx, dist, None, mtx)
    csimg = combined_sobelx_s_channel(dst)
    binary_warped = warp(csimg)

    ploty, leftx, rightx, lefty, righty, left_fit, right_fit, resultimg = fit_poly(binary_warped, left_fit, right_fit)

    left_curverad, right_curverad = calculate_radii(ploty, leftx, rightx, lefty, righty)

    binary_unwarped = unwarp(resultimg)
    finalimg = cv2.addWeighted(dst, 1, binary_unwarped, 0.8, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    turnstr = "Left"
    if right_curverad < left_curverad:
        turnstr = "Right"
    elif right_curverad == left_curverad:
        turnstr = "Straight"

    cv2.putText(finalimg, str("Curvature "+str((left_curverad+right_curverad)//2)+" (m)") ,(10,20), font, 0.5,(255,255,255),1,cv2.LINE_AA)

    cv2.imshow('frame',finalimg)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videofile.release()
cv2.destroyAllWindows()

# img = mpimg.imread('test_images/test1.jpg')
# dst = cv2.undistort(img, mtx, dist, None, mtx)
# plt.imshow(dst)
# plt.show()
# dst = combined_sobelx_s_channel(dst)
# plt.imshow(dst,  cmap='gray')
# plt.show()
# binary_warped = warp(dst)
# plt.imshow(binary_warped,  cmap='gray')
# plt.show()

# #ploty, leftx, rightx, lefty, righty, left_fit, right_fit = fit_poly(binary_warped, left_fit, right_fit)

# #calculate_radii(ploty, leftx, rightx, lefty, righty)