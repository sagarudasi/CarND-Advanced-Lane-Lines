import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import math
from moviepy.editor import VideoFileClip

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

    # Calibrate the camera using opencv to find the camera matrix and distortion coef dist.
    return cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

def combined_sobelx_hs_channel(img, thresh_min=30, thresh_max=130, s_thresh_min=50, s_thresh_max=200, h_thresh_min=30, h_thresh_max=100):
    # Convert to HLS color space and separate the H and S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]

    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply threshold to s channel and get a binary image
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Apply threshold to h channel and get a binary image
    thresh = (30, 100)
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel > thresh[0]) & (h_channel <= thresh[1])] = 1

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Apply threshold to sobel x and get a binary image    
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # New image to hold combined data
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((h_binary == 1) & (s_binary == 1)) | (sxbinary == 1)] = 1

    # plt.imshow(h_binary, cmap="gray")
    # plt.show()

    return combined_binary


def warp(img):
    img_size = (img.shape[1], img.shape[0])
    
    imgx = img.shape[1]
    imgy = img.shape[0]

    # Warp co-ordinates
    x1 = 270
    y1 = 675
    x2 = 587
    y2 = 455
    x3 = 693
    y3 = y2
    x4 = 1035
    y4 = y1 

    src = np.float32(
        [[x1, y1],
          [x2, y2],
          [x3, y3],
          [x4, y4]])

    # plt.imshow(img)
    # plt.plot(x1, y1, x2, y2, '.')
    # plt.plot(x2, y2, x3, y3, '.')
    # plt.plot(x3, y3, x4, y4, '.')
    # plt.plot(x4, y4, x1, y1, '.')
    # plt.show()

    dst = np.float32(
        [[x1, y1],
          [x1, 20],
          [x4, 20],
          [x4, y4]])

    # plt.imshow(img)
    # plt.plot(x1, y1, x1, 20, '.')
    # plt.plot(x1, 20, x4, 20, '.')
    # plt.plot(x4, 20, x4, y4, '.')
    # plt.plot(x4, y4, x1, y1, '.')
    # plt.show()

    M = cv2.getPerspectiveTransform(src, dst)
    wimg = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return wimg

def unwarp(img):
    img_size = (img.shape[1], img.shape[0])
    
    imgx = img.shape[1]
    imgy = img.shape[0]

    x1 = 270
    y1 = 675
    x2 = 587
    y2 = 455
    x3 = 693
    y3 = y2
    x4 = 1035
    y4 = y1 

    src = np.float32(
        [[x1, y1],
          [x2, y2],
          [x3, y3],
          [x4, y4]])

    # plt.imshow(img)
    # plt.plot(x1, y1, x2, y2)
    # plt.plot(x2, y2, x3, y3)
    # plt.plot(x3, y3, x4, y4)
    # plt.plot(x4, y4, x1, y1)
    # plt.show()

    dst = np.float32(
        [[x1, y1],
          [x1, 20],
          [x4, 20],
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
    # New blank image to hold data
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # if left_fit and right_fit are none, means we don't have any polynomial fit so far
    if left_fit == None and right_fit == None:
        # Use sliding window to identify lane pixels 
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
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

        # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # plt.imshow(out_img)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # plt.show()
    else:
        # Search in the margin for lane pixels
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

    # New image to plot the lane area
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
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,0, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,0, 255))
    # print("windowimg: "+str(window_img.shape))
    # print("outimg: "+str(out_img.shape))
    # resultimg = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    # # print("resultimg: "+str(resultimg.shape))
    # plt.imshow(resultimg)
    # # plt.plot(left_fitx, ploty, color='yellow')
    # # plt.plot(right_fitx, ploty, color='yellow')
    # # plt.xlim(0, 1280)
    # # plt.ylim(720, 0)
    # plt.show()

    return ploty, leftx, rightx, lefty, righty, left_fit, right_fit, left_fitx, right_fitx, window_img


def calculate_radii(ploty, leftx, rightx, lefty, righty):

    # Use the identified lane polynomials to calculate the radius of curvature of lane in real world space

    y_eval = np.max(ploty)
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad

def car_location(left_fit, right_fit, imagex, imagey, cfactor):

    # Calculate the position of car in the lane
    # left lane point in the bottom
    leftlanepoint = left_fit[0]*imagey**2 + left_fit[1]*imagey + left_fit[2]
    # right lane point in the bottom
    rightlanepoint = right_fit[0]*imagey**2 + right_fit[1]*imagey + right_fit[2]
    # Center of the lane
    lanecenter = ((leftlanepoint + rightlanepoint)/2)
    # Car position calculated by taking difference in lane center and frame center
    camcenter = (imagex/2)
    carpos =  (camcenter - lanecenter) * cfactor
    return int(camcenter), int(lanecenter), carpos 

def process(img):
    global left_fit
    global right_fit 

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    csimg = combined_sobelx_hs_channel(dst)
    binary_warped = warp(csimg)

    ploty, leftx, rightx, lefty, righty, left_fit, right_fit, left_fitx, right_fitx, resultimg = fit_poly(binary_warped, left_fit, right_fit)

    left_curverad, right_curverad = calculate_radii(ploty, leftx, rightx, lefty, righty)

    # binary_unwarped = unwarp(resultimg)
    # finalimg = cv2.addWeighted(dst, 1, binary_unwarped, 0.8, 0)

    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # print("Left: "+str(pts_left.shape))
    # print("Rgiht: "+str(pts_right.shape))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = unwarp(color_warp)
    # Combine the result with the original image
    result = cv2.addWeighted(dst, 1, newwarp, 0.3, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX

    camcenter, lanecenter, carpos = car_location(left_fit, right_fit, dst.shape[1], dst.shape[0], (3.7/700))

    # cv2.circle(result,(result.shape[1]//2,result.shape[0]-20), 20, (0,0,255), -1)
    # cv2.circle(result,(int(camcenter),result.shape[0]), 20, (255,0,0), -1)
    # cv2.circle(result,(int(lanecenter),result.shape[0]-30), 20, (0,0,255), -1)
    # actpos = (dst.shape[1]//2) - carpos
    if carpos > 0:
        msg = "Right of center "+str(abs(round(carpos, 2)))+" (m)"
    else:
        msg = "Left of center "+str(abs(round(carpos, 2)))+" (m)"

    cv2.putText(result, str("Curvature: "+str((left_curverad+right_curverad)//2)+" (m)") ,(10,20), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(result, str("Car position:  "+msg) ,(10,50), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    return result

if __name__ == "__main__":
    # Start of main function and pipeline
    ret, mtx, dist, rvecs, tvecs = calibrate('camera_cal/calibration*.jpg')

    # ----------------------------------------------------- #
    # Test camera calibration
    # ------------------------------------------------------#

    # img = mpimg.imread('camera_cal/calibration2.jpg')
    # plt.imshow(img)
    # plt.show()

    # dst = cv2.undistort(img, mtx, dist, None, mtx)

    # plt.imshow(dst)
    # plt.show()

    # ----------------------------------------------------- #
    # Pipeline on singe image 
    # ------------------------------------------------------#

    # left_fit = None 
    # right_fit = None 

    # img = mpimg.imread('test_images/test6.jpg')
    # result = process(img)
    # plt.imshow(result)
    # plt.show()

    # ----------------------------------------------------- #
    # Pipeline on frames from video and live output 
    # ------------------------------------------------------#

    left_fit = None 
    right_fit = None

    videofile = cv2.VideoCapture('project_video.mp4')

    i=-1
    while(videofile.isOpened()):
        ret, frame = videofile.read()
        if ret == True:
            i = i + 1
            # skip frames
            # if i < 400:
            #     continue
            result = process(frame)
            cv2.imshow('frame',result)

            cv2.imwrite('output/frame_'+str(i)+'.jpg', result)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    videofile.release()
    cv2.destroyAllWindows()

    # ----------------------------------------------------- #
    # Pipeline on frames from video and record output 
    # ------------------------------------------------------#
    # left_fit = None 
    # right_fit = None
    # outputfile = 'output.mp4'
    # videoclip = VideoFileClip('project_video.mp4')
    # processed_clip = videoclip.fl_image(process)
    # processed_clip.write_videofile(outputfile, audio=False)
    
    