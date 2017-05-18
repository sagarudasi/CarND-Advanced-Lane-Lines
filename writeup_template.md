## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

Note: The code for this project is in main.py file.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Images in the project video are distorted. We need to undistort it before we pass it further in pipleline. 
In order to fix that I created a function calibrate which in turn, uses the OpenCV function calibrateCamera.

Camera frame can be undistorted using OpenCV function undistort and for that we need the camera matrix and distortion coefficients which are the result of camera calibration process.

For calculating these parameters, there is a set of chessboard images provided which are captured with the same camera. I used the OpenCV function findChessboardCorners which resulted in the image points and created another set of corners for reference called object points. 

I then passed these points to calibrate function which in turn retured the camera matrix and distortion coefficients by comparing the image points and the reference object points.
These values were then used to undistort the image. 

Following is the result - 

![Distorted original image][image1] ![Un-distorted image][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The very first step in the pipeline is to undistort the camera image otherwise the image may have incorrect shapes which will result in lot of error.
To undistort image, I used the OpenCV function called undistort as described in the point number 1. 

The result of the undistortion on the test image looks like below -

![Original test image][image3] ![Undistorted test image][image4]

We can clearly see the difference in the shape of the car when we undistorted the image. 
This image was then passed futher in the pipeline.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Function "combined_sobelx_hs_channel" in the main.py is responsible for filtering the image based on color transforms and gradients. Following points describe how I approached the problem.

First I used S channel + Sobel x to try to filter images. 
Saturation channel is good at identifying lane line where there are lines on light patches.
Sobel x can be used to filter lines which are close to y-axis that is the vertical lines.

Combination of these two was giving proper results in the first half, but the second half consisted of the shadow region where the S channel was not working well.

I then combined a binary image using S (Saturation) channel and H (Hue) channel and adjusted the thresholds to provide proper results on both the tricky patches. 

Finally the binary image was combined with Sobel X to get an even better result.

Following is the result of combined s channel, h channel and sobel x in two patches -

![Light patch orignal][image5] ![Light patch filtered][image6]
![Shadow patch original][image7] ![Shadow patch filtered][image8]

```python
  combined_binary[((h_binary == 1) & (s_binary == 1)) | (sxbinary == 1)] = 1
```

The entire code for filtering images is in the "combined_sobelx_hs_channel" function.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In order to transform the image into a "Bird eye" view, there are two functions "wrap" and "unwrap" written in the main.py file. These functions are responsible for taking an undistorted image and returing a wraped and unwarped binary respectively.

OpenCV provides a function called warpPerspective which takes a perspective transformation matrix as input and returns a warped image. 

To get the matrix I used getPerspectiveTransform function of OpenCV library. This function takes image co-ordinates as source and destination and return as matrix that can be used to strech and interpolate the image for transforming it. Liner interpolation was used to fill the new pixels.

I used straight line test image to derive the co-ordinates of the polygon because we can see if the transform is resulting in parallel lane lines. This confirms that the transformation is good.

Source and destination points (Bottom left to bottom right clockwise) -

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 270, 675      | 270, 675      | 
| 587, 455      | 587, 20       |
| 693, 455      | 693, 20       |
| 1035, 675     | 1035, 675     |

Following is the result of perpective transformation on the test image -

![Orignal test image][image3] ![Perspective transformation test image][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

At this point, we have a perspective transformed (bird eye view) filtered and undistored image to process. 

Following is the resulted binary image -
![Perspective transformed filterted and undistorted image][image10]

As described in the CarND training, we can plot a histogram to see where the pixels are bright (i.e. probably the lane line starts there) and we can then use sliding window to find out other pixels that belongs to the left and right lanes.

The function "fit_poly" in the main.py is responsible in doing that.

Also, once we find the pixels, we can then limit our search area for the next frame onwards to do a targetted search of lane lines in the area where there is more probability. This will reduce the error of getting the unwanted noise being detected as lane lines as it will fall outside of our search area.

I then used the numpy function "polyfit" to fit a second order polynomial to left and right lane lines -

Following is the result of the sliding window and poly fit -
![Lane lines on transformed image][image11]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Functions "calculate_radii" and "car_location" are responsible for computing the radius and car position in world space respectively.

In order to calculate the radius, we can utilize the final lane points and fit a poly line inside it.
Following code was taken from the Udacity training slide to calculate the radius.
```python
  left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
  right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
  left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
  right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

We need the factor that how much one pixel corresponds to in meters in real world space. These factors are xm_per_pix and ym_per_pix. Following values were taken for each respectively - 3.7/700 30/720. 

The left and right lane radius were then averaged out compute the estimated radius of the lane curvature.

![Lane curvature][image12]

To compute the car location in the lane, I used the camera frame center as reference (since the camera is fixed on car). Then I computed the center of the lower section of lane by dividing x co-ordinate at 0th pixel of left and right lane from bottom. I computed these value from the generated poly line of second order and max y co-ordinate.

Diffence between the frame center and the above point was used as a measure of car location in the lane.

![Car position][image13]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Finally all the above data was combined and a result image as following was generated.

![Final result][image14]

Polygon of the shape of fitted polyline was drawn on top of lane to visualize the lane area generated using the above pipeline.
The image was mutated in the fit_poly function only and it was unwarped on the original undistorted image.
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The difficult part was to identify a combination of channels that works best on the two patches. In order to understand that, I will have to understand the color space of images fully.

Another part was the sobel function where I faced difficulty. One improvement will be to add gradient filter so that it picks lines in specific gradient range. This will help in 

Pipeline is likely to fail in following circumstances -
If the noise is greater (like in the challenge video) and the lane lines are very close to the boundary of the road where it is difficult to filter lane and road boundaries.
If lane lines are very inconsistent and lot of gap is there in between (i.e. lines are missing for certain time).

