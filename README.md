
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

[image1]: ./output_images/undist.jpg "Undistorted"
[image2]: ./camera_cal/calibration5.jpg "Distorted"

[image3]: ./test_images/test5.jpg "Distorted"
[image4]: ./output_images/test5.jpg "Undistorted"

[image5]: ./output_images_test/straight_lines1.jpg_binary.jpg "binary"

[image6]: ./output_images/r_channel33.jpg "r_channel"
[image7]: ./output_images/binary33.jpg "binary"

[image8]: ./output_images/binary558.jpg "bad binary"

[image9]: ./output_images/warped.jpg "warped"

[image99]: ./output_images_test/test3.jpg_undist.jpg "binary warped"
[image10]: ./output_images_test/test3.jpg_binary_warped.jpg "binary warped"
[image11]: ./output_images/histogram.png "hist"
[image12]: ./output_images/sliding_window.png "sliding window"

[image13]: ./output_images_test/test3.jpg_final.jpg "final"


[video1]: ./test_videos_output/project_video_v2.mp4 "Video"
[video2]: ./test_videos_output/challenge_fail.mp4 "Video"
[video3]: ./test_videos_output/challenge_accetable.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code is located in the `calibrate_camera()` method of `helper.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. 
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time 
I successfully detect all chessboard corners in a test image.  
`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  
I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Distorted:
![alt text][image2]

Undistorted:
![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using the previous calculated camera matrix and distortion coefficient. We can use `cv2.undistort()` to correct our test images as well.
An example can be see below, other undistorted images can be found under `output_images/`:

Before distortion correction:

![alt text][image3]

After correction:

![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in `create_combined_binary()` method of `main.py`).  
Here's an example of my output for this step.  

![alt text][image5]

This is actually the main "brain" of this pipeline, the other steps depends heavily on this method to provide accurate results.
The problems i encounter here is that specific threshhold may suit for some situations (lighting, asphalt color, shadows), 
it may perform bad for other environments.
While i manage to get good results for the project video. The initial threshhold i used perform badly in the challenge videos.
The approach i took to determine combination of color and gradient threshholds was to capture several images from the video which 
represent difficult scenarios, like bad street condition or lot of shadows or bright lighting.
Some of these images can be found under `difficult_images\`

I expanded the tool describe at [Here](https://medium.com/@maunesh/finding-the-right-parameters-for-your-computer-vision-algorithm-d55643b6f954) 
with additional threshhold bars and loaded the challenging images to first show all the channel(Both RGB and HLS) information and gradients.
The code can be found under `image_tool\`

Then i try to identify the combination of threshholds to differentiate between lane markings and other objects.
At the beginning i used the following combination (not used in committed code):

```python

# Choose a Sobel kernel size
ksize = 11 # Choose a larger odd number to smooth gradient measurements

hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
s_channel = hls[:, :, 2]

s_binary2 = np.zeros_like(s_channel)
s_binary2[(s_channel >= 170) & (s_channel <= 210)] = 1

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.6, 1.4))

combined[(((gradx == 1) & (grady == 1) & (dir_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))) | (
          (s_binary2 == 1))] = 255
```
Basically it means, a pixel belongs to a lane if it matches the gradx and grady condition, because a lane should be detected by both as its neither perfectly horizontal nor vertical.
Then i also restrict the direction to filter out some of the unwanted gradients. Another similar condition use the magnitude of both gradients and the direction to determine
whether a pixel belongs to the lane, finally i add the pixels matching the s_channel restriction to the final binary image.

This worked well for the project video but failed completely for the challenge, at the end, through trial and error i used this set of combination, 
which still doesnt work perfect for the challenge video but at least provide more useful detection.
I notice that most failure are coming from false positives, basically i was detecting incorrect pixels. In order to prevent that, 
i lowered the threshholds and applied more conditions for a pixel to be classified as lane marking, i used a min value threshhold for the R channel
to exclude dark lines which can otherwise not be distingushed from a lane line. Then i added pixels with really high 
R-values which i determine directly as lane marking (in the roi). Last but not least i also added a filter for the HSV color space to detect the yellow lane marking, 
which also directly indicates a lane.

I ended up with the following method to create the binary image:
```python
    r_channel = img[:, :, 0]
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_low = np.array([0, 100, 100])
    yellow_high = np.array([50, 255, 255])


    ksize =11

    gradx = abs_sobel_thresh(l_channel, orient='x', sobel_kernel=ksize, thresh=(2, 50))
    grady = abs_sobel_thresh(l_channel, orient='y', sobel_kernel=ksize, thresh=(2, 50))
    mag_binary = mag_thresh(l_channel, sobel_kernel=ksize, mag_thresh=(20, 100))
    dir_binary = dir_threshold(l_channel, sobel_kernel=ksize, thresh=(0.5, 1.0))

    # Threshold color channel

    yellow_binary_output = np.zeros((img.shape[0], img.shape[1]))
    yellow_binary_output[(imgHLS[:, :, 0] >= yellow_low[0]) & (imgHLS[:, :, 0] <= yellow_high[0]) & (imgHLS[:, :, 1] >= yellow_low[1]) & (
                    imgHLS[:, :, 1] <= yellow_high[1]) & (imgHLS[:, :, 2] >= yellow_low[2]) & (
                    imgHLS[:, :, 2] <= yellow_high[2])] = 1

    r_binarymax = np.zeros_like(r_channel)
    r_binarymax[(r_channel >= 220) & (r_channel <= 255)] = 1

    r_binarymin = np.zeros_like(r_channel)
    r_binarymin[(r_channel >= 10) & (r_channel <= 255)] = 1

    combined = np.zeros_like(dir_binary)

    combined[(((gradx == 1) & (grady == 1) & (dir_binary == 1) & (mag_binary == 1) & (r_binarymin == 1)) |
              (r_binarymax == 1) | (yellow_binary_output == 1))] = 255
```

With that i get acceptable detection in the challenge video while retaining a correct detection in the project video.
The following example was taking out of the challenge video, where we can see a lot of lines which could lead to false detection.

R_Channel of this image:

![alt text][image6]

Binary image:

![alt text][image7]

Even though we lose some pixel information of the lane markings, we are also filtering out a lot of false positives. Its a tradeoff that
i had to make in order to get useful results for the challenge video.

I append also an example for a bad binary result during my trial and error process:

![alt text][image8]

This situation i could observe either for shadows of the trees or for bright asphalt, where pixel are falsely detected and the following steps
could not provide lane information.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a method called `calculate_warp_parameter()` in `helper.py`.
Using a image with straight lanes, we eyeball the values as follow (instead of using absolute values, 
we should use relative values depending on image size, since test images and all testvideos are the same size, 
i use absolute value for the sake of simplicity):

We use pixel coordinates based off the center of the image, because later on we calculate our position based on the offset to the
middle of the image (we assume the camera was mounted exactly in the middle of the car)

```python
    src = np.float32([[595, 450], [685, 450], [260, 680], [1020, 680]])
    dst = np.float32([[320, 0], [960, 0], [320, imshape[0]], [960, imshape[0]]])
    m = cv2.getPerspectiveTransform(src, dst)
    minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(sourceimage, M, sourceimage.shape[1::-1], flags=cv2.INTER_LINEAR)
```

I verified that my perspective transform was working as expected by warping `test_images/straight_lines1.jpg` into bird-eye view and confirm
that the left and right lanes are straight and parallel as can be seen here:

![alt text][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

There are two parts involved in this step, first, the detection of lane pixels. Depending on whether we have detected the lane in the previous frames and have already the polynom coefficients,
we either use the histogram and sliding windows methods or use the previous polynomial as a baseline to find our pixel.

First we introduce the histogram and sliding window method:
As introduced in the previous steps, we calculated a binary representation of the image which should include the lane pixels in the roi.
Using a perspektive warp we get the bird-eye view of this binary image:

Original image:

![alt text][image99]

Resulting binary image:

![alt text][image10]

From this image we calculate the histogram of the bottom half, which will likely reveal the base position of our lanes on 
the left and right side of the image.
The histogram for the previous image is:

![alt text][image11]

Using these two max values as initial starting coordinates we draw a window around them to find lane related pixels, the code for this
process is found in the `find_lane_pixels()` method in `main.py`, for each window we calculate the mean x coordinate as base for the next window.
For this project i used a windows size of 9 and a left/right margin of 90 pixels.

Sliding window applied to the previous binary image results in:

![alt text][image12]

The second step is to fit all pixels identified in the green windows to a second grade polynomial. Details can be found in the code in the `fit_polynomial()` method 

```python
    left_fit = None
    right_fit = None
    if len(leftx) > 0 and len(lefty) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) > 0 and len(righty) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
```
where leftx/lefty and rightx/righty are the pixelcoordinates within the windows for the left and right lane respectively.

As i mentioned before, the initial step of finding lane pixels can be skipped when processing a video stream after we identified a polynomial.
Since the polynomial shouldn't change too much between frames we can use the polynomial for the previous frame as a reference window to find the lane
pixels. If for some reason, we fail to find the lane (because lane pixels are not found or our new polynomial differs too much from the previous one) we skip that frame.
Should it happen multiple frames in a row, we reset our polynomial information store in our Lane and start with the sliding window method again.
The implemention are found between the `Lane.py` class and the `find_lane_pixels()` method in `main.py`.

In case we already have a fit from a previous polynomial, the window where we search for lane pixels change to:

```python
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    if LeftLane.detected:
        print("leftlane detected")
        left_fit = LeftLane.best_fit
        # Set the area of search based on activated x-values #
        # within the +/- margin of our polynomial function #
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - parameter.margin))
                          & (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                         left_fit[2] + parameter.margin)))
```

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the `Lane.py` class of my code, after a new polynomial fit was accepted, i will calculate the radius of the curvature according to the formula introduces in Lesson 8-7.

```python
    def calc_curvature(self):
        ploty = np.linspace(0, 719, 720)
        y_eval = np.max(ploty)

        # Calc new polynomial coefficients in world space
        if self.best_fit is not None:
            fitA = (parameter.xm_per_pix / parameter.ym_per_pix ** 2) * self.best_fit[0]
            fitB = (parameter.xm_per_pix / parameter.ym_per_pix) * self.best_fit[1]

        # Calculate radius in meter
        self.radius_of_curvature = (1+(2*fitA*y_eval*parameter.ym_per_pix+fitB)**2)**(3/2)/np.absolute(2*fitA)
```
The Lane center is calculated in the `main.py` since it requires information from both Lanes.
```python
center_offset = (((LeftLane.best_fit[0] * 720 ** 2 + LeftLane.best_fit[1] * 720 + LeftLane.best_fit[2]) + (
            RightLane.best_fit[0] * 720 ** 2 + RightLane.best_fit[1] * 720 + RightLane.best_fit[2])) / 2 - 640) * parameter.xm_per_pix
```

In the video we can check the results, the curvature of the left and right lanes are consistent with the actual curve radius of around 1000m.
(They are consistently lower most of the time, i think its because i used the parameters from the Lesson, but my window for the perspective transformation is actually a bit different)
The lane offset are mostly centered with an offset of some decimeters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The drawing of the resulting detected polygon representing the lane is in the `main()` method.

First we get the polynomial coefficients and then calculate the polygon border points before drawing the polygon and overlaying it on top of the 
original image
```python
left_fitx, right_fitx, ploty = fit_polynomial(binary_combined_warped)

warp_zero = np.zeros_like(binary_combined_warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))
# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, 
                             (binary_combined_warped.shape[1], binary_combined_warped.shape[0]))
# Combine the result with the original image
result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
```
The final image looks as follows:

![alt text][image13]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Final video output can be found in `test_videos_output/project_video_v2.mp4`. 

[video1]

The project video has a small "hiccup" at 0:22 where a misdetection was corrected. Although i was able to prevent the misdetection, i let this one inside the video to prove our "reset functionality".
When the polynomial coefficients starts to change too much we try to skip that frame and keep our previous coefficients, but if there are multiple frames which "fails" our specification, 
we delete our previous fits and starts with the histogram sliding windows methods again.

I also included some other results:

challenge_fail.mp4 are the failed results from a initial threshhold/combination which worked perfectly for the project video

challenge_accetable.mp4 are the results from the final threshhold/combination which has some tradeoffs but provide at least non-catastrophic failure.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most challenging part was to find the right combination and threshhold for a robust detection, in optimal lighting and low curvature situation without a lot of shadows (like in the project video),
we can find some appropriate threshhold by trial and error. As the environment variables expands like in the challenge videos, it becomes exponentially harder to find the right sets of combination and threshhold.
We just have to much possibilities to test. What we can do is create a ground truth and use all the image information (RGB channel, HLS channel, gradients, direction, magnitude etc.) as input and train classifiers to differentiate between lane and not lane pixel.
As for our manual effort, the results from both challenge videos reveal where our pipeline will fail. 
1. street is sloped (perspektive warp will be incorrect)
2. lots of shadows
3. reflections on the glass in front of the camera
4. bright sunlight on asphalt(remebering the autopilot from tesla crashing into a lkw)
5. strong curvatures
6. street repair markings
7. lane marking are completely or partly obscured (by other vehicles or leafs etc)

Without machine learning, our best chance to make our pipeline more robust in my opinion is (beside trying out better threshhold and combinations) to
use adaptive threshholds, which can for example change based on average lighting of the image, detected lane pixels etc.
We could also instead of using binary combinations of different filters, create a probability map where the pixel value represents the probability of a pixel being a lane. But here we would have to manually test a lot as well.
A complicated method would involve information from previous high confidence detections and extrapolation of those to get better results, or using high confidence of small pieces of the lane (which is partly obscures) + information that both lanes are parallel to generate more information.
Another possible improvement is to include high definition map information and gps coordinates to have a baseline when lane detection fails. 

In any case, what becomes clear is that one system of sensor information will never be enough to be safely detect the environment, since there are too many 
variables involved, therefore we always need multiple redundent sensor information. Additionally, it became obvious that a human engineer will most likely not manage
to cover all possible street scenarios while tinkering with parameter, thats why machine learning is so important in this aspect.

