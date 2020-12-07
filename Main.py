import os
import numpy as np
import cv2
from Lane import Line
import parameter
import helper
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

def find_lane_pixels(binary_warped):
    global LeftLane, RightLane
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    if not LeftLane.detected or not RightLane.detected:
        print("using hist method")
        # Take a histogram of the bottom half of the image
        size = binary_warped.shape
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and visualize the result

        midpoint = np.int(histogram.shape[0] // 2)

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // parameter.nwindows)

        # Visualize the resulting histogram
        #plt.plot(histogram)
        #plt.show()

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(parameter.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            if not LeftLane.detected:
                win_xleft_low = leftx_current - parameter.margin
                win_xleft_high = leftx_current + parameter.margin
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high), (0, 255, 0), 2)
                # Identify the nonzero pixels in x and y within the window ###
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                if len(good_left_inds) > parameter.minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

            if not RightLane.detected:
                win_xright_low = rightx_current - parameter.margin
                win_xright_high = rightx_current + parameter.margin
                cv2.rectangle(out_img, (win_xright_low, win_y_low),
                              (win_xright_high, win_y_high), (0, 255, 0), 2)
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                   (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                right_lane_inds.append(good_right_inds)
                if len(good_right_inds) > parameter.minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass
        try:
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

    if LeftLane.detected:
        print("leftlane detected")
        left_fit = LeftLane.best_fit
        # Set the area of search based on activated x-values #
        # within the +/- margin of our polynomial function #
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                       left_fit[2] - parameter.margin))
                          & (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                         left_fit[2] + parameter.margin)))
    if RightLane.detected:
        print("rightlane detected")
        right_fit = RightLane.best_fit
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                        right_fit[2] - parameter.margin))
                           & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                          right_fit[2] + parameter.margin)))
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    LeftLane.allx = leftx
    LeftLane.ally = lefty
    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = None
    right_fit = None
    if len(leftx) > 0 and len(lefty) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) > 0 and len(righty) > 0:
        right_fit = np.polyfit(righty, rightx, 2)

    LeftLane.update_best_fit(left_fit)
    RightLane.update_best_fit(right_fit)
    if LeftLane.best_fit is None or RightLane.best_fit is None:
        print("retry to find lane pixels")
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)
        # Fit a second order polynomial to each using `np.polyfit` #
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        LeftLane.update_best_fit(left_fit)
        RightLane.update_best_fit(right_fit)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = LeftLane.best_fit[0] * ploty ** 2 + LeftLane.best_fit[1] * ploty + LeftLane.best_fit[2]
        right_fitx = RightLane.best_fit[0] * ploty ** 2 + RightLane.best_fit[1] * ploty + RightLane.best_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    # # Visualization #
    # # Colors in the left and right lane regions
    # window_img = np.zeros_like(out_img)
    # out_img[lefty, leftx] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]
    #
    # # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    #
    # # Visualization for search from prior#
    # # Generate a polygon to illustrate the search window area
    # # And recast the x and y points into usable format for cv2.fillPoly()
    # left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - parameter.margin, ploty]))])
    # left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + parameter.margin,
    #                                                                 ploty])))])
    # left_line_pts = np.hstack((left_line_window1, left_line_window2))
    # right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - parameter.margin, ploty]))])
    # right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + parameter.margin,
    #                                                                  ploty])))])
    # right_line_pts = np.hstack((right_line_window1, right_line_window2))
    #
    # # Draw the lane onto the warped blank image
    # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    # result = cv2.addWeighted(out_img, 1, window_img, 0.0, 0)
    #
    #
    #
    # # Plot the polynomial lines onto the image
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.imshow(out_img)
    # plt.show()
    # # End visualization for search from prior #

    # todo plausibility check
    #LeftLane.detected = True
    #RightLane.detected = True

    return left_fitx, right_fitx, ploty

def create_combined_binary(img, s_thresh=(170, 255)):
    global count
    img = np.copy(img)
    r_channel = img[:, :, 0]
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_low = np.array([0, 100, 100])
    yellow_high = np.array([50, 255, 255])
    yellow_binary_output = np.zeros((img.shape[0], img.shape[1]))
    yellow_binary_output[(imgHLS[:, :, 0] >= yellow_low[0]) & (imgHLS[:, :, 0] <= yellow_high[0]) & (imgHLS[:, :, 1] >= yellow_low[1]) & (
                    imgHLS[:, :, 1] <= yellow_high[1]) & (imgHLS[:, :, 2] >= yellow_low[2]) & (
                    imgHLS[:, :, 2] <= yellow_high[2])] = 1

    ksize =11

    gradx = helper.abs_sobel_thresh(l_channel, orient='x', sobel_kernel=ksize, thresh=(2, 50))
    grady = helper.abs_sobel_thresh(l_channel, orient='y', sobel_kernel=ksize, thresh=(2, 50))
    mag_binary = helper.mag_thresh(l_channel, sobel_kernel=ksize, mag_thresh=(20, 100))
    dir_binary = helper.dir_threshold(l_channel, sobel_kernel=ksize, thresh=(0.5, 1.0))

    # Threshold color channel
    s_binary2 = np.zeros_like(s_channel)
    s_binary2[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    r_binarymax = np.zeros_like(r_channel)
    r_binarymax[(r_channel >= 220) & (r_channel <= 255)] = 1

    r_binarymin = np.zeros_like(r_channel)
    r_binarymin[(r_channel >= 10) & (r_channel <= 255)] = 1

    combined = np.zeros_like(dir_binary)
    combined2 = np.zeros_like(dir_binary)
    combined[(((gradx == 1) & (grady == 1) & (dir_binary == 1) & (mag_binary == 1) & (r_binarymin == 1) & (s_binary2 == 1)) |
              (r_binarymax == 1) | (yellow_binary_output == 1))] = 255

    combined2[((gradx == 1) & (grady == 1) & (dir_binary == 1) & (mag_binary == 1) & (r_binarymin == 1)& (s_binary2 == 1))] = 255

    # combined[(((gradx == 1) & (grady == 1) & (dir_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1))) | (
    #     (s_binary2 == 1) | (r_binarymax == 1))] = 255

    # combined[(((gradx == 1) & (grady == 1) & (dir_binary == 1) & (r_binary2 == 1))
    #           | ((s_binary2 == 1) & (dir_binary == 1)))] = 255

    #cv2.imshow("r2", combined2)
    #cv2.waitKey(0)
    # Stack each channel
    # binary_combined = np.dstack((combined, combined, combined)) * 255
    #cv2.imwrite("output_images_project/r_binarymax" + str(count) + ".jpg", r_binarymax*255)
    #cv2.imwrite("output_images_project/r_channel" + str(count) +".jpg", r_channel)
    #cv2.imwrite("output_images_project/s_channel" + str(count) +".jpg", s_channel)
    #cv2.imwrite("output_images_project/l_channel" + str(count) +".jpg", l_channel)
    return combined

def process_image(image):
    global mtx, dist, M, Minv, count
    print("processing: "+str(count)+"\n")
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    binary_combined = create_combined_binary(undist)
    combined = np.zeros_like(binary_combined).astype(np.uint8)
    binary_combined2 = np.dstack((binary_combined* 255, binary_combined* 255, binary_combined* 255)).astype(np.uint8)

    imshape = image.shape
    p1x = 0
    p1y = int(imshape[0])
    p2x = int(imshape[1] / 2)
    p2y = int(0.55 * imshape[0])
    p3x = imshape[1]
    p3y = imshape[0]
    vertices = np.array([[(p1x, p1y), (p2x, p2y), (p3x, p3y)]], dtype=np.int32)

    binary_combined_roi = helper.region_of_interest(binary_combined, vertices)
    warped = cv2.warpPerspective(binary_combined_roi, M, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    left_fitx, right_fitx, ploty = fit_polynomial(warped)
    #cv2.imshow("warped", warped*255)
    #cv2.waitKey()
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    warped3 = np.dstack((warped, warped, warped)).astype(np.uint8)
    warped3_result = cv2.addWeighted(warped3, 1, color_warp, 0.3, 0)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    count = count +1
    cv2.imwrite("output_images_project/binary" + str(count) +".jpg", binary_combined*255)
    #cv2.imwrite("output_images_project/binary_warped" + str(count) +".jpg", warped*255)

    center_offset = (((LeftLane.best_fit[0] * 720 ** 2 + LeftLane.best_fit[1] * 720 + LeftLane.best_fit[2]) + (
            RightLane.best_fit[0] * 720 ** 2 + RightLane.best_fit[1] * 720 + RightLane.best_fit[2])) / 2 - 640) * parameter.xm_per_pix

    # Create merged output image
    ## This layout was introduces in https://chatbotslife.com/advanced-lane-line-project-7635ddca1960
    img_out = np.zeros((576, 1280, 3), dtype=np.uint8)

    img_out[0:576, 0:1024, :] = cv2.resize(result, (1024, 576))
    # combined binary image
    img_out[0:288, 1024:1280, 0] = cv2.resize(binary_combined * 255, (256, 288))
    img_out[0:288, 1024:1280, 1] = cv2.resize(binary_combined * 255, (256, 288))
    img_out[0:288, 1024:1280, 2] = cv2.resize(binary_combined * 255, (256, 288))
    # warped bird eye view
    img_out[310:576, 1024:1280, :] = cv2.resize(warped3_result, (256, 266))

    # Write curvature and center in image
    TextLeft = "Left curv: " + str(int(LeftLane.radius_of_curvature)) + " m"
    TextRight = "Right curv: " + str(int(RightLane.radius_of_curvature)) + " m"
    TextCenter = "Center offset: " + str(round(center_offset, 2)) + "m"

    fontScale = 1
    thickness = 2
    fontFace = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img_out, TextLeft, (130, 40), fontFace, fontScale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    cv2.putText(img_out, TextRight, (130, 70), fontFace, fontScale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    cv2.putText(img_out, TextCenter, (130, 100), fontFace, fontScale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    return img_out


mtx = []
dist = []
M = []
Minv = []
LeftLane = Line()
RightLane = Line()
center_offset = 0
count = 0

# main function for processing videos
def main():
    global mtx, dist, M, Minv
    mtx, dist = helper.calibrate_camera()
    M, Minv = helper.calculate_warp_parameter()
    output = 'test_videos_output/project_video_v2.mp4'
    clip1 = VideoFileClip("project_video.mp4")#.subclip(22, 30)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(output, audio=False)

# #main function for processing test images
# def main():
#     global mtx, dist, M, Minv
#     mtx, dist = calibrate_camera()
#     M, Minv = calculate_warp_parameter()
#
#     test_images = os.listdir("test_images/")
#     for fname in test_images:
#         print(fname)
#         img1 = cv2.imread("test_images/"+fname)
#         undist = cv2.undistort(img1, mtx, dist, None, mtx)
#         cv2.imwrite("output_images_test/" + fname + "_undist.jpg", undist)
#         undist_warped = cv2.warpPerspective(undist, M, img1.shape[1::-1], flags=cv2.INTER_LINEAR)
#         cv2.imwrite("output_images_test/" + fname + "_undist_warped.jpg", undist_warped)
#         binary_combined = create_combined_binary(undist)
#         cv2.imwrite("output_images_test/" + fname + "_binary.jpg", binary_combined)
#
#         binary_combined_warped = cv2.warpPerspective(binary_combined, M, img1.shape[1::-1], flags=cv2.INTER_LINEAR)
#         cv2.imwrite("output_images_test/" + fname + "_binary_warped.jpg", binary_combined_warped)
#
#         left_fitx, right_fitx, ploty = fit_polynomial(binary_combined_warped)
#
#         warp_zero = np.zeros_like(binary_combined_warped).astype(np.uint8)
#         color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
#
#         # Recast the x and y points into usable format for cv2.fillPoly()
#         pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
#         pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
#         pts = np.hstack((pts_left, pts_right))
#         # Draw the lane onto the warped blank image
#         cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
#
#         # Warp the blank back to original image space using inverse perspective matrix (Minv)
#         newwarp = cv2.warpPerspective(color_warp, Minv, (binary_combined_warped.shape[1], binary_combined_warped.shape[0]))
#         # Combine the result with the original image
#         result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
#
#         plt.imshow(binary_combined_warped)
#         plt.show()
#         cv2.imwrite("output_images_test/" + fname + "_final.jpg", result)
#         # cv2.imshow("afterwarp", result3)


if __name__ == "__main__":
    main()
