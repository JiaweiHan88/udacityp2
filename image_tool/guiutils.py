import cv2
import numpy as np


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sbinary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return sbinary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobel = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    sbinary = np.zeros_like(abs_sobel)
    sbinary[(abs_sobel >= thresh[0]) & (abs_sobel <= thresh[1])] = 1
    return sbinary


class LaneFinder:
    def __init__(self, image, filter_size=1, threshold1x=0, threshold2x=0, threshold1y=0, threshold2y=0):
        self.image = image
        self._filter_size = 11
        self._threshold1x = 2
        self._threshold2x = 50
        self._threshold1y = 2
        self._threshold2y = 50
        self._threshold1mag = 20
        self._threshold2mag = 50
        self._threshold1dir = 6
        self._threshold2dir = 10

        def onchangeThreshold1x(pos):
            self._threshold1x = pos
            self._render()

        def onchangeThreshold2x(pos):
            self._threshold2x = pos
            self._render()

        def onchangeThreshold1y(pos):
            self._threshold1y = pos
            self._render()

        def onchangeThreshold2y(pos):
            self._threshold2y = pos
            self._render()

        def onchangeThreshold1mag(pos):
            self._threshold1mag = pos
            self._render()

        def onchangeThreshold2mag(pos):
            self._threshold2mag = pos
            self._render()

        def onchangeThreshold1dir(pos):
            self._threshold1dir = pos
            self._render()

        def onchangeThreshold2dir(pos):
            self._threshold2dir = pos
            self._render()

        def onchangeFilterSize(pos):
            self._filter_size = pos
            self._filter_size += (self._filter_size + 1) % 2
            self._render()

        cv2.namedWindow('Lanes')
        cv2.resizeWindow('Lanes', 1000, 500)
        cv2.createTrackbar('threshold1x', 'Lanes', self._threshold1x, 255, onchangeThreshold1x)
        cv2.createTrackbar('threshold2x', 'Lanes', self._threshold2x, 255, onchangeThreshold2x)
        cv2.createTrackbar('threshold1y', 'Lanes', self._threshold1y, 255, onchangeThreshold1y)
        cv2.createTrackbar('threshold2y', 'Lanes', self._threshold2y, 255, onchangeThreshold2y)
        cv2.createTrackbar('threshold1mag', 'Lanes', self._threshold1mag, 255, onchangeThreshold1mag)
        cv2.createTrackbar('threshold2mag', 'Lanes', self._threshold2mag, 150, onchangeThreshold2mag)
        cv2.createTrackbar('threshold1dir', 'Lanes', self._threshold1dir, 255, onchangeThreshold1dir)
        cv2.createTrackbar('threshold2dir', 'Lanes', self._threshold2dir, 150, onchangeThreshold2dir)
        cv2.createTrackbar('filter_size', 'Lanes', self._filter_size, 20, onchangeFilterSize)

        self._render()

        print("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow('gradx')
        cv2.destroyWindow('grady')

    def threshold1x(self):
        return self._threshold1x

    def threshold2x(self):
        return self._threshold2x

    def threshold1y(self):
        return self._threshold1y

    def threshold2y(self):
        return self._threshold2y

    def filterSize(self):
        return self._filter_size

    def edgeImage(self):
        return self._edge_img

    def smoothedImage(self):
        return self._smoothed_img

    def _render(self):
        img = np.copy(self.image)
        # Convert to HLS color space and separate the V channel
        r_channel = img[:, :, 2]
        g_channel = img[:, :, 1]
        b_channel = img[:, :, 0]

        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        gradx = abs_sobel_thresh(l_channel, orient='x', sobel_kernel=self._filter_size,
                                 thresh=(self._threshold1x, self._threshold2x))
        grady = abs_sobel_thresh(l_channel, orient='y', sobel_kernel=self._filter_size,
                                 thresh=(self._threshold1y, self._threshold2y))
        mag_binary = mag_thresh(l_channel, sobel_kernel=self._filter_size, mag_thresh=(self._threshold1mag, self._threshold2mag))
        dir_binary = dir_threshold(l_channel, sobel_kernel=self._filter_size, thresh=(self._threshold1dir/10, self._threshold2dir/10))

        # Threshold color channel
        s_binary2 = np.zeros_like(s_channel)
        s_binary2[(s_channel >= 150) & (s_channel <= 210)] = 1

        r_binary2 = np.zeros_like(r_channel)
        r_binary2[(r_channel >= 220) & (r_channel <= 255)] = 1

        r_binarymin = np.zeros_like(r_channel)
        r_binarymin[(r_channel >= 00) & (r_channel <= 255)] = 1

        combined = np.zeros_like(dir_binary)
        combined[(((gradx == 1) & (grady == 1) & (dir_binary == 1)& (mag_binary == 1) & (r_binarymin == 1))| ((r_binary2 == 1)))] = 255

        #combined[(((gradx == 1) & (grady == 1) & (dir_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1)))] = 1

        cv2.imshow("gradx", gradx * 255)
        cv2.imshow("grady", grady * 255)
        cv2.imshow("mag", mag_binary * 255)
        cv2.imshow("dir", dir_binary)
        cv2.imshow("combined", combined)
        cv2.imshow("g_channel", g_channel)
        cv2.imshow("l_channel", l_channel)
