import cv2
import numpy as np

from const.hsv_idx import HUE_IDX, SATURATION_IDX, VALUE_IDX


class HSVProcessor:
    def __init__(self, min_hsv, max_hsv, binary_threshold):
        self.min_hsv = min_hsv
        self.max_hsv = max_hsv
        self.binary_threshold = binary_threshold

    @property
    def min_hsv(self):
        return self._min_hsv

    @property
    def max_hsv(self):
        return self._max_hsv

    @property
    def binary_threshold(self):
        return self._binary_threshold

    @min_hsv.setter
    def min_hsv(self, min_hsv):
        if self.__allow_set_hsv(min_hsv):
            self._min_hsv = np.array(min_hsv)
        else:
            raise ValueError("The min_hsv is not in the range.")

    @max_hsv.setter
    def max_hsv(self, max_hsv):
        if self.__allow_set_hsv(max_hsv):
            self._max_hsv = np.array(max_hsv)
        else:
            raise ValueError("The min_hsv is not in the range.")

    @binary_threshold.setter
    def binary_threshold(self, binary_threshold):
        if self.__allow_set_binary_threshold:
            self._binary_threshold = binary_threshold

    def __allow_set_hsv(self, hsv):
        is_in_range = [0 <= value <= 255 for value in hsv]
        return all(is_in_range)

    def __allow_set_binary_threshold(self, binary_threshold):
        return 0 <= binary_threshold <= 100

    def generate_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if self._min_hsv[HUE_IDX] >= 0:
            mask = cv2.inRange(hsv, self._min_hsv, self._max_hsv)
        else:
            hue = hsv[:, :, HUE_IDX]
            saturation = hsv[:, :, SATURATION_IDX]
            value = hsv[:, :, VALUE_IDX]
            mask = np.zeros(hue.shape, dtype=np.uint8)
            mask[
                ((hue < self._min_hsv[HUE_IDX] * -1) | hue > self._max_hsv[HUE_IDX])
                & (saturation > self._min_hsv[SATURATION_IDX])
                & (saturation < self._max_hsv[SATURATION_IDX])
                & (value > self._min_hsv[VALUE_IDX])
                & (value < self._max_hsv[VALUE_IDX])
            ] = 255

        return cv2.bitwise_and(image, image, mask=mask)

    def __find_contours(self, image):
        mask_image = self.generate_mask(image)
        gray_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray_image, self._binary_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(filter(lambda x: cv2.contourArea(x) > 100, contours))

    def draw_contours(self, image):
        contours = self.__find_contours(image)
        cv2.drawContours(image, contours, -1, color=(0, 0, 255), thickness=2)

        for contour in contours:
            for point in contour:
                cv2.circle(image, point[0], 3, (0, 255, 0), -1)

        return image

    def find_outline_of_circle(self, image):
        contours = self.__find_contours(image)
        contours.sort(key=cv2.contourArea, reverse=True)

        if len(contours) > 0:
            (x, y), radius = cv2.minEnclosingCircle(contours[-1])
            center = (int(x), int(y))
            radius = int(radius)
        else:
            center = None
            radius = None

        return (center, radius)

    def find_outline_of_rect(self, image):
        contours = self.__find_contours(image)
        contours.sort(key=cv2.contourArea, reverse=True)

        if len(contours) > 0:
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
        else:
            box = None

        return box

    def draw_outline_of_circle(self, image, outline_info):
        (center, radius) = outline_info
        cv2.circle(image, center, radius, (0, 255, 0), 2)
        return image

    def draw_outline_of_rect(self, image, outline_info):
        cv2.drawContours(image, [outline_info], 0, (0, 255, 0), 2)
        return image
