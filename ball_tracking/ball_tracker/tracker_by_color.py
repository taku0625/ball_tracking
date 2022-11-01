import cv2
import numpy as np

from .tracker_base import TrackerBase
from const.hsv_idx import HUE_IDX, SATURATION_IDX, VALUE_IDX


class TrackerByColor(TrackerBase):
    def __init__(self, min_hsv, max_hsv, threshold):
        super().__init__()
        self._min_hsv = np.array(min_hsv)
        self._max_hsv = np.array(max_hsv)
        self._threshold = threshold

    def set_param(self, min_hsv, max_hsv, threshold):
        self._min_hsv = np.array(min_hsv)
        self._max_hsv = np.array(max_hsv)
        self._threshold = threshold

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
        _, binary = cv2.threshold(
            gray_image,
            self._threshold,
            255,
            cv2.THRESH_BINARY,
        )
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(filter(lambda x: cv2.contourArea(x) > 100, contours))

    def draw_contours(self, image):
        contours = self.__find_contours(image)
        cv2.drawContours(image, contours, -1, color=(0, 0, 255), thickness=2)

        for contour in contours:
            for point in contour:
                cv2.circle(image, point[0], 3, (0, 255, 0), -1)

        return image

    def _find_outline(self, image):
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

    def draw_outline(self, image, outline_info):
        (center, radius) = outline_info
        cv2.circle(image, center, radius, (0, 255, 0), 2)
        return image
