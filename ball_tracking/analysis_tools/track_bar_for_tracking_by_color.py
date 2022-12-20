import cv2
import numpy as np

from const.hsv_idx import HUE_IDX, SATURATION_IDX, VALUE_IDX


class TrackBarForTrackingByColor:
    def __init__(self, initial_min_hsv, initial_max_hsv, initial_threshold):
        def nothing(x):
            pass

        cv2.namedWindow("track_bar", cv2.WINDOW_NORMAL)

        cv2.createTrackbar("minH", "track_bar", initial_min_hsv[HUE_IDX], 255, nothing)
        cv2.createTrackbar("minS", "track_bar", initial_min_hsv[SATURATION_IDX], 255, nothing)
        cv2.createTrackbar("minV", "track_bar", initial_min_hsv[VALUE_IDX], 255, nothing)

        cv2.createTrackbar("maxH", "track_bar", initial_max_hsv[HUE_IDX], 255, nothing)
        cv2.createTrackbar("maxS", "track_bar", initial_max_hsv[SATURATION_IDX], 255, nothing)
        cv2.createTrackbar("maxV", "track_bar", initial_max_hsv[VALUE_IDX], 255, nothing)

        cv2.createTrackbar("threshold", "track_bar", initial_threshold, 100, nothing)

    @property
    def param_for_tracking_by_color(self):
        self.__set_param_for_tracking_by_color_to_track_bar_value()
        return self._min_hsv, self._max_hsv, self._threshold

    @property
    def min_hsv(self):
        self.__set_min_hsv_to_track_bar_value()
        return self._min_hsv

    @property
    def max_hsv(self):
        self.__set_max_hsv_to_track_bar_value()
        return self._max_hsv

    @property
    def threshold(self):
        self.__set_threshold_to_track_bar_value()
        return self._threshold

    def __set_param_for_tracking_by_color_to_track_bar_value(self):
        self.__set_min_hsv_to_track_bar_value()
        self.__set_max_hsv_to_track_bar_value()
        self.__set_threshold_to_track_bar_value()

    def __set_min_hsv_to_track_bar_value(self):
        self._min_hsv = (
            cv2.getTrackbarPos("minH", "track_bar"),
            cv2.getTrackbarPos("minS", "track_bar"),
            cv2.getTrackbarPos("minV", "track_bar"),
        )

    def __set_max_hsv_to_track_bar_value(self):
        self._max_hsv = (
            cv2.getTrackbarPos("maxH", "track_bar"),
            cv2.getTrackbarPos("maxS", "track_bar"),
            cv2.getTrackbarPos("maxV", "track_bar"),
        )

    def __set_threshold_to_track_bar_value(self):
        self._threshold = cv2.getTrackbarPos("threshold", "track_bar")

    def save_param_for_tracking_by_color(self, save_path):
        self.__set_param_for_tracking_by_color_to_track_bar_value()
        np.savez(
            save_path,
            min_hsv=self._min_hsv,
            max_hsv=self._max_hsv,
            threshold=self._threshold,
        )
        print("This param for tracking by color was saved.")
