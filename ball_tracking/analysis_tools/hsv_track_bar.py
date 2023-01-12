import cv2
import numpy as np

from const.hsv_idx import HUE_IDX, SATURATION_IDX, VALUE_IDX


class HSVTrackBar:
    def __init__(self, initial_min_hsv, initial_max_hsv, initial_binary_threshold):
        def nothing(x):
            pass

        cv2.namedWindow("track_bar", cv2.WINDOW_NORMAL)

        cv2.createTrackbar("minH", "track_bar", initial_min_hsv[HUE_IDX], 255, nothing)
        cv2.createTrackbar("minS", "track_bar", initial_min_hsv[SATURATION_IDX], 255, nothing)
        cv2.createTrackbar("minV", "track_bar", initial_min_hsv[VALUE_IDX], 255, nothing)

        cv2.createTrackbar("maxH", "track_bar", initial_max_hsv[HUE_IDX], 255, nothing)
        cv2.createTrackbar("maxS", "track_bar", initial_max_hsv[SATURATION_IDX], 255, nothing)
        cv2.createTrackbar("maxV", "track_bar", initial_max_hsv[VALUE_IDX], 255, nothing)

        cv2.createTrackbar("binary_threshold", "track_bar", initial_binary_threshold, 100, nothing)

    def get_min_hsv(self):
        return(
            cv2.getTrackbarPos("minH", "track_bar"),
            cv2.getTrackbarPos("minS", "track_bar"),
            cv2.getTrackbarPos("minV", "track_bar"),
        )

    def get_max_hsv(self):
        return (
            cv2.getTrackbarPos("maxH", "track_bar"),
            cv2.getTrackbarPos("maxS", "track_bar"),
            cv2.getTrackbarPos("maxV", "track_bar"),
        )

    def get_binary_threshold(self):
        return cv2.getTrackbarPos("binary_threshold", "track_bar")

    def save_track_bar_values(self, save_path):
        min_hsv = self.get_min_hsv()
        max_hsv = self.get_max_hsv()
        binary_threshold = self.get_binary_threshold()

        np.savez(
            save_path,
            min_hsv=min_hsv,
            max_hsv=max_hsv,
            binary_threshold=binary_threshold,
        )

        print("This param for tracking by color was saved.")
