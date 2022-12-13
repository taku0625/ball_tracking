import cv2
import numpy as np

from .tracker_base import TrackerBase


class TrackerByColor(TrackerBase):
    def __init__(self, image, tracker_type="KCF"):
        super().__init__()
        self.__create_tracker(tracker_type)
        roi = cv2.selectROI(image)
        _ = self._tracker.init(image, roi)

    def __create_tracker(self, tracker_type):
        if tracker_type == "Boosting":
            self._tracker = cv2.legacy.TrackerBoosting_create()
        elif tracker_type == "MIL":
            self._tracker = cv2.legacy.TrackerMIL_create()
        elif tracker_type == "KCF":
            self._tracker = cv2.legacy.TrackerKCF_create()
        elif tracker_type == "TLD":
            self._tracker = cv2.legacy.TrackerTLD_create()
        elif tracker_type == "MedianFlow":
            self._tracker = cv2.legacy.TrackerMedianFlow_create()
        else:
            raise ValueError("The tracker type does not exist.")

    def _find_outline(self, image):
        success, roi = self._tracker.update(image)
        (x, y, w, h) = tuple(map(int, roi))
        if success:
            p1 = (x, y)
            p2 = (x + w, y + h)
            center = (p1 + p2) / 2
        else:
            p1 = None
            p2 = None
            center = None

        return (center, p1, p2)

    def draw_outline(self, image, outline_info):
        (_, p1, p2) = outline_info
        return cv2.rectangle(image, p1, p2, (0, 255, 0), 3)
