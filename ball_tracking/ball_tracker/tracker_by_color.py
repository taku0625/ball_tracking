from .tracker_base import TrackerBase
from analysis_tools.hsv_processor import HSVProcessor


class TrackerByColor(TrackerBase):
    def __init__(self, min_hsv, max_hsv, threshold):
        super().__init__()
        self._hsv_processor = HSVProcessor(min_hsv, max_hsv, threshold)

    def find_outline(self, image):
        return self._hsv_processor.find_outline_of_circle(image)

    def draw_outline(self, image, outline_info):
        return self._hsv_processor.draw_outline_of_circle(image, outline_info)