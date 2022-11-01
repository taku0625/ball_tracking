from abc import ABCMeta, abstractmethod
import cv2


class TrackerBase(metaclass=ABCMeta):
    def __init__(self):
        self._start_draw_trajectory = True

    @abstractmethod
    def _find_outline(self, image):
        """_summary_

        Args:
            image (_type_):
        Return:
            tuple: Index0 must always be the center of obeject's outline.
        Raises:
            NotImplementedError:
        """
        raise NotImplementedError

    @abstractmethod
    def draw_outline(self, image, outline_info):
        raise NotImplementedError

    def draw_trajectory(self, image, draw_outline=True):
        outline_info = self._find_outline(image)
        center = outline_info[0]

        if self._start_draw_trajectory:
            self._start_draw_trajectory = False
            self._trajectory_points = []

        if type(center) is tuple:
            self._trajectory_points.append(center)
            if draw_outline:
                image = self.draw_outline(image, outline_info)

        for point in self._trajectory_points:
            cv2.circle(image, point, 5, (0, 0, 255), -1)

        return image

    @property
    def trajectory_points(self):
        return self._trajectory_points

    def reset_trajectory(self):
        self._start_draw_trajectory = True
        self._trajectory_points = []
