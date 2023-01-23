import cv2
import glob
import numpy as np
import os

from ball_tracker.tracker_by_color import TrackerByColor
from const.pkg_path import (
    BASE_VIDEO_DIR_HOME,
    HSV_PARAM_DIR_HOME,
    PROCESSED_VIDEO_DIR_HOME,
    TRAJECTORY_IMAGE_DIR_HOME,
    TRAJECTORY_POINTS_DIR_HOME,
    TRAJECTORY_POINTS_IMG_DIR_HOME,
)
from const.design_and_drafting_no3 import SUPPORT_WIDTH, SUPPORT_HEIGHT
import matplotlib.pyplot as plt


class TrackBall:
    def __init__(self, project_name, hsv_param_file_name):
        self._project_name = project_name
        BASE_VIDEO_DIR = f"{BASE_VIDEO_DIR_HOME}\\{self._project_name}"
        HSV_PARAM_PATH = f"{HSV_PARAM_DIR_HOME}\\{hsv_param_file_name}"

        ball_hsv_param = np.load(HSV_PARAM_PATH, allow_pickle=True)
        self._ball_min_hsv = ball_hsv_param["min_hsv"]  # [0, 130, 100]  # [150, 50, 200]
        self._ball_max_hsv = ball_hsv_param["max_hsv"]  # [88, 165, 255]  # [180, 255, 255]
        self._ball_binary_threshold = float(ball_hsv_param["binary_threshold"])  # 50
        
        base_video_path_list = glob.glob(f"{BASE_VIDEO_DIR}\\*")
        for base_video_path in base_video_path_list:
            self.__prepare_track(base_video_path)

            first_loop = True
            frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for _ in range(frame_count):
                _, frame = self._cap.read()

                if first_loop:
                    self._x1, self._y1, self._roi_width, self._roi_height = cv2.selectROI("image", frame)
                    cv2.resizeWindow("cut_image", 1000, int(1000 * self._roi_height / self._roi_width))
                    first_loop = False

                self.__track(frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            self.__finish_track(frame)

    def __prepare_track(self, base_video_path):
        PROCESSED_VIDEO_DIR = f"{PROCESSED_VIDEO_DIR_HOME}\\{self._project_name}"
        self._base_video_basename = os.path.splitext(os.path.basename(base_video_path))[0]
        # get video info
        self._cap = cv2.VideoCapture(base_video_path)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self._cap.get(cv2.CAP_PROP_FPS))

        self._base_tracker = TrackerByColor(self._ball_min_hsv, self._ball_max_hsv, self._ball_binary_threshold)
        self._roi_tracker = TrackerByColor(self._ball_min_hsv, self._ball_max_hsv, self._ball_binary_threshold)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._writer = cv2.VideoWriter(f"{PROCESSED_VIDEO_DIR}\\{self._base_video_basename}.avi", fourcc, fps, (width, height))

        cv2.namedWindow("cut_image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("cut_image", 1000, int(1000 * height / width))

    def __track(self, frame):
        roi_frame = frame[self._y1 : self._y1 + self._roi_height, self._x1 : self._x1 + self._roi_width]
        roi_frame = self._roi_tracker.draw_trajectory(roi_frame)
        frame = self._base_tracker.draw_trajectory(frame)
        cv2.imshow("image",frame)
        cv2.imshow("cut_image",roi_frame)
        self._writer.write(frame)

    def __finish_track(self, frame):
        TRAJECTORY_IMAGE_DIR = f"{TRAJECTORY_IMAGE_DIR_HOME}\\{self._project_name}"
        TRAJECTORY_POINTS_DIR = f"{TRAJECTORY_POINTS_DIR_HOME}\\{self._project_name}"
        TRAJECTORY_POINTS_IMG_DIR = f"{TRAJECTORY_POINTS_IMG_DIR_HOME}\\{self._project_name}"

        cv2.imwrite(f"{TRAJECTORY_IMAGE_DIR}\\{self._base_video_basename}.jpeg", frame)
        roi_trajectory_points = np.array(
            [
                (x / self._roi_width * SUPPORT_WIDTH, - y / self._roi_height * SUPPORT_HEIGHT + SUPPORT_HEIGHT)
                for x, y in self._roi_tracker.trajectory_points
            ]
        )
        np.save(f"{TRAJECTORY_POINTS_DIR}\\{self._base_video_basename}.npy", roi_trajectory_points)

        self._base_tracker.reset_trajectory()
        self._roi_tracker.reset_trajectory()

        self._cap.release()
        self._writer.release()
        cv2.destroyAllWindows()

        plt.scatter(roi_trajectory_points[:, 0], roi_trajectory_points[:, 1])
        plt.savefig(f"{TRAJECTORY_POINTS_IMG_DIR}\\{self._base_video_basename}.png")
        plt.show()


if __name__ == "__main__":
    PROJECT_NAME = "Design_and_Dragting_No3"
    HSV_PARAM_FILE_NAME = "ball_hsv_param.npz"
    track_ball = TrackBall(PROJECT_NAME, HSV_PARAM_FILE_NAME)
