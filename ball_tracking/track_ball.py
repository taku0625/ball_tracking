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
)
from const.design_and_drafting_no3 import SUPPORT_WIDTH, SUPPORT_HEIGHT
import matplotlib.pyplot as plt

def main():
    PROJECT_NAME = "Design_and_Dragting_No3"
    BASE_VIDEO_DIR = f"{BASE_VIDEO_DIR_HOME}\\{PROJECT_NAME}"
    PROCESSED_VIDEO_DIR = f"{PROCESSED_VIDEO_DIR_HOME}\\{PROJECT_NAME}"
    TRAJECTORY_IMAGE_DIR = f"{TRAJECTORY_IMAGE_DIR_HOME}\\{PROJECT_NAME}"
    TRAJECTORY_POINTS_DIR = f"{TRAJECTORY_POINTS_DIR_HOME}\\{PROJECT_NAME}"
    TRAJECTORY_POINTS_IMG_DIR = f"{TRAJECTORY_POINTS_IMG_DIR_HOME}\\{PROJECT_NAME}"
    HSV_PARAM_PATH = f"{HSV_PARAM_DIR_HOME}\\ball_hsv_param.npz"

    ball_hsv_param = np.load(HSV_PARAM_PATH, allow_pickle=True)
    ball_min_hsv = ball_hsv_param["min_hsv"]  # [0, 130, 100]  # [150, 50, 200]
    ball_max_hsv = ball_hsv_param["max_hsv"]  # [88, 165, 255]  # [180, 255, 255]
    ball_binary_threshold = float(ball_hsv_param["binary_threshold"])  # 50
    
    base_video_path_list = glob.glob(f"{BASE_VIDEO_DIR}\\*")

    for base_video_path in base_video_path_list:
        base_video_basename = os.path.splitext(os.path.basename(base_video_path))[0]
        # get video info
        cap = cv2.VideoCapture(base_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"XVID")

        base_tracker = TrackerByColor(ball_min_hsv, ball_max_hsv, ball_binary_threshold)
        roi_tracker = TrackerByColor(ball_min_hsv, ball_max_hsv, ball_binary_threshold)

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", 1000, int(1000 * height / width))

        first_loop = True
        for _ in range(frame_count):
            _, frame = cap.read()

            if first_loop:
                x1, y1, roi_width, roi_height = cv2.selectROI("image", frame)
                writer = cv2.VideoWriter(
                    f"{PROCESSED_VIDEO_DIR}\\{base_video_basename}.avi", fourcc, fps, (width, height)
                )
                cv2.resizeWindow("image", 1000, int(1000 * roi_height / roi_width))
                first_loop = False

            roi_frame = frame[y1 : y1 + roi_height, x1 : x1 + roi_width]
            roi_frame = roi_tracker.draw_trajectory(roi_frame)
            frame = base_tracker.draw_trajectory(frame)
            writer.write(frame)

        cv2.imwrite(f"{TRAJECTORY_IMAGE_DIR}\\{base_video_basename}.jpeg", frame)
        roi_trajectory_points = np.array(
            [
                (x / roi_width * SUPPORT_WIDTH, - y / roi_height * SUPPORT_HEIGHT + SUPPORT_HEIGHT) 
                for x, y in roi_tracker.trajectory_points
            ]
        )
        np.save(f"{TRAJECTORY_POINTS_DIR}\\{base_video_basename}.npy", roi_trajectory_points)

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        plt.scatter(roi_trajectory_points[:, 0], roi_trajectory_points[:, 1])
        plt.savefig(f"{TRAJECTORY_POINTS_IMG_DIR}\\{base_video_basename}.png")
        plt.show()

if __name__ == "__main__":
    main()
