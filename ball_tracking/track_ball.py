import cv2
import numpy as np

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
    HSV_PARAM_PATH = f"{HSV_PARAM_DIR_HOME}\\ball_hsv_param.npz"

    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(
        "bin/draw_trajectory.avi",
        fourcc,
        fps,
        (width, height)
    )

    tracker = TrackerByColor(min_hsv, max_hsv, threshold)

    do_save_video = False

    while True:
        _, frame = cap.read()

        if do_save_video:
            frame = tracker.draw_trajectory(frame)
            writer.write(frame)
        cv2.imshow("image", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        elif cv2.waitKey(1) & 0xFF == ord("s"):
            do_save_video = True
            print("Saving video")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
