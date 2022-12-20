import cv2

from ball_tracker.tracker_by_color import TrackerByColor
from analysis_tools.track_bar_for_tracking_by_color import TrackBarForTrackingByColor
from const.pkg_path import (
    BASE_VIDEO_DIR_HOME,
    HSV_PARAM_DIR_HOME,
)

def main():
    PROJECT_NAME = "Design_and_Dragting_No3"
    BASE_VIDEO_PATH = f"{BASE_VIDEO_DIR_HOME}\\{PROJECT_NAME}\\sample.mp4"
    SAVE_HSV_PARAM_PATH = f"{HSV_PARAM_DIR_HOME}\\ball_hsv_param.npz"

    MIN_HSV = [0, 180, 100]  # [150, 50, 200]
    MAX_HSV = [255, 255, 255]  # [180, 255, 255]
    THRESHOLD = 50

    cap = cv2.VideoCapture(BASE_VIDEO_PATH)
    tracker = TrackerByColor(MIN_HSV, MAX_HSV, THRESHOLD)
    track_bar = TrackBarForTrackingByColor(MIN_HSV, MAX_HSV, THRESHOLD)

    while True:
        _, frame = cap.read()

        tracker.set_param_for_tracking(*track_bar.param_for_tracking_by_color)
        frame = tracker.generate_mask(frame)

        cv2.imshow("image", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        elif cv2.waitKey(1) & 0xFF == ord("s"):
            track_bar.save_param_for_tracking_by_color(SAVE_HSV_PARAM_PATH)
            print("The param is saved.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
