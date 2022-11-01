import cv2
import numpy as np

from ball_tracker.tracker_by_color import TrackerByColor


def main():
    param_for_tracking_by_color = np.load("bin/param_for_tracking_by_color.npz", allow_pickle=True)
    min_hsv = param_for_tracking_by_color["min_hsv"]  # [0, 130, 100]  # [150, 50, 200]
    max_hsv = param_for_tracking_by_color["max_hsv"]  # [88, 165, 255]  # [180, 255, 255]
    threshold = float(param_for_tracking_by_color["threshold"])  # 50

    cap = cv2.VideoCapture(0)
    tracker = TrackerByColor(min_hsv, max_hsv, threshold)

    while True:
        _, frame = cap.read()

        frame = tracker.draw_trajectory(frame)

        cv2.imshow("image", frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
