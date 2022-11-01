import cv2

from ball_tracker.tracker_by_color import TrackerByColor
from analysis_tools.track_bar_for_hsv import TrackBarForHSV


def main():
    MIN_HSV = [0, 180, 100]  # [150, 50, 200]
    MAX_HSV = [255, 255, 255]  # [180, 255, 255]
    THRESHOLD = 50

    cap = cv2.VideoCapture(0)
    tracker = TrackerByColor(MIN_HSV, MAX_HSV, THRESHOLD)
    track_bar = TrackBarForHSV(MIN_HSV, MAX_HSV, THRESHOLD)

    while True:
        _, frame = cap.read()

        tracker.set_param(*track_bar.track_bar_positions)
        frame = tracker.generate_mask(frame)

        cv2.imshow("image", frame)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
        elif cv2.waitKey(30) & 0xFF == ord("s"):
            track_bar.save_hsv_param()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
