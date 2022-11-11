import cv2
import numpy as np

from ball_tracker.tracker_by_color import TrackerByColor


def main():
    param_for_tracking_by_color = np.load("bin/param_for_tracking_by_color.npz", allow_pickle=True)
    min_hsv = param_for_tracking_by_color["min_hsv"]  # [0, 130, 100]  # [150, 50, 200]
    max_hsv = param_for_tracking_by_color["max_hsv"]  # [88, 165, 255]  # [180, 255, 255]
    threshold = float(param_for_tracking_by_color["threshold"])  # 50

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

        cv2.imshow("image", frame)

        if do_save_video:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        elif cv2.waitKey(1) & 0xFF == ord("s"):
            frame = tracker.draw_trajectory(frame)
            do_save_video = True
            print("Saving video")
        elif cv2.waitKey(1) & 0xFF == ord("f"):
            pass

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
