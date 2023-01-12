import cv2

from analysis_tools.hsv_processor import HSVProcessor
from analysis_tools.hsv_track_bar import HSVTrackBar
from const.pkg_path import BASE_VIDEO_DIR_HOME, HSV_PARAM_DIR_HOME


def main():
    PROJECT_NAME = "Design_and_Dragting_No3"
    BASE_VIDEO_PATH = f"{BASE_VIDEO_DIR_HOME}\\{PROJECT_NAME}\\competitive.mp4"
    SAVE_HSV_PARAM_PATH = f"{HSV_PARAM_DIR_HOME}\\ball_hsv_param.npz"

    MIN_HSV = [0, 180, 100]  # [150, 50, 200]
    MAX_HSV = [255, 255, 255]  # [180, 255, 255]
    BINARY_THRESHOLD = 50

    cap = cv2.VideoCapture(BASE_VIDEO_PATH)
    hsv_processor = HSVProcessor(MIN_HSV, MAX_HSV, BINARY_THRESHOLD)
    track_bar = HSVTrackBar(MIN_HSV, MAX_HSV, BINARY_THRESHOLD)

    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # pipeline between track_bar and hsv_processor
        min_hsv = track_bar.get_min_hsv()
        max_hsv = track_bar.get_max_hsv()
        binary_threshold = track_bar.get_binary_threshold()
        hsv_processor.min_hsv = min_hsv
        hsv_processor.max_hsv = max_hsv
        hsv_processor.binary_threshold = binary_threshold

        mask_frame = hsv_processor.generate_mask(frame)
        out_line_info = hsv_processor.find_outline_of_rect(frame)
        cut_frame = hsv_processor.draw_outline_of_rect(frame, out_line_info)

        cv2.imshow("image", mask_frame)
        cv2.imshow("cut_image", cut_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        elif cv2.waitKey(1) & 0xFF == ord("s"):
            track_bar.save_track_bar_values(SAVE_HSV_PARAM_PATH)
            print("The param is saved.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
