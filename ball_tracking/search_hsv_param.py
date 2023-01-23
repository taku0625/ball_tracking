import cv2

from analysis_tools.hsv_processor import HSVProcessor
from analysis_tools.hsv_track_bar import HSVTrackBar
from const.pkg_path import BASE_VIDEO_DIR_HOME, HSV_PARAM_DIR_HOME


class SearchHSVParam:
    def __init__(self, project_name, base_video_file_name, save_hsv_param_file_name):
        BASE_VIDEO_PATH = f"{BASE_VIDEO_DIR_HOME}\\{project_name}\\{base_video_file_name}"
        SAVE_HSV_PARAM_PATH = f"{HSV_PARAM_DIR_HOME}\\{save_hsv_param_file_name}"

        MIN_HSV = [0, 180, 100]  # [150, 50, 200]
        MAX_HSV = [255, 255, 255]  # [180, 255, 255]
        BINARY_THRESHOLD = 50

        self._cap = cv2.VideoCapture(BASE_VIDEO_PATH)
        self._hsv_processor = HSVProcessor(MIN_HSV, MAX_HSV, BINARY_THRESHOLD)
        self._track_bar = HSVTrackBar(MIN_HSV, MAX_HSV, BINARY_THRESHOLD)
        
        while True:
            grabbed, frame = self._cap.read()
            if not grabbed:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            self.__hsv_process(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            elif cv2.waitKey(1) & 0xFF == ord("s"):
                self._track_bar.save_track_bar_values(SAVE_HSV_PARAM_PATH)
                print("The param is saved.")
        self.__finish()

    def __hsv_process(self, frame):
        # pipeline between track_bar and hsv_processor
        min_hsv = self._track_bar.get_min_hsv()
        max_hsv = self._track_bar.get_max_hsv()
        binary_threshold = self._track_bar.get_binary_threshold()
        self._hsv_processor.min_hsv = min_hsv
        self._hsv_processor.max_hsv = max_hsv
        self._hsv_processor.binary_threshold = binary_threshold

        mask_frame = self._hsv_processor.generate_mask(frame)
        out_line_info = self._hsv_processor.find_outline_of_rect(frame)
        cut_frame = self._hsv_processor.draw_outline_of_rect(frame, out_line_info)

        cv2.imshow("image", mask_frame)
        cv2.imshow("cut_image", cut_frame)

    def __finish(self):
        self._cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    PROJECT_NAME = "Design_and_Dragting_No3"
    BASE_VIDEO_FILE_NAME = "competitive.mp4"
    SAVE_HSV_PARAM_FILE_NAME = "ball_hsv_param.npz"
    search_hsv_param = SearchHSVParam(PROJECT_NAME, BASE_VIDEO_FILE_NAME, SAVE_HSV_PARAM_FILE_NAME)
