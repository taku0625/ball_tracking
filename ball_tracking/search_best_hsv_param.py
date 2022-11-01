import asyncio
import aioconsole
import cv2

from ball_tracker.tracker_by_color import TrackerByColor
from analysis_tools.track_bar_for_hsv import TrackBarForHSV




def main():
    MIN_HSV = [0,180,100]    # [150, 50, 200]
    MAX_HSV = [255,255,255]   # [180, 255, 255]
    THRESHOLD = 50
    
    do_save = None
    
    cap = cv2.VideoCapture(0)
    tracker = TrackerByColor(MIN_HSV, MAX_HSV, THRESHOLD)
    track_bar = TrackBarForHSV(MIN_HSV, MAX_HSV, THRESHOLD)

    while True:
        # do_save = await aioconsole.ainput("Do you save this value? y/n")
        _, frame = cap.read()
        print("----")
        
        tracker.set_param(*track_bar.track_bar_poses)
        frame = tracker.generate_mask(frame)

        cv2.imshow('track_bar', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        
        if do_save == "y":
            track_bar.save_hsv_param()
        elif do_save == "n":
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()