import cv2

from ball_tracker.tracker_by_color import TrackerByColor


def main():
    MIN_HSV = [0,130,100]    # [150, 50, 200]
    MAX_HSV = [88,165,255]    # [180, 255, 255]
    THRESHOLD = 50
    
    cap = cv2.VideoCapture(1)
    tracker = TrackerByColor(MIN_HSV, MAX_HSV, THRESHOLD)
    
    while True:
        _, frame = cap.read()
        
        frame = tracker.draw_trajectory(frame)
        
        cv2.imshow("image", frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()