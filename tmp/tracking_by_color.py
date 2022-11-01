import cv2
import numpy as np


def main(): 
    cap = cv2.VideoCapture(1) # 任意の動画
    
    while True:
        _, frame = cap.read()
    
        #マスク画像取得
        def getMask(l, u):
            # HSVに変換
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
            lower = np.array(l)
            upper = np.array(u)
            if lower[0] >= 0:
                #色相が正の値のとき、赤以外のマスク
                mask = cv2.inRange(hsv, lower, upper)
            else:
                #色相が負の値のとき、赤用マスク
                h = hsv[:, :, 0]
                s = hsv[:, :, 1]
                v = hsv[:, :, 2]
                mask = np.zeros(h.shape, dtype=np.uint8)
                mask[((h < lower[0]*-1) | h > upper[0]) & (s > lower[1]) & (s < upper[1]) & (v > lower[2]) & (v < upper[2])] = 255
    
            return cv2.bitwise_and(frame,frame, mask= mask)
    
        # 輪郭取得
        def getContours(img,t,r):
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, thresh = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
            (
                contours, hierarchy
            )= cv2.findContours(
                thresh, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            # 小さい輪郭は誤検出として削除する
            contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))
            
            """# 輪郭を描画する。
            cv2.drawContours(frame, contours, -1, color=(0, 0, 255), thickness=2)

            # 輪郭の点の描画
            for contour in contours:
                for point in contour:
                    cv2.circle(frame, point[0], 3, (0, 255, 0), -1)"""

            # 一番大きい輪郭を抽出
            contours.sort(key=cv2.contourArea, reverse=True)
            
            #一つ以上検出
            if len(contours) > 0:
                for cnt in contours:
                    # 最小外接円を描く
                    (x,y), radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x),int(y))
                    radius = int(radius)
    
                    if radius > r:
                        radius_frame = cv2.circle(frame,center,radius,(0,255,0),2)
                return radius_frame
            else:
                return frame
    
        # マスクの最小HSVと最大HSVを指定　ただし赤の場合は最小Hを負の値にする（独自ルール）
        # 青マスク H110～150、S45～255、V100～255
        res_blue = getMask([110,45,100], [150,255,255])
    
        # 赤マスク H0～10または170～180、S50～255、V200～255
        res_red = getMask([150,50,200], [180,255,255])
    
        #輪郭取得
        contours_frame = getContours(res_blue, 50, 0) # (画像, 明度閾値, 最小半径)
    
    
        # 再生
        cv2.imshow('video', frame)
    
        k = cv2.waitKey(25) & 0xFF
        #Q で終了
        if k == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()