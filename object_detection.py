import cv2
import numpy as np

def detect_contour(path):
    src = cv2.imread(path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    retval, bw = cv2.threshold(gray, 50, 225, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierachy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    detect_count = 0
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 1e2 or 1e5 < area:
            continue

        if len(contours[i]) > 0:
            rect = contours[i]
            x, y, w, h = cv2.boundingRect(rect)
            cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite("./results_img/" + str(detect_count) + ".jpg", src[y:y + h, x:x + w])
            detect_count += 1

    cv2.imshow("output", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    num = input()
    detect_contour("./img/test" + num + ".jpg")
