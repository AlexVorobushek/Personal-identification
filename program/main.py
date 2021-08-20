import cv2
import numpy as np

cap = cv2.VideoCapture(1)


def update_image(img, data=(1, 4, 2)):
    blur, c, k = data
    # img = cv2.GaussianBlur(img, (blur, blur), 0)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # return img
    for i in range(k):
        l, a, b = cv2.split(img)
        l2 = cv2.createCLAHE(clipLimit=3., tileGridSize=(c, c)).apply(l)
        img = cv2.merge((l2, a, b))
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)


while cv2.waitKey(10) != 32:
    frame = cap.read()[1]
    cv2.imshow('img', frame)
    cv2.imshow('updated', update_image(frame))

cap.release()
cv2.destroyAllWindows()
