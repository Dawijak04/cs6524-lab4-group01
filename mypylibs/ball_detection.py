
import cv2 as cv
import numpy as np


def detect_ball(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Orange colour range
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([25, 255, 255])

    mask = cv.inRange(hsv, lower_orange, upper_orange)

    # Remove small areas of noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(
        mask, cv.MORPH_OPEN, kernel
    )

    contours, _ = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, mask

    # Find largest orange region
    contour = max(contours, key=cv.contourArea)

    if cv.contourArea(contour) < 100:
        return None, mask

    (x, y), radius = cv.minEnclosingCircle(contour)

    return (int(x), int(y), int(radius)), mask
