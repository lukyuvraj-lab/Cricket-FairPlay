import cv2
import numpy as np

class BallTracker:

def __init__(self):
    self.previous_center = None
    self.ball_detected = False
    self.frames_without_ball = 0

def detect_ball(self, frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect bright/white cricket ball
    lower = np.array([0, 0, 120])
    upper = np.array([180, 100, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_circle = None
    best_score = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 30 or area > 5000:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = (
            4 * np.pi * area
            / (perimeter * perimeter)
        )

        if circularity < 0.45:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)

        if radius < 3 or radius > 50:
            continue

        score = circularity * area

        if score > best_score:
            best_score = score
            best_circle = (
                int(x),
                int(y),
                int(radius)
            )

    return best_circle

def update(self, frame):

    ball = self.detect_ball(frame)

    if ball is not None:

        x, y, radius = ball

        self.previous_center = (x, y)
        self.ball_detected = True
        self.frames_without_ball = 0

        return ball

    self.frames_without_ball += 1

    if self.frames_without_ball > 10:
        self.ball_detected = False

    return None
