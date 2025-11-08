"""
Simple Dice Detector (7% Bigger Square Boxes — No Mask Display)
Author: Brent Knopp
University of Idaho
Class: CS 5553
Assignment: 7

Detects yellow dice using HSV thresholding, applies mild dilation,
and draws centered square bounding boxes (7% larger) around each die.
Only the final result is displayed.
"""

import cv2
import numpy as np

# --- PARAMETERS ---
MIN_DICE_AREA = 200
BOX_SCALE = 1.07   # 7% larger squares
LOWER_YELLOW = np.array([15, 60, 60])
UPPER_YELLOW = np.array([45, 255, 255])

# --- LOAD IMAGE ---
image = cv2.imread("test.png")
if image is None:
    raise FileNotFoundError("Error: Could not load 'test.png'")

# --- CONVERT TO HSV & CREATE MASK ---
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

# --- CLEAN + REGULAR DILATION ---
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove small noise
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill small gaps
mask = cv2.dilate(mask, kernel, iterations=1)           # mild dilation

# --- FIND CONTOURS ---
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- DRAW 7% BIGGER SQUARE BOXES ---
for c in contours:
    if cv2.contourArea(c) > MIN_DICE_AREA:
        x, y, w, h = cv2.boundingRect(c)

        # make box square and slightly larger
        side = int(max(w, h) * BOX_SCALE)
        cx, cy = x + w // 2, y + h // 2  # center point

        x1 = max(cx - side // 2, 0)
        y1 = max(cy - side // 2, 0)
        x2 = min(cx + side // 2, image.shape[1])
        y2 = min(cy + side // 2, image.shape[0])

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# --- DISPLAY ONLY DETECTED DICE IMAGE ---
cv2.imshow("Detected Dice (Square Boxes)", image)

# --- SAVE RESULT ---
cv2.imwrite("output_detected_dice_square_nomask.png", image)

cv2.waitKey(0)
cv2.destroyAllWindows()