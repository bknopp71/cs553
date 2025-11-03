"""
Dice Detection and Pip Counting
Author: Brent Knopp
University of Idaho
Class: CS 5553
Assignment: 7

This program uses OpenCV to detect yellow dice in an image via color
thresholding in the HSV color space. Once a die is located, it draws a
bounding box around it and isolates the inner region of the die to avoid
detecting noisy edges. The cropped region is processed in grayscale and
thresholded to identify dark circular regions representing pips (dots).
The program then filters valid pip contours by size and counts them. The
detected pip count is displayed next to each die.
"""

import cv2
import numpy as np


def display_pictures(a, b, c):
    """
    Displays three images side by side for visual comparison.

    This function takes three images:
      a – the HSV image,
      b – the binary mask image,
      c – the final processed (output) image.

    Each image is resized to 16.5% of its original dimensions so that
    they fit together in one window. The grayscale mask is converted
    to a 3-channel BGR image to allow stacking with the color images.
    All three are then horizontally concatenated using NumPy and shown
    together in a single OpenCV display window.
    """

    # 16.5% display added back
    view_scale = 0.165  # scale factor for downsizing display images

    # Resize HSV image for display
    display_hsv_width = int(a.shape[1] * view_scale)
    display_hsv_height = int(a.shape[0] * view_scale)
    display_hsv_image = cv2.resize(
        a, (display_hsv_width, display_hsv_height),
        interpolation=cv2.INTER_AREA
    )

    # Resize mask image for display
    display_mask_width = int(b.shape[1] * view_scale)
    display_mask_height = int(b.shape[0] * view_scale)
    display_mask_image = cv2.resize(
        b, (display_mask_width, display_mask_height),
        interpolation=cv2.INTER_AREA
    )

    # Resize final processed (output) image for display
    display_width = int(c.shape[1] * view_scale)
    display_height = int(c.shape[0] * view_scale)
    display_image = cv2.resize(
        c, (display_width, display_height),
        interpolation=cv2.INTER_AREA
    )

    # Convert grayscale images (mask) to 3-channel BGR format for stacking
    display_mask_image = cv2.cvtColor(
        display_mask_image, cv2.COLOR_GRAY2BGR
    )

    # Combine all three images horizontally
    combined = np.hstack(
        (display_hsv_image, display_mask_image, display_image)
    )

    # Display all three images side-by-side in one window
    cv2.imshow("HSV -> Mask -> Processed Image", combined)


# Margin percentage to ignore edges when analyzing dice faces
INNER_MARGIN = 0.08  # simplified full-res version with chosen margin

# Load the image file
image = cv2.imread("dice_on_table.jpg")

if image is None:
    # Check if the image was successfully loaded
    print("Error: Could not load image. Check the filename or path.")
else:
    # Copy to preserve original
    processed_image = image.copy()

    # Convert to HSV color space
    hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)

    # Define color thresholds for yellow dice detection
    lower_yellow = np.array([22, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create binary mask for yellow
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply morphological operations to clean up noise in the mask
    kernel = np.ones((5, 5), np.uint8)

    # Remove small white noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Detect outer contours (potential dice) from the binary mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Iterate over all detected contours
    for cnt in contours:
        area = cv2.contourArea(cnt)  # compute contour area

        # Required die size threshold to avoid tiny specks
        if area > 3000:
            # Get bounding box for detected die
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(
                processed_image, (x, y), (x + w, y + h),
                (0, 0, 255), 2
            )

            # Extract region of interest (ROI) for current die
            roi = processed_image[y:y + h, x:x + w]
            h_roi, w_roi = roi.shape[:2]

            # Apply inner margin to crop out noisy edges
            m_h = int(h_roi * INNER_MARGIN)
            m_w = int(w_roi * INNER_MARGIN)
            inner_roi = roi[m_h:h_roi - m_h, m_w:w_roi - m_w]

            # Convert ROI to grayscale for pip (dot) detection
            gray = cv2.cvtColor(inner_roi, cv2.COLOR_BGR2GRAY)

            # Threshold grayscale image to binary (invert so pips are white)
            _, thresh = cv2.threshold(
                gray, 100, 255, cv2.THRESH_BINARY_INV
            )

            # Morphological opening to remove small white specks
            kernel_small = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(
                thresh, cv2.MORPH_OPEN, kernel_small
            )

            # Find contours for each potential pip
            pip_contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Compute dice face area and min/max valid pip contour areas
            face_area = float((w_roi - 2 * m_w) * (h_roi - 2 * m_h))
            min_area = face_area * 0.015  # lower bound for pip area
            max_area = face_area * 0.20   # upper bound for pip area

            # Store all valid pip contour areas
            areas = []
            for pc in pip_contours:
                a = cv2.contourArea(pc)
                if min_area < a < max_area:
                    areas.append(a)

            pip_count = 0  # initialize pip counter
            if areas:
                # Sort areas and use median as reference pip size
                areas.sort()
                ref_area = areas[len(areas) // 2]

                # Count valid pips within acceptable size range
                for pc in pip_contours:
                    a = cv2.contourArea(pc)
                    if (
                        min_area < a < max_area and
                        0.65 * ref_area <= a <= 1.45 * ref_area
                    ):
                        pip_count += 1  # valid pip found
                        px, py, pw, ph = cv2.boundingRect(pc)

                        # Draw small bounding box around each detected pip
                        cv2.rectangle(
                            inner_roi, (px, py),
                            (px + pw, py + ph), (255, 255, 100), 1
                        )

            # Display pip count next to each die
            cv2.putText(
                processed_image, f"Pip Count = {pip_count}",
                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                2, (100, 0, 0), 5
            )

    # Save output image with annotations
    cv2.imwrite("dice_output.png", processed_image)
    print("Output saved as 'dice_output.png'")

    # Display images to screen (HSV, mask, processed)
    display_pictures(hsv, mask, processed_image)

    # Wait for ESC key (27) to exit the display loop
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Close all OpenCV windows
    cv2.destroyAllWindows()