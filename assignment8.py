import cv2
import numpy as np

# Load the image
image = cv2.imread("dice_on_table.jpg")

# Check if image loaded correctly
if image is None:
    print("Error: Could not load image.")
    exit()

# Get image dimensions
height, width, channels = image.shape
print(f"Image width: {width}px")
print(f"Image height: {height}px")
print(f"Number of channels: {channels}")

# Convert to HSV (keep your exact yellow range)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([29, 50, 50])
upper_yellow = np.array([33, 255, 255])
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Clean up noise
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask = cv2.GaussianBlur(mask, (3, 3), 0)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = image.copy()

# Temporary storage (no numbering yet)
dice_data = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 300:
        continue

    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect

    if min(w, h) == 0:
        continue

    # --- Make it a perfect square (side = smallest dimension) ---
    side = min(w, h)
    square_rect = ((cx, cy), (side, side), angle)

    # --- Normalize angle (-90° to +90°) ---
    if w < h:
        angle = 90 + angle
    if angle > 90:
        angle -= 180

    # --- If negative, add 90° (your requested behavior) ---
    if angle < 0:
        angle += 90

    # Save data (no label yet)
    dice_data.append([cx, cy, angle, square_rect])

# --- Sort left-to-right by center X ---
dice_data.sort(key=lambda d: d[0])

# --- Now label and draw correctly ---
for i, (cx, cy, angle, square_rect) in enumerate(dice_data, start=1):
    box = cv2.boxPoints(square_rect)
    box = np.int32(box)
    cv2.drawContours(output, [box], -1, (0, 0, 255), 2)
    cv2.circle(output, (int(cx), int(cy)), 4, (0, 255, 0), -1)
    cv2.putText(output, f"Die {i}", (int(cx) - 25, int(cy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    cv2.putText(output, f"{angle:.1f}°", (int(cx) - 25, int(cy) + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

# --- Print summary table ---
print("\nDetected Dice (Negative Angles Shifted +90°):")
print(f"{'Die #':<6} {'Center X':<10} {'Center Y':<10} {'Angle (deg)':<10}")
print("-" * 40)
for i, (cx, cy, angle, _) in enumerate(dice_data, start=1):
    print(f"{i:<6} {round(cx,1):<10} {round(cy,1):<10} {round(angle,1):<10}")

# --- Save the full-resolution labeled image ---
cv2.imwrite("dice_labeled_output.jpg", output)


# --- Resize for easier viewing ---
scale_percent = 20
width = int(output.shape[1] * scale_percent / 100)
height = int(output.shape[0] * scale_percent / 100)
resized = cv2.resize(output, (width, height))

cv2.imshow("Dice - Angles Adjusted (+90 if Negative)", resized)


cv2.waitKey(0)
cv2.destroyAllWindows()