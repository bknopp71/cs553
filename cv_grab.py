# # coding=utf-8
# import cv2
# import numpy as np
# import mvsdk
# import time  # for timestamped filenames

# def capture_image(frame):
#     """Save the current frame to disk as 'dice.jpg'."""
#     filename = "dice_capture.jpg"
#     cv2.imwrite(filename, frame)
#     print(f"Image captured and saved as {filename}")

# def main_loop():
#     # Enumerate all connected cameras
#     DevList = mvsdk.CameraEnumerateDevice()
#     nDev = len(DevList)
#     if nDev < 1:
#         print("No camera found!")
#         return

#     # List all detected cameras
#     for i, DevInfo in enumerate(DevList):
#         print(f"{i}: {DevInfo.GetFriendlyName()} ({DevInfo.GetPortType()})")

#     # Camera selection
#     i = 0 if nDev == 1 else int(input("Select camera index: "))
#     DevInfo = DevList[i]
#     print("Selected camera:", DevInfo)

#     # Initialize camera
#     hCamera = 0
#     try:
#         hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
#     except mvsdk.CameraException as e:
#         print(f"CameraInit Failed({e.error_code}): {e.message}")
#         return

#     # Get camera capabilities
#     cap = mvsdk.CameraGetCapability(hCamera)
#     monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

#     # Set output format
#     if monoCamera:
#         mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
#     else:
#         mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

#     # Continuous capture
#     mvsdk.CameraSetTriggerMode(hCamera, 0)

#     # Manual exposure: 30ms
#     mvsdk.CameraSetAeState(hCamera, 0)
#     mvsdk.CameraSetExposureTime(hCamera, 30 * 1000)

#     # Start capture
#     mvsdk.CameraPlay(hCamera)

#     # Allocate buffer
#     FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
#     pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

#     # Main capture loop
#     while True:
#         try:
#             pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
#             mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
#             mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

#             # Convert to OpenCV format
#             frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
#             frame = np.frombuffer(frame_data, dtype=np.uint8)
#             frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

#             # Resize for display
#             display_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
#             cv2.imshow("Live Feed - Press 'c' to capture, 'q' to quit", display_frame)

#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('c'):
#                 capture_image(frame)  # Save the current frame
#             elif key == ord('q'):
#                 break

#         except mvsdk.CameraException as e:
#             if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
#                 print(f"CameraGetImageBuffer failed({e.error_code}): {e.message}")

#     # Clean up
#     mvsdk.CameraUnInit(hCamera)
#     mvsdk.CameraAlignFree(pFrameBuffer)

# def main():
#     try:
#         main_loop()
#     finally:
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

# # Load the image
# # image = cv2.imread("dice_on_table.jpg")
# image = cv2.imread("dice_capture.jpg")

# # Check if image loaded correctly
# if image is None:
#     print("Error: Could not load image.")
#     exit()

# # Get image dimensions
# height, width, channels = image.shape
# print(f"Image width: {width}px")
# print(f"Image height: {height}px")
# print(f"Number of channels: {channels}")

# # Convert to HSV (keep your exact yellow range)
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# # lower_yellow = np.array([29, 50, 50])
# # upper_yellow = np.array([33, 255, 255])
# # Very wide yellow range in HSV
# lower_yellow = np.array([18,50,50])   # H=20, low S and V to include dark/dull yellows
# upper_yellow = np.array([33, 255, 255]) # H=40, full S and V to include bright yellow

# mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# # Clean up noise
# kernel = np.ones((5, 5), np.uint8)
# mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
# mask = cv2.GaussianBlur(mask, (3, 3), 0)

# # Find contours
# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# output = image.copy()

# # Temporary storage (no numbering yet)
# dice_data = []

# for cnt in contours:
#     area = cv2.contourArea(cnt)
#     if area < 300:
#         continue

#     rect = cv2.minAreaRect(cnt)
#     (cx, cy), (w, h), angle = rect

#     if min(w, h) == 0:
#         continue

#     # --- Make it a perfect square (side = smallest dimension) ---
#     side = min(w, h)
#     square_rect = ((cx, cy), (side, side), angle)

#     # --- Normalize angle (-90° to +90°) ---
#     if w < h:
#         angle = 90 + angle
#     if angle > 90:
#         angle -= 180

#     # --- If negative, add 90° (your requested behavior) ---
#     if angle < 0:
#         angle += 90
        
        

#     # Save data (no label yet)
#     dice_data.append([cx, cy, angle, square_rect])

# # --- Sort left-to-right by center X ---
# dice_data.sort(key=lambda d: d[0])

# # --- Now label and draw correctly ---
# for i, (cx, cy, angle, square_rect) in enumerate(dice_data, start=1):
#     box = cv2.boxPoints(square_rect)
#     box = np.int32(box)
#     cv2.drawContours(output, [box], -1, (0, 0, 255), 2)
#     cv2.circle(output, (int(cx), int(cy)), 4, (0, 255, 0), -1)
#     cv2.putText(output, f"Die {i}", (int(cx) - 25, int(cy) - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
#     cv2.putText(output, f"{angle:.1f}°", (int(cx) - 25, int(cy) + 10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

# # --- Print summary table ---
# print("\nDetected Dice (Negative Angles Shifted +90°):")
# print(f"{'Die #':<6} {'Center X':<10} {'Center Y':<10} {'Angle (deg)':<10}")
# print("-" * 40)
# for i, (cx, cy, angle, _) in enumerate(dice_data, start=1):
#     print(f"{i:<6} {round(cx,1):<10} {round(cy,1):<10} {round(angle,1):<10}")

# # --- Save the full-resolution labeled image ---
# cv2.imwrite("dice_labeled_output.jpg", output)


# # --- Resize for easier viewing ---
# scale_percent = 100
# width = int(output.shape[1] * scale_percent / 100)
# height = int(output.shape[0] * scale_percent / 100)
# resized = cv2.resize(output, (width, height))

# cv2.imshow("Dice - Angles Adjusted (+90 if Negative)", resized)


# cv2.waitKey(0)
# cv2.destroyAllWindows()

# coding=utf-8
# coding=utf-8
# coding=utf-8
import cv2
import numpy as np
import mvsdk
import time

# --- Function 1: Connect to camera and capture image ---
def connect_camera():
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        raise RuntimeError("No camera found!")

    DevInfo = DevList[0]
    print("Selected camera:", DevInfo.GetFriendlyName())

    try:
        hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
    except mvsdk.CameraException as e:
        raise RuntimeError(f"CameraInit Failed({e.error_code}): {e.message}")

    cap = mvsdk.CameraGetCapability(hCamera)
    monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

    if monoCamera:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
    else:
        mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

    mvsdk.CameraSetTriggerMode(hCamera, 0)
    mvsdk.CameraSetAeState(hCamera, 0)
    mvsdk.CameraSetExposureTime(hCamera, 30 * 1000)
    mvsdk.CameraPlay(hCamera)

    FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
    pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    try:
        pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
        mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

        frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if monoCamera else 3))

        filename = "dice_capture.jpg"
        cv2.imwrite(filename, frame)
        print(f"Image captured and saved as {filename}")

    finally:
        mvsdk.CameraUnInit(hCamera)
        mvsdk.CameraAlignFree(pFrameBuffer)

    return filename


# --- Function 2: Process image and detect dice ---
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([17, 40, 40])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = image.copy()
    dice_data = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        if min(w, h) == 0:
            continue

        side = min(w, h)
        square_rect = ((cx, cy), (side, side), angle)
        if w < h:
            angle = 90 + angle
        if angle > 90:
            angle -= 180
        if angle < 0:
            angle += 90

        dice_data.append([cx, cy, angle, square_rect])

    dice_data.sort(key=lambda d: d[0])

    for i, (cx, cy, angle, square_rect) in enumerate(dice_data, start=1):
        box = cv2.boxPoints(square_rect)
        box = np.int32(box)
        cv2.drawContours(output, [box], -1, (0, 0, 255), 2)
        cv2.circle(output, (int(cx), int(cy)), 4, (0, 255, 0), -1)
        cv2.putText(output, f"Die {i}", (int(cx) - 25, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(output, f"{angle:.1f}°", (int(cx) - 25, int(cy) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    print("\nDetected Dice (Negative Angles Shifted +90°):")
    print(f"{'Die #':<6} {'Center X':<10} {'Center Y':<10} {'Angle (deg)':<10}")
    print("-" * 40)
    for i, (cx, cy, angle, _) in enumerate(dice_data, start=1):
        print(f"{i:<6} {round(cx,1):<10} {round(cy,1):<10} {round(angle,1):<10}")

    cv2.imwrite("dice_labeled_output.jpg", output)
    cv2.imshow("Dice - Angles Adjusted (+90 if Negative)", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Return dice centers (cx, cy) for further processing
    return [(d[0], d[1]) for d in dice_data]


# --- Function 3: Convert image coordinates to robot coordinates ---
def image_to_robot_coords(dice_coords):
    """
    Swap X,Y coordinates and apply affine transformation to get robot X,Y.
    """
    # Affine transformation matrix
    affine_matrix = np.array([
        [1.48414, -0.089869, -535.93],
        [0.070689, 1.464, 220.31]
    ])

    robot_coords = []
    for cx, cy in dice_coords:
        # Swap X and Y
        img_pt = np.array([cy, cx, 1])  # [Y, X, 1] for affine
        rx, ry = affine_matrix @ img_pt
        robot_coords.append((rx, ry))

    print("\nDice coordinates converted to robot X,Y:")
    for i, (rx, ry) in enumerate(robot_coords, start=1):
        print(f"Die {i}: X={rx:.2f}, Y={ry:.2f}")

    return robot_coords


if __name__ == "__main__":
    image_path = connect_camera()          # Capture image automatically
    dice_coords = process_image(image_path)  # Detect dice centers
    robot_coords = image_to_robot_coords(dice_coords)  # Convert to robot coordinates
