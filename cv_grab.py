
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
    return [(d[0], d[1], d[2]) for d in dice_data]


# --- Function 3: Convert image coordinates to robot coordinates ---
def image_to_robot_coords(dice_coords):
    """
    Swap X,Y coordinates and apply affine transformations to get robot X,Y.
    """

    # Fanuc affine matrix
    fanuc_affine_matrix = np.array([
        [1.48414, -0.089869, -535.93],
        [0.070689, 1.464, 220.31]
    ])

    fanuc_coords = []
    for cx, cy, c_theta in dice_coords:
        # Swap X and Y before applying transform
        img_pt = np.array([cy, cx, 1])
        fx, fy = fanuc_affine_matrix @ img_pt

        # Convert image angle to Fanuc angle
        fanuc_theta = 30 + (90 - c_theta)
        fanuc_coords.append((fx, fy, fanuc_theta))

    print("\nDice coordinates converted to Fanuc X,Y:")
    for i, (fx, fy, fanuc_theta) in enumerate(fanuc_coords, start=1):
        print(f"Die {i}: X={fx:.2f}, Y={fy:.2f}, r={fanuc_theta:.2f}")

    # StandardBot affine matrix
    standard_affine_matrix = np.array([
        [-0.00151949, 0.0000987, 0.4648],
        [-0.0001228, -0.001549, 2.293]
    ])

    standard_coords = []
    for cx, cy, c_theta in dice_coords:
        # Swap X and Y before applying transform
        img_pt = np.array([cy, cx, 1])
        sx, sy = standard_affine_matrix @ img_pt
        image_theta = c_theta
        standard_coords.append((sx, sy, image_theta))

    print("\nDice coordinates converted to Standard X,Y:")
    for i, (sx, sy, image_theta) in enumerate(standard_coords, start=1):
        print(f"Die {i}: X={sx:.2f}, Y={sy:.2f}, Joint 5 = j5 + {image_theta:.2f}")

    return fanuc_coords, standard_coords



if __name__ == "__main__":
    image_path = connect_camera()          # Capture image automatically
    dice_coords = process_image(image_path)  # Detect dice centers
    robot_coords = image_to_robot_coords(dice_coords)  # Convert to robot coordinates
