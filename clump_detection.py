import numpy as np
import cv2


#Takes image path as input and returns clump center coordinates as output
def detect_dice_clumps(image_path, merge_distance=60, min_area=300, box_scale=1.00, area_threshold=6000):

    """
    Detect yellow dice in an image, merge close boxes, and return center coordinates (cx, cy, c_theta)
    of boxes with area > area_threshold, formatted for image_to_robot_coords().

    Parameters:
        image_path (str): Path to image, or None to capture via connect_camera().
        merge_distance (float): Distance threshold (in pixels) to merge nearby boxes.
        min_area (float): Minimum contour area to consider as a die.
        box_scale (float): Scaling factor for enlarging detected boxes.
        area_threshold (float): Only boxes with area > this value are returned.

    Returns:
        list of tuples: [(cx, cy, c_theta), ...] formatted for image_to_robot_coords()
    """
    # --- LOAD IMAGE ---
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"❌ Could not load image: {image_path}")

    # --- COLOR THRESHOLDING ---
    LOWER_YELLOW = np.array([15, 60, 60])
    UPPER_YELLOW = np.array([45, 255, 255])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

    # --- CLEAN MASK ---
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # --- FIND INITIAL BOXES ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) > min_area:
            x, y, w, h = cv2.boundingRect(c)
            side = int(max(w, h) * box_scale)
            cx, cy = x + w // 2, y + h // 2
            x1 = max(cx - side // 2, 0)
            y1 = max(cy - side // 2, 0)
            x2 = min(cx + side // 2, image.shape[1])
            y2 = min(cy + side // 2, image.shape[0])
            boxes.append([x1, y1, x2, y2])

    # --- MERGE CLOSE BOXES ---
    def merge_boxes(boxes, dist_thresh):
        merged = True
        while merged:
            merged = False
            new_boxes = []
            skip = set()
            for i in range(len(boxes)):
                if i in skip:
                    continue
                x1a, y1a, x2a, y2a = boxes[i]
                for j in range(i + 1, len(boxes)):
                    if j in skip:
                        continue
                    x1b, y1b, x2b, y2b = boxes[j]
                    # Compute distance between centers
                    cxa, cya = (x1a + x2a) / 2, (y1a + y2a) / 2
                    cxb, cyb = (x1b + x2b) / 2, (y1b + y2b) / 2
                    dist = np.sqrt((cxa - cxb) ** 2 + (cya - cyb) ** 2)
                    if dist < dist_thresh:
                        # Merge
                        x1 = min(x1a, x1b)
                        y1 = min(y1a, y1b)
                        x2 = max(x2a, x2b)
                        y2 = max(y2a, y2b)
                        new_boxes.append([x1, y1, x2, y2])
                        skip.add(j)
                        merged = True
                        break
                else:
                    new_boxes.append(boxes[i])
            boxes = new_boxes
        return boxes

    merged_boxes = merge_boxes(boxes, merge_distance)

    # --- FILTER + COMPUTE CENTER COORDS ---
    clump_coords = []
    for (x1, y1, x2, y2) in merged_boxes:
        area = (x2 - x1) * (y2 - y1)
        if area > area_threshold:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            c_theta = 0.0  # placeholder angle, can be replaced by orientation detection
            clump_coords.append((cx, cy, c_theta))
            # draw on image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)

    # --- DISPLAY RESULT ---
    cv2.imshow("Detected Dice (Merged + Centers)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- RETURN (cx, cy, c_theta) FOR EACH DIE ---
    return clump_coords


#### CLUMP BREAKING FUNCTION FOR FANUC
def DJ_bulldozer_pull(x, y, z, w, p, r):
    bulldoze1_up = [x-100, y-100, z+100, w, p, r]
    bulldoze1 = [x-100, y-100, z-30, w, p, r]
    bulldoze2_up = [x+100, y+100, z+100, w, p, r]
    bulldoze2 = [x+100, y+100, z-30, w, p, r]

    robotDJ.schunk_gripper('open')
    robotDJ.write_cartesian_position(bulldoze2_up)
    robotDJ.write_cartesian_position(bulldoze2)
    robotDJ.write_cartesian_position(bulldoze1)
    robotDJ.write_cartesian_position(bulldoze1_up)

def DJ_bulldozer_push(x, y, z, w, p, r):
    bulldoze1_up = [x-100, y-100, z+100, w, p, r]
    bulldoze1 = [x-100, y-100, z-30, w, p, r]
    bulldoze2_up = [x+100, y+100, z+100, w, p, r]
    bulldoze2 = [x+100, y+100, z-30, w, p, r]

    robotDJ.schunk_gripper('open')
    robotDJ.write_cartesian_position(bulldoze1_up)
    robotDJ.write_cartesian_position(bulldoze1)
    robotDJ.write_cartesian_position(bulldoze2)
    robotDJ.write_cartesian_position(bulldoze2_up)