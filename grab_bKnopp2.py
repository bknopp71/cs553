import cv2
import numpy as np
import mvsdk
import time
from standardbots import StandardBotsRobot, models

sdk = StandardBotsRobot(
    url = "http://10.8.4.11:3000",
    token = "8geqfqu0-qbbkig-ozwgr4-tl2xfj7",
    robot_kind=StandardBotsRobot.RobotKind.Live
)


def get_position_info():
    with sdk.connection():
        print("Connected to Standard Bots server successfully!")
        sdk.movement.brakes.unbrake().ok()
        print("unbraked successfully!")
        response = sdk.movement.position.get_arm_position()

        try:
            data = response.ok()
            j_1, j_2, j_3, j_4, j_5, j_6 = data.joint_rotations
            position = data.tooltip_position.position
            orientation = data.tooltip_position.orientation
            joints = data.joint_rotations

            print(f"Joints: {joints}")
            print(f"Got Position: {position}")
            print(f"Got orientation: {orientation}")

            return j_1, j_2, j_3, j_4, j_5, j_6

        except Exception:
            print(response.data.message )
            
def gripper_request(WIDTH, FORCE):
    with sdk.connection():
        response = sdk.equipment.control_gripper(
            models.GripperCommandRequest(
                kind=models.GripperKindEnum.Onrobot2Fg14,
                onrobot_2fg14=models.OnRobot2FG14GripperCommandRequest(
                    grip_direction=models.LinearGripDirectionEnum.Inward,
                    target_grip_width=models.LinearUnit(
                        value=WIDTH, unit_kind=models.LinearUnitKind.Meters
                    ),
                    target_force=models.ForceUnit(
                        value=FORCE,
                        unit_kind=models.ForceUnitKind.Newtons,
                    ),
                    control_kind=models.OnRobot2FG14ControlKindEnum.Move,
                ),
            )
        )
    try:
        data = response.ok()
    except Exception:
        print(response.data.message)
    
def gripper_command(STRING):
    if STRING == 'open':
        print('OPENING GRIPPER')
        gripper_request(0.11, 10.0)
    if STRING == 'close':
        print('CLOSING GRIPPER')
        gripper_request(0.032,10.0)
        
def move_robot_cartesian(x,y,z, a = -0.0031365594398657447, b = 0.7087946554159895, c = -0.00016275162300366026, d = 0.7054078763102367):
        
    with sdk.connection():
        sdk.movement.brakes.unbrake().ok()
        sdk.movement.position.move(
            position=models.Position(
                unit_kind=models.LinearUnitKind.Meters,
                x=x,
                y=y,
                z=z,
            ),
            orientation=models.Orientation(
                kind=models.OrientationKindEnum.Quaternion,
                quaternion=models.Quaternion(
                    a,
                    b, 
                    c,
                    d,
                ),
            ),
        ).ok()

def move_robot_joint(j1, j2, j3, j4, j5, j6):
    with sdk.connection():
        sdk.movement.brakes.unbrake().ok()
        arm_rotations = models.ArmJointRotations(joints=(j1, j2, j3, j4, j5, j6))
        #Log to ensure the values are correct
        position_request = models.ArmPositionUpdateRequest(
            kind=models.ArmPositionUpdateRequestKindEnum.JointRotation,
            joint_rotation=arm_rotations,
        )
        sdk.movement.position.set_arm_position(position_request).ok()

print("Brent")

get_position_info()

# gripper_command('close')
# gripper_command('open')
# gripper_command('close')

move_robot_joint(0.002951115369796753, -0.0009422596776857972, -1.569915533065796, 0.004852112848311663, 1.5750012397766113, -3.1428329944610596)

def HOME():
    move_robot_joint(0.002951115369796753, -0.0009422596776857972, -1.569915533065796, 0.004852112848311663, 1.5750012397766113, -3.1428329944610596)
    print('Theodore move to home position')



#gripper_command('close')
#gripper_command('open')
#gripper_command('close')


def pick_up_1():

    above_x = -.8822
    above_y = .4614
    above_z = .4569

    x = -.8822
    y = .4614
    z = .3632   

    move_robot_cartesian(above_x, above_y, above_z)
    gripper_command('open')
    move_robot_cartesian(x, y, z)
    gripper_command('close')
    move_robot_cartesian(above_x, above_y, above_z)

def pick_up_2():

    above_x = -.8022
    above_y = .4614
    above_z = .4569

    x = -.8022
    y = .4614
    z = .3632   

    move_robot_cartesian(above_x, above_y, above_z)
    gripper_command('open')
    move_robot_cartesian(x, y, z)
    gripper_command('close')
    move_robot_cartesian(above_x, above_y, above_z)

def pick_up_3():

    above_x = -.7195
    above_y = .4614
    above_z = .4569

    x = -.7195
    y = .4614
    z = .3632   

    move_robot_cartesian(above_x, above_y, above_z)
    gripper_command('open')
    move_robot_cartesian(x, y, z)
    gripper_command('close')
    move_robot_cartesian(above_x, above_y, above_z)

def pick_up_4():

    above_x = -.6368
    above_y = .4614
    above_z = .4569

    x = -.6368
    y = .4614
    z = .3632   

    move_robot_cartesian(above_x, above_y, above_z)
    gripper_command('open')
    move_robot_cartesian(x, y, z)
    gripper_command('close')
    move_robot_cartesian(above_x, above_y, above_z)


def drop_off_1():

    above_x = -.8822
    above_y = .4614
    above_z = .46

    x = -.8822
    y = .4614
    z = .3632   

    move_robot_cartesian(above_x, above_y, above_z)
    move_robot_cartesian(x, y, z)
    gripper_command('open')
    move_robot_cartesian(above_x, above_y, above_z)

def drop_off_2():

    above_x = -.8022
    above_y = .4614
    above_z = .46

    x = -.8022
    y = .4614
    z = .3632   

    move_robot_cartesian(above_x, above_y, above_z)
    move_robot_cartesian(x, y, z)
    gripper_command('open')
    move_robot_cartesian(above_x, above_y, above_z)

def drop_off_3():

    above_x = -.7195
    above_y = .4614
    above_z = .46

    x = -.7195
    y = .4614
    z = .3632   

    move_robot_cartesian(above_x, above_y, above_z)
    move_robot_cartesian(x, y, z)
    gripper_command('open')
    move_robot_cartesian(above_x, above_y, above_z)

def drop_off_4():

    above_x = -.6368
    above_y = .4614
    above_z = .46

    x = -.6368
    y = .4614
    z = .3632   

    move_robot_cartesian(above_x, above_y, above_z)
    move_robot_cartesian(x, y, z)
    gripper_command('open')
    move_robot_cartesian(above_x, above_y, above_z)

def pick_up(x, y, theta):

    offset_x = 0.012
    offset_y = -0.02
    above_z = .46  
    pick_up_z = .41
    
    move_robot_cartesian(x+offset_x, y+offset_y, above_z)
    move_robot_cartesian(x+offset_x, y+offset_y, pick_up_z)

    gripper_command('open')
    j1, j2, j3, j4, j5, j6 = get_position_info()
    rad = theta * 3.14/180
    j6 = j6+rad
    move_robot_joint(j1, j2, j3, j4, j5, j6)
    # off_set_z_pick = 0.36

    vertical_drop = -1.3*3.14/180
    # vertical decent in joint movement to the die
    j2 = j2 + vertical_drop
    move_robot_joint(j1, j2, j3, j4, j5, j6)
    gripper_command('close')
    j2 = j2 - vertical_drop
    move_robot_joint(j1, j2, j3, j4, j5, j6)
    move_robot_cartesian(x+offset_x, y+offset_y, pick_up_z)
    drop_off_3()


   

    # move_robot_cartesian(x+offset_x, y+offset_y, orien_z + 0.18)
    #move_robot_cartesian(x, y, above_z, orien_x, orien_y, orien_z, orien_w)
    gripper_command('open')


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
    global pick_point
    # new_l = 0
    for i, (sx, sy, image_theta) in enumerate(standard_coords, start=1):
        fourth = (sx, sy, image_theta)
        print(f"Die {i}: X={sx:.4f}, Y={sy:.4f}, Joint 5 = j5 + {image_theta:.4f}")

    return fanuc_coords, standard_coords



if __name__ == "__main__":
    image_path = connect_camera()          # Capture image automatically
    dice_coords = process_image(image_path)  # Detect dice centers
    robot_coords = image_to_robot_coords(dice_coords)  # Convert to robot coordinate4
    print(robot_coords[1][-1][0], robot_coords[1][-1][1], robot_coords[1][-1][2])
    pick_up(robot_coords[1][-1][0], robot_coords[1][-1][1], robot_coords[1][-1][2])

    #pick_up(-.2776, .879, 0)


# print("COORDINATES: ", robot_coords)


# HOME()
# get_position_info()
# pick_up(-0.34929, 0.71896, 0)
# print(standard_coords[1])
# pick_up(standard_coords[1])
# print("COORDINATE: ", standard_coords[6][0])
# print("COORDINATE: ", standard_coords[6][1])
# print("COORDINATE: ", standard_coords[6][2])