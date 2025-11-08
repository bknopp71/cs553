from run_theodore import get_position_info, gripper_command, gripper_request, move_robot_cartesian, move_robot_joint, HOME, deg_to_rad
import cv2 as cv
import numpy as np
from standardbots import StandardBotsRobot, models
import math
from cv_grab import image_to_robot_coords, fanuc_to_full_pose, process_image, connect_camera
from fanuc_dice import DJ_home, DJ_to_dice

# sdk = StandardBotsRobot(
# 	url = "http://10.8.4.11:3000",
# 	token = "8geqfqu0-qbbkig-ozwgr4-tl2xfj7",
# 	robot_kind=StandardBotsRobot.RobotKind.Live
# )



DJ_home()

get_position_info()

if __name__ == "__main__":
    image_path = connect_camera()          # Capture image automatically
    dice_coords = process_image(image_path)  # Detect dice centers
    robot_coords = image_to_robot_coords(dice_coords)  # Convert to robot coordinates

z = 130.0
yaw = -179.9
pitch = 0.0

image_path = connect_camera()          # Capture image automatically
dice_coords = process_image(image_path)  # Detect dice centers
robot_coords = image_to_robot_coords(dice_coords)  # Convert to robot coordinates

fanuc_coords_list = robot_coords[0]  # first element of your tuple

fanuc_full_poses = fanuc_to_full_pose(fanuc_coords_list, z, yaw, pitch)

DJ_to_dice(*fanuc_full_poses[0])

print('done running')

# move_robot_joint(0.002951115369796753, -0.0009422596776857972, -1.569915533065796, 0.004852112848311663, 1.5750012397766113, -3.5428329944610596)

# HOME()

# move_robot_cartesian(-0.7013216685847421, 0.18893406493062648, 0.8867914224576314)

