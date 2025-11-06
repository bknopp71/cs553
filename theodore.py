import cv2 as cv
import numpy as np
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
		print("brake successfully!")
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
			print(response.data.message	)

print("Brent")

get_position_info()