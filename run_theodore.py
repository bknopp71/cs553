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
			print(response.data.message	)
			
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
		
def move_robot_cartesian(x,y,z):
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
					-0.50344496,
					-0.4864938,
					0.0513632,
					-0.496033247,
                ),
            ),
        ).ok()


print("Brent")

get_position_info()

gripper_command('close')
gripper_command('open')
gripper_command('close')

