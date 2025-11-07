import numpy as np
from standardbots import StandardBotsRobot

# --- Connect to the robot -----------------------------------------------------
# Initialize the Standard Bots SDK connection to the robot controller.
# 'url' is the robot's IP and port where the Standard Bots server is running.
# 'token' is your authentication key for API access.
# 'robot_kind' specifies that this connection controls a live physical robot.
sdk = StandardBotsRobot(
    url="http://172.29.208.19:3000",
    token="pcabdkvh-ktr69i-bzh47wpj-alcfy",
    robot_kind=StandardBotsRobot.RobotKind.Live
)

# --- Open a connection session -------------------------------------------------
# The 'with' statement automatically manages connection setup and teardown.
# When this block starts, it connects; when it ends, it disconnects.
with sdk.connection():

    # --- Release joint brakes --------------------------------------------------
    # Before any movement, the robot's brakes must be released.
    # This command ensures all joints are free to move.
    sdk.movement.brakes.unbrake().ok()

    # --- Get the current arm pose ---------------------------------------------
    # Retrieves the robot’s current TCP (Tool Center Point) position and orientation.
    # Returns a structure containing both position (x, y, z) and orientation (quaternion).
    pose = sdk.movement.position.get_arm_position().ok()

    # Extract the position components (in mm) into a NumPy array.
    # These correspond to Cartesian coordinates in the world frame.
    p = np.array([
        pose.tooltip_position.position.x,  # X coordinate
        pose.tooltip_position.position.y,  # Y coordinate
        pose.tooltip_position.position.z   # Z coordinate
    ])

    # Extract the orientation as a quaternion [w, x, y, z].
    # Quaternions avoid the gimbal-lock problem of Euler angles.
    q = np.array([
        pose.tooltip_position.orientation.w,  # Real component
        pose.tooltip_position.orientation.x,  # i-component
        pose.tooltip_position.orientation.y,  # j-component
        pose.tooltip_position.orientation.z   # k-component
    ])

    # --- Compute target pose ---------------------------------------------------
    # We want to move straight down by 5 mm along the world Z-axis.
    # Negative Z means moving downward relative to the world coordinate frame.
    p_target = p + np.array([0, 0, -5])

    # --- Move to the new pose --------------------------------------------------
    # Sends a motion command to move the robot's TCP to the target position
    # while maintaining the same orientation (quaternion unchanged).
    # The 'move_to_pose()' command specifies position (x, y, z)
    # and quaternion orientation (w, i, j, k).
    sdk.movement.position.move_to_pose(
        x=p_target[0],  # Target X (same as current)
        y=p_target[1],  # Target Y (same as current)
        z=p_target[2],  # Target Z (5 mm lower)
        w=q[0],         # Orientation W
        i=q[1],         # Orientation X
        j=q[2],         # Orientation Y
        k=q[3]          # Orientation Z
    ).ok()

    # --- Confirmation output ---------------------------------------------------
    # Print a success message to the console showing the new Z value.
    print(f"✅ Dropped 5 mm down from current pose to Z = {p_target[2]:.2f}")