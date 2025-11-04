import numpy as np
import cv2

def get_points(n=4):
    """Prompt user to input n points for pixel and robot coordinates."""
    pixel_pts = []
    robot_pts = []
    print(f"Enter {n} corresponding pixel and robot coordinates:")

    for i in range(n):
        print(f"\n--- Point {i+1} ---")
        u = float(input("Pixel u (x): "))
        v = float(input("Pixel v (y): "))
        X = float(input("Robot X (mm): "))
        Y = float(input("Robot Y (mm): "))
        pixel_pts.append([u, v])
        robot_pts.append([X, Y])

    return np.array(pixel_pts, dtype=np.float32), np.array(robot_pts, dtype=np.float32)

def pixel_to_robot(u, v, M):
    """Map a pixel coordinate (u, v) to robot (X, Y)."""
    point = np.array([u, v, 1])
    X, Y = np.dot(M, point)
    return X, Y

# Step 1: Get calibration data
pixels, robots = get_points()

# Step 2: Compute the affine transformation
M, _ = cv2.estimateAffine2D(pixels, robots)

if M is None:
    print("❌ Could not compute transformation. Check your input points.")
    exit()

print("\n✅ Affine Transformation Matrix (2x3):")
print(M)

# Step 3: Test the transformation
print("\nNow test mapping a pixel to robot coordinate.")
u_test = float(input("Enter pixel u (x): "))
v_test = float(input("Enter pixel v (y): "))

X_out, Y_out = pixel_to_robot(u_test, v_test, M)
print(f"\nPixel ({u_test:.2f}, {v_test:.2f}) → Robot ({X_out:.2f}, {Y_out:.2f}) mm")