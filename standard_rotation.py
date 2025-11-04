def standard_rotation_transform(angle_deg):
    """
    standard_rotation_transform(angle_deg)

    Maps an input angle (0°–90°) into a special rotation range used for
    transforming or remapping orientations in a two-phase system.

    Behavior:
      • From 0° → 45°: starts at 180° and increases linearly to 225°.
      • From 45° → 90°: starts at 135° and increases linearly to 180°.

    Args:
        angle_deg (float): Input angle in degrees (0–90).

    Returns:
        float: Transformed angle in degrees.
    """
    if angle_deg < 0 or angle_deg > 90:
        raise ValueError("Input angle must be between 0 and 90 degrees.")

    if angle_deg <= 45:
        # Linear interpolation: 180 → 225 over 0 → 45
        return 180 + (angle_deg * (225 - 180) / 45)
    else:
        # Linear interpolation: 135 → 180 over 45 → 90
        return 135 + ((angle_deg - 45) * (180 - 135) / 45)


# --- Example test ---
if __name__ == "__main__":
    for a in [0, 15, 30, 45, 60, 75, 90]:
        print(f"Input: {a:>3}°  ->  Output: {standard_rotation_transform(a):.2f}°")