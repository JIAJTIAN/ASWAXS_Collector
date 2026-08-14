"""
autofocus.py — ASWAXS autofocus script (fixed from original).

Fixes:
  - Typo in parameter name: `camara_pv` → `camera_pv`
  - Undefined `camera_pv` inside autofocus() (was using wrong variable name)
  - Added usage message when called with wrong arguments
"""

import sys
import epics
import cv2
import numpy as np


def get_focus_measure(image_pv: str, width: int, height: int) -> float:
    image_data = epics.caget(image_pv)
    arr = np.array(image_data, dtype=np.uint8)
    if arr.ndim != 2:
        gray = cv2.cvtColor(arr.reshape((height, width, 3)), cv2.COLOR_BGR2GRAY)
    else:
        gray = arr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def autofocus(camera_pv: str, motor_pv: str, delta_z: float):
    """
    Sweep the Z motor while monitoring Laplacian sharpness.
    Halves the step and reverses direction whenever sharpness drops.
    Stops when step < 0.005 mm and moves to the best-focus position.
    """
    width  = int(epics.caget(camera_pv + "ArraySizeX_RBV"))
    height = int(epics.caget(camera_pv + "ArraySizeY_RBV"))
    # Derive image array PV from camera prefix  (e.g. Teslong:cam1: → Teslong:image1:ArrayData)
    image_pv = camera_pv.split(':')[0] + ':image1:ArrayData'

    motor = epics.Motor(motor_pv)
    current_pos = motor.get('VAL')
    best_focus  = get_focus_measure(image_pv, width, height)
    best_pos    = current_pos
    direction   = 1

    while delta_z > 0.005:
        new_pos = current_pos + direction * delta_z
        motor.move(new_pos, wait=True)
        current_focus = get_focus_measure(image_pv, width, height)

        if current_focus < best_focus:
            # Worse — go back, reverse direction, halve step
            motor.move(current_pos, wait=True)
            direction *= -1
            delta_z   /= 2
            print(f"Reversing — dz={delta_z:.4f}")
        else:
            current_pos = new_pos
            best_focus  = current_focus
            best_pos    = current_pos

        print(f"dz={delta_z:.4f}  dir={direction:+d}  pos={best_pos:.3f}  focus={best_focus:.3f}")

    motor.move(best_pos, wait=True)
    print("Autofocus complete.")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: autofocus.py <camera_prefix> <motor_pv> <step_mm>")
        print("  e.g. autofocus.py Teslong:cam1: 15IDD:m7 0.2")
        sys.exit(1)

    camera_pv = sys.argv[1]
    motor_pv  = sys.argv[2]
    delta_z   = float(sys.argv[3])
    autofocus(camera_pv, motor_pv, delta_z)
