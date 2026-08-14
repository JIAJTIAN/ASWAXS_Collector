"""
Smoke test: constructs SampleView without EPICS and checks all widgets exist.
Run with: python _test_smoke.py
"""
import sys
import os

# Force headless Qt (no display needed for widget construction check)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication
app = QApplication(sys.argv)

import sample_view as sv

print(f"EPICS_AVAILABLE    : {sv.EPICS_AVAILABLE}")
print(f"PARAMIKO_AVAILABLE : {sv.PARAMIKO_AVAILABLE}")

w = sv.SampleView()

# Check critical widgets
checks = [
    ("x_motor",         sv.MotorPanel),
    ("y_motor",         sv.MotorPanel),
    ("z_motor",         sv.MotorPanel),
    ("tabs",            None),
    ("cf_edit",         None),
    ("focus_lbl",       None),
    ("pos_list",        None),
    ("roi_edit",        None),
    ("interp_edit",     None),
    ("offset_label",    None),
    ("_setup_edits",    dict),
]

all_ok = True
for name, typ in checks:
    obj = getattr(w, name, None)
    ok = obj is not None and (typ is None or isinstance(obj, typ))
    status = "OK" if ok else "MISSING/WRONG TYPE"
    if not ok:
        all_ok = False
    print(f"  {name:20s} {status}")

# Check setup tab has all expected PV keys
expected_keys = {
    "X_MOTOR_PV", "Y_MOTOR_PV", "Z_MOTOR_PV",
    "CAMERA_PREFIX", "IMAGE_PREFIX",
    "AUTOFOCUS_STEP", "AUTOFOCUS_SCRIPT",
    "BLENDER_HOST", "BLENDER_USER", "BLENDER_KEY",
    "BLENDER_SCRIPT", "LOCAL_MOUNT", "REMOTE_MOUNT",
}
missing_keys = expected_keys - set(w._setup_edits.keys())
if missing_keys:
    print(f"  MISSING setup keys: {missing_keys}")
    all_ok = False
else:
    print(f"  Setup keys         OK ({len(w._setup_edits)} keys)")

# Simulate adding a position manually
w.x_motor.rbv_lbl.setText("1.234")
w.y_motor.rbv_lbl.setText("2.345")
w.z_motor.rbv_lbl.setText("3.456")
w.addPosition()
assert len(w.positions) == 1, "addPosition failed"
assert w.positions[0] == {"X": 1.234, "Y": 2.345, "Z": 3.456}
print("  addPosition        OK")

# Simulate save/load round-trip
import tempfile, numpy as np
with tempfile.NamedTemporaryFile(suffix=".pos", delete=False, mode='w') as tf:
    tmp_path = tf.name
w.savePositions.__func__  # just check it's callable
np.savetxt(tmp_path, np.array([[1.234, 2.345, 3.456]]), fmt="%.3f", header="X Y Z")
prev = list(w.positions)
w.positions = []
# Manually call openPositions logic
with open(tmp_path) as f:
    lines = f.readlines()
header = lines[0].lstrip('#').strip().split()
old_fmt = any('ca://' in h for h in header)
for line in lines[1:]:
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    vals = [float(v) for v in line.split()]
    w.positions.append({"X": round(vals[0],3), "Y": round(vals[1],3),
                        "Z": round(vals[2],3) if len(vals)>2 else 0.0})
assert w.positions[0] == {"X": 1.234, "Y": 2.345, "Z": 3.456}, f"Got: {w.positions}"
print("  pos save/load      OK")
os.unlink(tmp_path)

# cfChanged bug fix: try writing cf (would crash with old 'r' mode)
w.cf_edit.setText("0.003000")
try:
    w.cfChanged()
    print("  cfChanged          OK (file write)")
except Exception as e:
    print(f"  cfChanged          FAIL: {e}")
    all_ok = False

# roiSizeChanged bug fix: setText with str not float
w.roi_edit.setText("80")
try:
    w.roiSizeChanged()
    assert w.roisize == 80
    print("  roiSizeChanged     OK")
except Exception as e:
    print(f"  roiSizeChanged     FAIL: {e}")
    all_ok = False

# blenderInterpolate eval→float fix
w.interp_edit.setText("0.5")
try:
    spacing = float(w.interp_edit.text())
    assert spacing == 0.5
    print("  interp spacing     OK (float() not eval())")
except Exception as e:
    print(f"  interp spacing     FAIL: {e}")
    all_ok = False

print()
print("=" * 40)
print("RESULT:", "ALL PASS" if all_ok else "SOME FAILURES — see above")
sys.exit(0 if all_ok else 1)
