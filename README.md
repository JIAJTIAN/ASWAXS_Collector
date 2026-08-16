# ASWAXS_Collector

**ASWAXS_Collector** is the beamline data-collection front-end for the ASWAXS (Anomalous Small- and Wide-Angle X-ray Scattering) instrument at the Advanced Photon Source (APS), Argonne National Laboratory.

It is part of a three-component software stack:

| Component | Role |
|---|---|
| **ASWAXS_Collector** (this repo) | Sample positioning, camera viewing, motor control, position list management |
| [FrameByFrame](https://github.com/JIAJTIAN/FrameByFrame) | Live data reduction and visualization pipeline |
| [ASWAXS_Sample_Position_App](https://github.com/JIAJTIAN/ASWAXS_Sample_Position_App) | Legacy standalone position planning tool |

---

## Overview

ASWAXS_Collector provides a unified GUI for the beamline scientist to:

- View the live sample camera and navigate to sample positions by double-clicking on the image
- Control X, Y, and Z (focus) motors with real-time EPICS readbacks
- Build and manage a sample position list with metadata (role, group, solvent group, layout, notes)
- Visualize the spatial distribution of all positions on an interactive 2D map
- Export position lists to CSV/JSON/POS formats for downstream acquisition scripts
- Interpolate intermediate sample positions via Blender on a remote SSH server

---

## Features

### Camera Tab
- **Live camera stream** via EPICS area detector (OpenCV display with pyqtgraph)
- **Double-click to move**: click any point on the image to drive X/Y motors to that location (requires pixel-to-mm calibration)
- **Autofocus**: runs an external autofocus script against the Z motor
- **Auto-add to list**: optionally records each move as a new sample position

### Motor Control Bar
- Compact aligned panel for **X, Y, Z motors** showing:
  - EPICS DESC (motor description)
  - RBV (real-value readback, live)
  - SP (setpoint entry, press Enter to move)
  - MOVN indicator (animated dot while moving)
  - Step size + tweak ◀▶ buttons for manual jog
- All motor PVs are configurable via the Setup dialog

### Sample Positions Tab
- **Position table** with fields: `name`, `x`, `y`, `z`, `role`, `layout`, `group`, `solvent_group`, `note`
- **Role presets**: Sample, Solvent, GC, Air, Empty, Standard, Background, Inlet, Outlet, Channel, Observation, Skip — each with a distinct color on the map
- **Drag-and-drop reordering** of rows
- **Bulk role assignment** to multiple selected rows
- **Capture from motor**: record current X/Y/Z position directly into the list
- **Move to position**: drive motors to any selected row's coordinates
- **2D position map**: interactive pyqtgraph scatter plot with color-coded roles, sequence arrows, and name labels
- **Blender interpolation**: SSH to remote server and run Blender headless to interpolate intermediate positions along a curve, with configurable spacing
- **File I/O**: open/save as CSV, JSON, or `.pos`; import with automatic column alias resolution

### Setup Dialog
Configurable PV and connection settings, accessible from the menu bar:
- **Motor PVs**: X, Y, Z motor base PV names
- **Camera**: EPICS camera prefix, image prefix, autofocus step size
- **SSH / Blender**: remote host, user, SSH key path, Blender executable path, Blender macro script path, local/remote mount point mapping

Settings are persisted to `sample_station_config.json` alongside the script.

### Menu Bar
| Menu | Actions |
|---|---|
| **File** | New, Open, Save, Save As, Export |
| **Acquisition** | Capture Position, Move to Selected |
| **Positions** | Add, Delete, Duplicate, Move Up/Down, Assign Role, Clear All |
| **Blender** | Run Blender Interpolation |
| **Setup** | Open Setup Dialog |
| **Help** | About |

---

## Requirements

```
Python >= 3.10
PyQt6 >= 6.7
pyqtgraph >= 0.13
numpy
opencv-python      # optional — camera streaming
paramiko           # optional — Blender SSH
pyepics            # optional — EPICS motor/PV control
```

Install dependencies (recommended: use the `pydm-env` conda environment at the beamline):

```bash
pip install PyQt6 pyqtgraph numpy opencv-python paramiko pyepics
```

### Offline mode

The application runs without EPICS, camera, or SSH available. All hardware-dependent features degrade gracefully — the GUI remains fully functional for position list editing, file I/O, and map visualization.

---

## Usage

```bash
python sample_station.py
```

On first launch, default PV names for Sector 15-ID-D are loaded. Open **Setup** from the menu bar to configure PVs and connection settings for a different instrument.

---

## Configuration

Settings are stored in `sample_station_config.json` (auto-created on first run):

```json
{
    "X_MOTOR_PV":       "PREFIX:mXX",
    "Y_MOTOR_PV":       "PREFIX:mXX",
    "Z_MOTOR_PV":       "PREFIX:mXX",
    "CAMERA_PREFIX":    "CAMERA:cam1:",
    "IMAGE_PREFIX":     "CAMERA:image1:",
    "AUTOFOCUS_STEP":   "0.2",
    "AUTOFOCUS_SCRIPT": "autofocus.py",
    "BLENDER_HOST":     "your.remote.host",
    "BLENDER_USER":     "username",
    "BLENDER_KEY":      "/home/username/.ssh/id_rsa",
    "BLENDER_EXE":      "blender",
    "BLENDER_SCRIPT":   "/path/to/Blender_Macro.py",
    "LOCAL_MOUNT":      "/local/data/mount",
    "REMOTE_MOUNT":     "/remote/data/mount"
}
```

---

## File Structure

```
ASWAXS_Collector/
├── sample_station.py        # Main application (PyQt6)
├── sample_view.py           # Earlier single-window version (reference)
├── autofocus.py             # Autofocus helper script
├── _test_smoke.py           # Smoke tests
├── Data/
│   └── camera_calib.txt     # Pixel-to-mm calibration factor
└── sample_station_config.json  # Runtime config (auto-generated, git-ignored)
```

---

## Position File Formats

| Format | Notes |
|---|---|
| `.csv` | Comma-separated, flexible column aliases supported |
| `.json` | List of position objects |
| `.pos` | Legacy whitespace-delimited format from original ASWAXS software |

Column aliases for CSV import (e.g. `s_x`, `sx`, `sp_x`, `sample_x`, `motor_x` are all recognized as `x`).

---

## Related Repositories

- [FrameByFrame](https://github.com/JIAJTIAN/FrameByFrame) — live SAXS/WAXS data reduction pipeline
- [ASWAXS_Sample_Position_App](https://github.com/JIAJTIAN/ASWAXS_Sample_Position_App) — PyQt5 predecessor to the position management module

---

## Author

Jiajun Tian — ChemMatCARS, University of Chicago  
Contact: jiajtian@uchicago.edu
