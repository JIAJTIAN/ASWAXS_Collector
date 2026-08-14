"""
sample_view.py — Qt6 rewrite of the ASWAXS Sample View widget.

Drops all PyDM dependencies. Every EPICS PV is configurable at runtime via
the Setup tab and persisted to sample_view_config.json next to this file.
"""

import os
import sys
import time
import json
import socket
import subprocess
import atexit

import numpy as np
import pyqtgraph as pg

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 (OpenCV) not available — focus parameter and colour conversion disabled")

try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    print("Warning: paramiko not available — SSH/Blender remote functions disabled")

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QCheckBox, QListWidget,
    QTabWidget, QFileDialog, QMessageBox, QSplitter,
    QApplication, QAbstractItemView, QDialog,
    QFrame, QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QObject
from PyQt6.QtGui import QFont

# Row-major so setImage((H, W, 3)) works naturally; must be set before any
# pyqtgraph widget is constructed.
pg.setConfigOption('imageAxisOrder', 'row-major')

try:
    from epics import Motor, PV
    # Verify CA can actually initialise (fails on Windows without IOC or with
    # broken encoding, e.g. LookupError: unknown encoding: utf-8:surrogateescape)
    import epics.ca as _ca
    _ca.initialize_libca()
    EPICS_AVAILABLE = True
except Exception as _epics_err:
    EPICS_AVAILABLE = False
    Motor = None  # type: ignore[assignment,misc]
    PV    = None  # type: ignore[assignment,misc]
    print(f"Warning: EPICS unavailable ({_epics_err}) — running in offline mode")

# ── Paths ────────────────────────────────────────────────────────────────────

_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(_DIR, "sample_view_config.json")
CALIB_FILE  = os.path.join(_DIR, "Data", "camera_calib.txt")

DEFAULT_CONFIG = {
    "X_MOTOR_PV":      "15IDD:m19",
    "Y_MOTOR_PV":      "15IDD:m18",
    "Z_MOTOR_PV":      "15IDD:m7",
    "CAMERA_PREFIX":   "Teslong:cam1:",
    "IMAGE_PREFIX":    "Teslong:image1:",
    "AUTOFOCUS_STEP":  "0.2",
    "AUTOFOCUS_SCRIPT": os.path.join(_DIR, "autofocus.py"),
    "BLENDER_HOST":    "164.54.169.92",
    "BLENDER_USER":    "chem_epics",
    "BLENDER_KEY":     "/home/chem_epics/.ssh/mykey",
    "BLENDER_SCRIPT":  "/home/chem_epics/cars6/Data/chemmat/ASWAXS/ASWAXS/Scripts/Blender_Macro.py",
    "LOCAL_MOUNT":     "/home/chem_epics/cars6/Data",
    "REMOTE_MOUNT":    "/home/chem_epics/cars6/Data",
}


# ── Thread-safe EPICS → Qt bridge ────────────────────────────────────────────

class _PVBridge(QObject):
    """Emits a Qt signal from a pyepics CA callback (background thread)."""
    changed = pyqtSignal(str, object)   # pvname, value

    def __call__(self, pvname, value, **_):
        if value is not None:
            self.changed.emit(pvname, value)


# ── Single-motor control panel ────────────────────────────────────────────────

class MotorPanel(QWidget):
    """Compact motor panel: DESC label, RBV readback, SP entry, tweak buttons, MOVN dot."""

    def __init__(self, label="Motor", parent=None):
        super().__init__(parent)
        self._label = label
        self._base = ""
        self._pvs: dict[str, PV] = {}
        self._motor = None
        self._bridge = _PVBridge()
        self._bridge.changed.connect(self._on_pv)
        self._build_ui()

    def _build_ui(self):
        row = QHBoxLayout(self)
        row.setContentsMargins(2, 1, 2, 1)
        row.setSpacing(4)

        # Axis badge (X / Y / Z)
        axis_lbl = QLabel(self._label[0])
        axis_lbl.setFixedWidth(28)
        axis_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        axis_lbl.setObjectName("axisLabel")
        row.addWidget(axis_lbl)

        # DESC readback
        self.desc_lbl = QLabel("—")
        self.desc_lbl.setFixedWidth(110)
        self.desc_lbl.setObjectName("descLabel")
        row.addWidget(self.desc_lbl)

        # RBV tag + value
        rbv_tag = QLabel("RBV")
        rbv_tag.setObjectName("fieldTag")
        row.addWidget(rbv_tag)

        self.rbv_lbl = QLabel("—")
        self.rbv_lbl.setFixedWidth(75)
        self.rbv_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.rbv_lbl.setObjectName("rbvLabel")
        row.addWidget(self.rbv_lbl)

        # SP tag + edit
        sp_tag = QLabel("SP")
        sp_tag.setObjectName("fieldTag")
        row.addWidget(sp_tag)

        self.sp_edit = QLineEdit()
        self.sp_edit.setFixedWidth(75)
        self.sp_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.sp_edit.returnPressed.connect(self._send_sp)
        row.addWidget(self.sp_edit)

        # MOVN indicator
        self.movn_lbl = QLabel("⬤")
        self.movn_lbl.setFixedWidth(18)
        self.movn_lbl.setObjectName("movnIdle")
        row.addWidget(self.movn_lbl)

        # Step tag + edit
        step_tag = QLabel("Step")
        step_tag.setObjectName("fieldTag")
        row.addWidget(step_tag)

        self.step_edit = QLineEdit("0.100")
        self.step_edit.setFixedWidth(58)
        self.step_edit.returnPressed.connect(self._send_twv)
        row.addWidget(self.step_edit)

        # Tweak buttons
        self.rev_btn = QPushButton("◀")
        self.rev_btn.setFixedSize(26, 24)
        self.rev_btn.setObjectName("tweakBtn")
        self.rev_btn.clicked.connect(self._tweak_rev)
        row.addWidget(self.rev_btn)

        self.fwd_btn = QPushButton("▶")
        self.fwd_btn.setFixedSize(26, 24)
        self.fwd_btn.setObjectName("tweakBtn")
        self.fwd_btn.clicked.connect(self._tweak_fwd)
        row.addWidget(self.fwd_btn)

    # ── public API ────────────────────────────────────────────────────────

    def connect(self, base_pv: str):
        self._disconnect()
        self._base = base_pv
        if not (EPICS_AVAILABLE and base_pv):
            return
        try:
            monitored   = ("DESC", "RBV", "VAL", "MOVN", "TWV")
            unmonitored = ("TWR", "TWF")
            for f in monitored:
                self._pvs[f] = PV(f"{base_pv}.{f}", callback=self._bridge, auto_monitor=True)
            for f in unmonitored:
                self._pvs[f] = PV(f"{base_pv}.{f}")
            self._motor = Motor(base_pv)
        except Exception as e:
            print(f"Motor({base_pv}) connection failed: {e}")

    def get_rbv(self) -> float:
        try:
            return float(self.rbv_lbl.text())
        except ValueError:
            return 0.0

    def get_sp(self) -> float:
        try:
            return float(self.sp_edit.text())
        except ValueError:
            return 0.0

    def move_to(self, value: float):
        self.sp_edit.setText(f"{value:.3f}")
        self._send_sp()

    def is_moving(self) -> bool:
        m = self._motor
        if m is None:
            return False
        try:
            return bool(m.MOVN)
        except Exception:
            return False

    # ── private ───────────────────────────────────────────────────────────

    def _disconnect(self):
        for pv in self._pvs.values():
            try:
                pv.disconnect()
            except Exception:
                pass
        self._pvs.clear()
        self._motor = None

    @pyqtSlot(str, object)
    def _on_pv(self, pvname: str, value):
        b = self._base
        if pvname == f"{b}.DESC":
            self.desc_lbl.setText(str(value))
        elif pvname == f"{b}.RBV":
            self.rbv_lbl.setText(f"{float(value):.3f}")
        elif pvname == f"{b}.VAL":
            if not self.sp_edit.hasFocus():
                self.sp_edit.setText(f"{float(value):.3f}")
        elif pvname == f"{b}.MOVN":
            moving = bool(int(value))
            self.movn_lbl.setObjectName("movnActive" if moving else "movnIdle")
            self.movn_lbl.style().unpolish(self.movn_lbl)
            self.movn_lbl.style().polish(self.movn_lbl)
        elif pvname == f"{b}.TWV":
            if not self.step_edit.hasFocus():
                self.step_edit.setText(f"{float(value):.3f}")

    def _send_sp(self):
        pv = self._pvs.get("VAL")
        if pv is None:
            return
        try:
            pv.put(float(self.sp_edit.text()))
        except ValueError:
            pass

    def _send_twv(self):
        pv = self._pvs.get("TWV")
        if pv is None:
            return
        try:
            pv.put(float(self.step_edit.text()))
        except ValueError:
            pass

    def _tweak_rev(self):
        self._send_twv()
        pv = self._pvs.get("TWR")
        if pv:
            pv.put(1)

    def _tweak_fwd(self):
        self._send_twv()
        pv = self._pvs.get("TWF")
        if pv:
            pv.put(1)


# ── Calibration dialog ────────────────────────────────────────────────────────

class _CalibDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Pixel Calibration")
        layout = QVBoxLayout(self)

        instr = QLabel(
            "1. Click 'Select First' then click a point on the camera image.\n"
            "2. Click 'Select Second' then click a second point.\n"
            "3. Enter the known distance between the two points and click Calibrate."
        )
        instr.setWordWrap(True)
        layout.addWidget(instr)

        r1 = QHBoxLayout()
        self.first_btn = QPushButton("Select First Point")
        self.first_lbl = QLabel("(not selected)")
        r1.addWidget(self.first_btn)
        r1.addWidget(self.first_lbl)
        layout.addLayout(r1)

        r2 = QHBoxLayout()
        self.second_btn = QPushButton("Select Second Point")
        self.second_lbl = QLabel("(not selected)")
        r2.addWidget(self.second_btn)
        r2.addWidget(self.second_lbl)
        layout.addLayout(r2)

        dr = QHBoxLayout()
        dr.addWidget(QLabel("Known distance (mm):"))
        self.dist_edit = QLineEdit()
        dr.addWidget(self.dist_edit)
        layout.addLayout(dr)

        br = QHBoxLayout()
        self.ok_btn = QPushButton("Calibrate")
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        br.addWidget(self.ok_btn)
        br.addWidget(cancel_btn)
        layout.addLayout(br)


# ── Main widget ───────────────────────────────────────────────────────────────

class SampleView(QWidget):
    def __init__(self, parent=None, config: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("ASWAXS Sample View")

        # State
        self.cfg = self._load_config(config)
        self.cf = 0.00245           # mm/pixel calibration factor
        self.roisize = 60
        self.offsetFactor = 1.0 / 3.3538
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.positions: list[dict] = []
        self.calibration_flag = False
        self.calib_chosen = 1
        self.calib_pos = [[0, 0], [1, 1]]
        self.pos1 = [0, 0]
        self.pos2 = [1, 1]
        self.image: np.ndarray | None = None
        self.image_width = 1280
        self.image_height = 960
        self.image_cx = 640
        self.image_cy = 480
        self.cursor_x = 0
        self.cursor_y = 0
        self.beam_x = 640
        self.beam_y = 480
        self._acquire_pv: PV | None = None
        self._center_initialized = False

        self._build_ui()
        self._connect_signals()
        self._load_calib_file()
        self._init_overlays()
        self._apply_config()

    # ── Config ────────────────────────────────────────────────────────────

    def _load_config(self, override: dict | None) -> dict:
        cfg = dict(DEFAULT_CONFIG)
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE) as f:
                    cfg.update(json.load(f))
            except Exception as e:
                print(f"Config load error: {e}")
        if override:
            cfg.update(override)
        return cfg

    def _save_config(self):
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.cfg, f, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "Config Error", f"Could not save:\n{e}")

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(4, 4, 4, 4)
        main.addWidget(self._build_motor_bar())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_camera_tab(),    "Camera")
        self.tabs.addTab(self._build_positions_tab(), "Sample Positions")
        self.tabs.addTab(self._build_setup_tab(),     "Setup")
        main.addWidget(self.tabs)
        self._apply_style()

    def _build_motor_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("motorBar")
        v = QVBoxLayout(bar)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(0)

        self.x_motor = MotorPanel("X")
        self.y_motor = MotorPanel("Y")
        self.z_motor = MotorPanel("Z (Focus)")

        # Row 0: X motor + ROI
        row0 = QHBoxLayout()
        row0.addWidget(self.x_motor, stretch=1)
        row0.addSpacing(12)
        row0.addWidget(QLabel("ROI:"))
        self.roi_edit = QLineEdit("60")
        self.roi_edit.setFixedWidth(44)
        row0.addWidget(self.roi_edit)
        v.addLayout(row0)

        sep0 = QFrame()
        sep0.setFrameShape(QFrame.Shape.HLine)
        sep0.setFrameShadow(QFrame.Shadow.Sunken)
        sep0.setObjectName("motorSep")
        v.addWidget(sep0)

        # Row 1: Y motor + offset label
        row1 = QHBoxLayout()
        row1.addWidget(self.y_motor, stretch=1)
        row1.addSpacing(12)
        self.offset_label = QLabel("Offset: X=0.000000, Y=0.000000")
        self.offset_label.setObjectName("offsetLabel")
        row1.addWidget(self.offset_label)
        v.addLayout(row1)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setFrameShadow(QFrame.Shadow.Sunken)
        sep1.setObjectName("motorSep")
        v.addWidget(sep1)

        # Row 2: Z motor + action buttons
        row2 = QHBoxLayout()
        row2.addWidget(self.z_motor, stretch=1)
        row2.addSpacing(12)
        self.calc_offset_btn = QPushButton("Calc Offset")
        self.calc_offset_btn.setObjectName("actionBtn")
        self.calc_offset_btn.setFixedHeight(26)
        self.center_x_btn = QPushButton("Cen X")
        self.center_x_btn.setObjectName("actionBtn")
        self.center_x_btn.setFixedHeight(26)
        self.center_y_btn = QPushButton("Cen Y")
        self.center_y_btn.setObjectName("actionBtn")
        self.center_y_btn.setFixedHeight(26)
        row2.addWidget(self.calc_offset_btn)
        row2.addWidget(self.center_x_btn)
        row2.addWidget(self.center_y_btn)
        v.addLayout(row2)

        return bar

    def _build_camera_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        # Top toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        # Cal factor group
        toolbar.addWidget(QLabel("Cal. Factor:"))
        self.cf_edit = QLineEdit()
        self.cf_edit.setFixedWidth(88)
        toolbar.addWidget(self.cf_edit)
        self.calibrate_btn = QPushButton("Calibrate")
        toolbar.addWidget(self.calibrate_btn)

        sep0 = QFrame()
        sep0.setFrameShape(QFrame.Shape.VLine)
        sep0.setFrameShadow(QFrame.Shadow.Sunken)
        toolbar.addWidget(sep0)

        self.click_move_cb = QCheckBox("DoubleClick Move")
        self.click_move_cb.setChecked(True)
        toolbar.addWidget(self.click_move_cb)
        self.auto_add_cb = QCheckBox("Auto Add2List")
        toolbar.addWidget(self.auto_add_cb)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setFrameShadow(QFrame.Shadow.Sunken)
        toolbar.addWidget(sep1)

        toolbar.addWidget(QLabel("Focus:"))
        self.focus_lbl = QLabel("—")
        self.focus_lbl.setFixedWidth(65)
        self.focus_lbl.setObjectName("focusLabel")
        toolbar.addWidget(self.focus_lbl)
        self.autofocus_btn = QPushButton("Autofocus")
        toolbar.addWidget(self.autofocus_btn)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        toolbar.addWidget(sep2)

        self.cam_start_btn = QPushButton("▶ Start")
        toolbar.addWidget(self.cam_start_btn)
        self.cam_stop_btn = QPushButton("■ Stop")
        toolbar.addWidget(self.cam_stop_btn)
        self.cam_state_lbl = QLabel("Camera: —")
        self.cam_state_lbl.setObjectName("camStateLabel")
        toolbar.addWidget(self.cam_state_lbl)

        toolbar.addStretch()
        v.addLayout(toolbar)

        # Camera image (pyqtgraph graphics view)
        self.gfx = pg.GraphicsLayoutWidget()
        self.view_box = self.gfx.addViewBox(row=0, col=0)
        self.view_box.setAspectLocked(True)
        self.view_box.invertY(True)
        self.image_item = pg.ImageItem()
        self.view_box.addItem(self.image_item)
        v.addWidget(self.gfx, stretch=1)

        # Bottom status bar
        status = QHBoxLayout()
        self.cursor_lbl = QLabel("X=0, Y=0, I=0")
        self.cursor_lbl.setObjectName("statusLabel")
        status.addWidget(self.cursor_lbl)
        status.addStretch()
        self.select_beam_cb = QCheckBox("Select Beam Position")
        status.addWidget(self.select_beam_cb)
        self.beam_lbl = QLabel("BeamX=—, BeamY=—")
        self.beam_lbl.setObjectName("statusLabel")
        status.addWidget(self.beam_lbl)
        v.addLayout(status)

        return w

    def _build_positions_tab(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(6, 6, 6, 6)
        h.setSpacing(8)

        # LEFT panel
        left_w = QWidget()
        left_w.setFixedWidth(240)
        left = QVBoxLayout(left_w)
        left.setSpacing(4)
        left.setContentsMargins(0, 0, 0, 0)

        btn_row = QHBoxLayout()
        self.add_pos_btn = QPushButton("＋ Add")
        self.add_pos_btn.setObjectName("greenBtn")
        self.rem_pos_btn = QPushButton("－ Remove")
        self.rem_pos_btn.setObjectName("redBtn")
        btn_row.addWidget(self.add_pos_btn)
        btn_row.addWidget(self.rem_pos_btn)
        left.addLayout(btn_row)

        self.pos_list = QListWidget()
        self.pos_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.pos_list.setAlternatingRowColors(True)
        mono_font = QFont("Consolas")
        mono_font.setStyleHint(QFont.StyleHint.Monospace)
        mono_font.setPointSizeF(8.5)
        self.pos_list.setFont(mono_font)
        left.addWidget(self.pos_list)

        order_row = QHBoxLayout()
        self.move_up_btn = QPushButton("↑ Up")
        self.move_dn_btn = QPushButton("↓ Down")
        order_row.addWidget(self.move_up_btn)
        order_row.addWidget(self.move_dn_btn)
        left.addLayout(order_row)

        file_row = QHBoxLayout()
        self.open_pos_btn = QPushButton("📂 Open")
        self.save_pos_btn = QPushButton("💾 Save")
        file_row.addWidget(self.open_pos_btn)
        file_row.addWidget(self.save_pos_btn)
        left.addLayout(file_row)

        # Blender section
        bl_box = QGroupBox("Blender Interpolation")
        bl = QHBoxLayout(bl_box)
        bl.addWidget(QLabel("Spacing (mm):"))
        self.interp_edit = QLineEdit("1.0")
        self.interp_edit.setFixedWidth(52)
        bl.addWidget(self.interp_edit)
        self.blender_btn = QPushButton("Run")
        bl.addWidget(self.blender_btn)
        left.addWidget(bl_box)

        h.addWidget(left_w)

        # Right: pyqtgraph scatter plot
        self.pos_plot = pg.PlotWidget(title="Sample Positions")
        self.pos_plot.setBackground('#1e1e2e')
        self.pos_plot.showGrid(x=True, y=True, alpha=0.3)
        self.pos_plot.setLabel('bottom', 'X (mm)')
        self.pos_plot.setLabel('left',   'Y (mm)')
        self.pos_plot.addLegend()
        self._pos_scatter = self.pos_plot.plot(
            [], [], pen=None, symbol='o',
            symbolPen='b', symbolBrush='b', name='Positions'
        )
        self._interp_line = self.pos_plot.plot(
            [], [], pen=pg.mkPen('r', width=2), symbol=None, name='Interpolated'
        )
        h.addWidget(self.pos_plot)

        return w

    def _build_setup_tab(self) -> QWidget:
        # Outer scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)
        self._setup_edits: dict[str, QLineEdit] = {}

        def add_section(title: str, fields: list[tuple[str, str]]) -> QGroupBox:
            box = QGroupBox(title)
            g = QGridLayout(box)
            g.setSpacing(6)
            for row, (key, label) in enumerate(fields):
                lbl = QLabel(label + ":")
                lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                g.addWidget(lbl, row, 0)
                edit = QLineEdit(self.cfg.get(key, ""))
                g.addWidget(edit, row, 1)
                self._setup_edits[key] = edit
            return box

        v.addWidget(add_section("EPICS PV Configuration", [
            ("X_MOTOR_PV",       "X Motor PV"),
            ("Y_MOTOR_PV",       "Y Motor PV"),
            ("Z_MOTOR_PV",       "Z Motor PV (Focus)"),
            ("CAMERA_PREFIX",    "Camera Prefix  (e.g. Teslong:cam1:)"),
            ("IMAGE_PREFIX",     "Image Array Prefix  (e.g. Teslong:image1:)"),
            ("AUTOFOCUS_STEP",   "Autofocus Step (mm)"),
            ("AUTOFOCUS_SCRIPT", "Autofocus Script Path"),
        ]))

        v.addWidget(add_section("Blender / SSH Configuration", [
            ("BLENDER_HOST",   "SSH Host"),
            ("BLENDER_USER",   "SSH Username"),
            ("BLENDER_KEY",    "SSH Private Key File"),
            ("BLENDER_SCRIPT", "Blender Script (remote path)"),
            ("LOCAL_MOUNT",    "Local Mount Path"),
            ("REMOTE_MOUNT",   "Remote Mount Path"),
        ]))

        btn_row = QHBoxLayout()
        self.apply_btn    = QPushButton("Apply")
        self.save_cfg_btn = QPushButton("Save Config")
        btn_row.addWidget(self.apply_btn)
        btn_row.addWidget(self.save_cfg_btn)
        btn_row.addStretch()
        v.addLayout(btn_row)
        v.addStretch()

        scroll.setWidget(container)
        return scroll

    def _apply_style(self):
        self.setStyleSheet("""
        QWidget {
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            font-size: 9pt;
            color: #1e1e2e;
        }

        /* ── Motor bar frame ── */
        QFrame#motorBar {
            background: #ffffff;
            border: 1px solid #dde1e7;
            border-radius: 6px;
        }
        QFrame#motorSep {
            color: #e8eaed;
            max-height: 1px;
            margin: 1px 4px;
        }

        /* ── Axis badge ── */
        QLabel#axisLabel {
            background: #3b82f6;
            color: white;
            border-radius: 4px;
            font-weight: bold;
            font-size: 9pt;
            padding: 1px 2px;
        }

        /* ── Field tags (RBV, SP, Step) ── */
        QLabel#fieldTag {
            color: #6b7280;
            font-size: 8pt;
        }

        /* ── RBV readback ── */
        QLabel#rbvLabel {
            font-family: "Consolas", "Courier New", monospace;
            font-size: 9pt;
            color: #15803d;
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 3px;
            padding: 1px 4px;
        }

        /* ── MOVN indicator ── */
        QLabel#movnIdle   { color: #d1d5db; font-size: 11pt; }
        QLabel#movnActive { color: #f59e0b; font-size: 11pt; }

        /* ── Offset / status labels ── */
        QLabel#offsetLabel {
            color: #6b7280;
            font-style: italic;
            font-size: 8.5pt;
        }
        QLabel#focusLabel {
            font-family: "Consolas", monospace;
            color: #0369a1;
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            border-radius: 3px;
            padding: 1px 4px;
        }
        QLabel#camStateLabel {
            color: #6b7280;
            font-size: 8.5pt;
        }
        QLabel#statusLabel {
            font-family: "Consolas", monospace;
            font-size: 8.5pt;
            color: #374151;
        }

        /* ── Buttons ── */
        QPushButton {
            background: #f3f4f6;
            border: 1px solid #d1d5db;
            border-radius: 4px;
            padding: 3px 10px;
            min-height: 24px;
            color: #1f2937;
        }
        QPushButton:hover  { background: #dbeafe; border-color: #93c5fd; }
        QPushButton:pressed { background: #3b82f6; color: white; border-color: #2563eb; }
        QPushButton:disabled { color: #9ca3af; background: #f9fafb; }

        QPushButton#tweakBtn {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 3px;
            padding: 1px 2px;
            font-size: 9pt;
        }
        QPushButton#tweakBtn:hover  { background: #bfdbfe; }
        QPushButton#tweakBtn:pressed { background: #3b82f6; color: white; }

        QPushButton#actionBtn {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 4px;
            color: #14532d;
            padding: 3px 8px;
        }
        QPushButton#actionBtn:hover  { background: #dcfce7; border-color: #86efac; }
        QPushButton#actionBtn:pressed { background: #16a34a; color: white; }

        QPushButton#greenBtn {
            background: #f0fdf4; border: 1px solid #86efac;
            color: #166534; border-radius: 4px;
        }
        QPushButton#greenBtn:hover  { background: #dcfce7; }
        QPushButton#greenBtn:pressed { background: #16a34a; color: white; }

        QPushButton#redBtn {
            background: #fef2f2; border: 1px solid #fca5a5;
            color: #991b1b; border-radius: 4px;
        }
        QPushButton#redBtn:hover  { background: #fee2e2; }
        QPushButton#redBtn:pressed { background: #dc2626; color: white; }

        /* ── Line edits ── */
        QLineEdit {
            background: white;
            border: 1px solid #d1d5db;
            border-radius: 3px;
            padding: 2px 5px;
            selection-background-color: #3b82f6;
        }
        QLineEdit:focus { border-color: #3b82f6; }

        /* ── Tab widget ── */
        QTabWidget::pane {
            border: 1px solid #e5e7eb;
            border-radius: 0 4px 4px 4px;
            background: #fafafa;
        }
        QTabBar::tab {
            background: #f3f4f6;
            border: 1px solid #e5e7eb;
            border-bottom: none;
            border-radius: 4px 4px 0 0;
            padding: 5px 16px;
            margin-right: 2px;
            color: #6b7280;
        }
        QTabBar::tab:selected {
            background: #fafafa;
            color: #1e40af;
            font-weight: bold;
            border-bottom: 2px solid #3b82f6;
        }
        QTabBar::tab:hover:!selected { background: #dbeafe; color: #1d4ed8; }

        /* ── Group boxes ── */
        QGroupBox {
            border: 1px solid #e5e7eb;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 6px;
            font-weight: bold;
            color: #374151;
            background: #fafafa;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 6px;
            left: 10px;
            color: #4b5563;
        }

        /* ── List widget ── */
        QListWidget {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
            alternate-background-color: #f8faff;
            outline: none;
        }
        QListWidget::item { padding: 3px 6px; }
        QListWidget::item:selected {
            background: #dbeafe;
            color: #1e40af;
        }
        QListWidget::item:hover:!selected { background: #f0f9ff; }

        /* ── Check boxes ── */
        QCheckBox { spacing: 5px; }
        QCheckBox::indicator {
            width: 14px; height: 14px;
            border: 1px solid #d1d5db;
            border-radius: 3px;
            background: white;
        }
        QCheckBox::indicator:checked {
            background: #3b82f6;
            border-color: #2563eb;
            image: none;
        }

        /* ── Scroll area ── */
        QScrollArea { border: none; background: transparent; }

        /* ── Vertical separators ── */
        QFrame[frameShape="5"] { color: #e5e7eb; max-width: 1px; }
    """)

    # ── Signal wiring ─────────────────────────────────────────────────────

    def _connect_signals(self):
        # Motor bar
        self.calc_offset_btn.clicked.connect(self.calcROI)
        self.center_x_btn.clicked.connect(self.centerX)
        self.center_y_btn.clicked.connect(self.centerY)
        self.roi_edit.returnPressed.connect(self.roiSizeChanged)

        # Camera tab
        self.cf_edit.returnPressed.connect(self.cfChanged)
        self.calibrate_btn.clicked.connect(self.openCalibration)
        self.autofocus_btn.clicked.connect(self.runAutofocus)
        self.cam_start_btn.clicked.connect(self._cam_start)
        self.cam_stop_btn.clicked.connect(self._cam_stop)

        # Positions tab
        self.add_pos_btn.clicked.connect(self.addPosition)
        self.rem_pos_btn.clicked.connect(self.removePositions)
        self.move_up_btn.clicked.connect(self.moveUp)
        self.move_dn_btn.clicked.connect(self.moveDown)
        self.open_pos_btn.clicked.connect(self.openPositions)
        self.save_pos_btn.clicked.connect(self.savePositions)
        self.pos_list.itemDoubleClicked.connect(self.moveOnListClicked)
        self.blender_btn.clicked.connect(self.blenderInterpolate)

        # Setup tab
        self.apply_btn.clicked.connect(self.applySetup)
        self.save_cfg_btn.clicked.connect(self.saveSetup)

        # Camera mouse events (pyqtgraph scene signals)
        scene = self.view_box.scene()
        scene.sigMouseMoved.connect(self.image_mouse_moved)
        scene.sigMouseClicked.connect(self.mouse_clicked)

    # ── EPICS connection ──────────────────────────────────────────────────

    def _apply_config(self):
        """Connect all EPICS PVs from self.cfg. Safe to call multiple times."""
        # Disconnect old camera PVs if reconnecting
        for attr in ('_img_pv', '_wid_pv', '_state_pv'):
            old = getattr(self, attr, None)
            if old is not None:
                try:
                    old.disconnect()
                except Exception:
                    pass

        self.x_motor.connect(self.cfg["X_MOTOR_PV"])
        self.y_motor.connect(self.cfg["Y_MOTOR_PV"])
        self.z_motor.connect(self.cfg["Z_MOTOR_PV"])

        if not EPICS_AVAILABLE:
            self.cam_state_lbl.setText("Camera: offline (no EPICS)")
            return

        try:
            self._img_bridge   = _PVBridge()
            self._wid_bridge   = _PVBridge()
            self._state_bridge = _PVBridge()
            self._img_bridge.changed.connect(self._on_image_data)
            self._wid_bridge.changed.connect(self._on_width_data)
            self._state_bridge.changed.connect(self._on_cam_state)

            cam = self.cfg["CAMERA_PREFIX"]
            img = self.cfg["IMAGE_PREFIX"]

            self._img_pv   = PV(img + "ArrayData",        callback=self._img_bridge,   auto_monitor=True)
            self._wid_pv   = PV(cam + "ArraySizeX_RBV",   callback=self._wid_bridge,   auto_monitor=True)
            self._state_pv = PV(cam + "DetectorState_RBV", callback=self._state_bridge, auto_monitor=True)

            if self._acquire_pv is None:
                self._acquire_pv = PV(cam + "Acquire")
                atexit.register(self._stop_acquire)
            self._acquire_pv.put(1)
        except Exception as e:
            print(f"Camera PV connection failed: {e}")
            self.cam_state_lbl.setText("Camera: PV error")

    def _stop_acquire(self):
        if self._acquire_pv:
            try:
                self._acquire_pv.put(0)
            except Exception:
                pass

    # ── Camera PV callbacks (main thread via Qt signal) ───────────────────

    @pyqtSlot(str, object)
    def _on_width_data(self, _pvname: str, value):
        self.image_width = int(value)

    @pyqtSlot(str, object)
    def _on_cam_state(self, _pvname: str, value):
        self.cam_state_lbl.setText(f"Camera: {value}")

    @pyqtSlot(str, object)
    def _on_image_data(self, _pvname: str, value):
        try:
            raw = np.asarray(value, dtype=np.uint8)
            w = self.image_width
            if w <= 0 or raw.size == 0:
                return
            h = raw.size // (w * 3)
            if h <= 0 or raw.size != h * w * 3:
                return
            self.image_height = h
            bgr = raw.reshape((h, w, 3))
            if CV2_AVAILABLE:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                gray = bgr.mean(axis=2).astype(np.uint8)
                rgb  = bgr[:, :, ::-1]          # BGR → RGB via slice
            self.image = gray
            self.image_item.setImage(rgb, autoLevels=False, levels=(0, 255))

            if not self._center_initialized:
                self.image_cx = w // 2
                self.image_cy = h // 2
                self._center_x_line.setValue(w / 2)
                self._center_y_line.setValue(h / 2)
                self._center_initialized = True

            if CV2_AVAILABLE:
                fp = cv2.Laplacian(gray, cv2.CV_64F).var()
                self.focus_lbl.setText(f"{fp:.3f}")
            else:
                self.focus_lbl.setText("(cv2 N/A)")
        except Exception as e:
            print(f"Image process error: {e}")

    # ── Overlay graphics ──────────────────────────────────────────────────

    def _init_overlays(self):
        w = self.image_width
        self._center_x_line = pg.InfiniteLine(pos=w / 2, angle=90, pen=pg.mkPen('r', width=1), movable=False)
        self._center_y_line = pg.InfiniteLine(pos=0,     angle=0,  pen=pg.mkPen('r', width=1), movable=False)
        self._cursor_x_line = pg.InfiniteLine(pos=0,     angle=90, pen=pg.mkPen('b', width=1), movable=False)
        self._cursor_y_line = pg.InfiniteLine(pos=0,     angle=0,  pen=pg.mkPen('b', width=1), movable=False)
        self._calib_pts     = pg.ScatterPlotItem()
        self._beam_x_curve  = pg.PlotCurveItem(pen=pg.mkPen('g', width=2))
        self._beam_y_curve  = pg.PlotCurveItem(pen=pg.mkPen('g', width=2))

        for item in (self._center_x_line, self._center_y_line,
                     self._cursor_x_line, self._cursor_y_line,
                     self._calib_pts, self._beam_x_curve, self._beam_y_curve):
            self.view_box.addItem(item)

        self._cursor_x_line.hide()
        self._cursor_y_line.hide()
        self._calib_pts.hide()
        self._beam_x_curve.hide()
        self._beam_y_curve.hide()

    # ── Camera controls ───────────────────────────────────────────────────

    def _cam_start(self):
        if self._acquire_pv:
            self._acquire_pv.put(1)

    def _cam_stop(self):
        if self._acquire_pv:
            self._acquire_pv.put(0)

    # ── Mouse events ──────────────────────────────────────────────────────

    @pyqtSlot(object)
    def image_mouse_moved(self, pos):
        try:
            coords = self.image_item.mapFromScene(pos)
            x, y = int(coords.x()), int(coords.y())
        except Exception:
            return
        if self.image is not None and 0 <= x < self.image_width and 0 <= y < self.image_height:
            self.cursor_x, self.cursor_y = x, y
            i = int(self.image[y, x])
            self.cursor_lbl.setText(f"X={x:4d}, Y={y:4d}, I={i:3d}")
            self._cursor_x_line.setValue(x)
            self._cursor_y_line.setValue(y)
            self._cursor_x_line.show()
            self._cursor_y_line.show()
        else:
            self._cursor_x_line.hide()
            self._cursor_y_line.hide()

    @pyqtSlot(object)
    def mouse_clicked(self, event):
        x, y = self.cursor_x, self.cursor_y
        if self.image is None:
            return

        if event._double and self.click_move_cb.isChecked():
            if 0 <= x < self.image_width and 0 <= y < self.image_height:
                new_x = self.x_motor.get_sp() - self.cf * (x - self.image_width  / 2 - 1)
                new_y = self.y_motor.get_sp() + self.cf * (y - self.image_height / 2 + 1)
                self.x_motor.move_to(new_x)
                self.y_motor.move_to(new_y)
                while self.x_motor.is_moving() or self.y_motor.is_moving():
                    QApplication.processEvents()
            if self.auto_add_cb.isChecked():
                self.addPosition()

        elif self.calibration_flag:
            if self.calib_chosen == 1:
                self.pos1 = [x, y]
                self.calib_pos[0] = [x, y]
                if hasattr(self, '_calib_dialog'):
                    self._calib_dialog.first_lbl.setText(f"x: {x}, y: {y}")
            else:
                self.pos2 = [x, y]
                self.calib_pos[1] = [x, y]
                if hasattr(self, '_calib_dialog'):
                    self._calib_dialog.second_lbl.setText(f"x: {x}, y: {y}")
            self._calib_pts.setData(
                pos=self.calib_pos, size=10, symbol='o', pen=pg.mkPen('red')
            )

        if self.select_beam_cb.isChecked():
            self.beam_x, self.beam_y = x, y
            self.beam_lbl.setText(f"BeamX={x:4d}, BeamY={y:4d}")
            self._beam_x_curve.setData(x=[x, x], y=[y - 10, y + 10])
            self._beam_y_curve.setData(x=[x - 10, x + 10], y=[y, y])
            self._beam_x_curve.show()
            self._beam_y_curve.show()

    # ── Calibration factor ────────────────────────────────────────────────

    def _load_calib_file(self):
        try:
            with open(CALIB_FILE) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    if line.startswith('cf='):
                        self.cf = float(line.split('=')[1])
                        break
        except Exception:
            pass
        self.cf_edit.setText(f"{self.cf:.6f}")

    def cfChanged(self):
        try:
            self.cf = float(self.cf_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Value Error", "Enter a number only.")
            self.cf_edit.setText(f"{self.cf:.6f}")
            return
        try:
            os.makedirs(os.path.dirname(CALIB_FILE), exist_ok=True)
            with open(CALIB_FILE, 'w') as f:                   # BUG FIX: was 'r'
                f.write(f"#Calibration saved {time.asctime()}\n")
                f.write(f"cf={self.cf:.6f}\n")
        except Exception as e:
            QMessageBox.warning(self, "File Error", f"Could not save calibration:\n{e}")

    # ── Calibration widget ────────────────────────────────────────────────

    def openCalibration(self):
        self.calib_pos = [
            [self.image_cx, self.image_cy],
            [self.image_cx + 50, self.image_cy + 50],
        ]
        self.calibration_flag = True
        self.calib_chosen = 1
        self._calib_pts.show()

        dlg = _CalibDialog(self)
        self._calib_dialog = dlg
        dlg.first_btn.clicked.connect(lambda: self._select_calib_pos(1))
        dlg.second_btn.clicked.connect(lambda: self._select_calib_pos(2))
        dlg.ok_btn.clicked.connect(lambda: self._do_calibrate(dlg))
        dlg.rejected.connect(self._cancel_calibration)
        dlg.show()

    def _select_calib_pos(self, n: int):
        self.calib_chosen = n

    def _cancel_calibration(self):
        self.calibration_flag = False
        self._calib_pts.hide()

    def _do_calibrate(self, dlg: _CalibDialog):
        try:
            known = float(dlg.dist_edit.text())
        except ValueError:
            QMessageBox.critical(dlg, "Value Error", "Enter a numeric distance.")
            return
        dx = self.pos1[0] - self.pos2[0]
        dy = self.pos1[1] - self.pos2[1]
        pixel_dist = np.sqrt(dx ** 2 + dy ** 2)
        if pixel_dist < 1.0:
            QMessageBox.warning(dlg, "Error", "Points are too close together.")
            return
        self.cf = round(known / pixel_dist, 6)
        self.cf_edit.setText(f"{self.cf:.6f}")
        self.cfChanged()
        self.calibration_flag = False
        self._calib_pts.hide()
        dlg.accept()

    # ── ROI / beam centering ──────────────────────────────────────────────

    def roiSizeChanged(self):
        try:
            self.roisize = int(self.roi_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Value Error", "Integer only.")
            self.roi_edit.setText(str(self.roisize))  # BUG FIX: was float(...)

    def calcROI(self):
        if self.image is None:
            QMessageBox.warning(self, "No Image", "Camera not streaming.")
            return
        cx, cy, r = self.image_cx, self.image_cy, self.roisize
        roi1 = self.image[cy - r:cy,   cx - r:cx  ]
        roi2 = self.image[cy - r:cy,   cx     :cx + r]
        roi3 = self.image[cy     :cy + r, cx     :cx + r]
        roi4 = self.image[cy     :cy + r, cx - r:cx  ]

        int_max = max(r_.max() for r_ in (roi1, roi2, roi3, roi4))
        rois = [np.abs(r_ - int_max) for r_ in (roi1, roi2, roi3, roi4)]
        int_max2 = max(r_.max() for r_ in rois)
        thresh = 0.1 * int_max2

        s = [np.sum(r_ > thresh) for r_ in rois]   # s[0..3] = roi1..roi4
        right = s[1] + s[2]
        left  = s[0] + s[3]
        top   = s[0] + s[1]
        bot   = s[2] + s[3]
        total = left + right
        if total == 0:
            return
        self.x_offset = (right - left) * self.offsetFactor / total
        self.y_offset = (top   - bot ) * self.offsetFactor / total
        self.offset_label.setText(
            f"Offset: X={self.x_offset:.6f}, Y={self.y_offset:.6f}"
        )

    def centerX(self):
        self.calcROI()
        if abs(self.x_offset) > 0.005:
            self.x_motor.move_to(self.x_motor.get_sp() - self.x_offset)
            while self.x_motor.is_moving():
                QApplication.processEvents()
        self.calcROI()

    def centerY(self):
        self.calcROI()
        if abs(self.y_offset) > 0.005:
            self.y_motor.move_to(self.y_motor.get_sp() - self.y_offset)
            while self.y_motor.is_moving():
                QApplication.processEvents()
        self.calcROI()

    # ── Autofocus ─────────────────────────────────────────────────────────

    def runAutofocus(self):
        script = self.cfg.get("AUTOFOCUS_SCRIPT", "")
        if not os.path.exists(script):
            QMessageBox.warning(self, "Autofocus", f"Script not found:\n{script}")
            return
        cam  = self.cfg["CAMERA_PREFIX"]
        zpv  = self.cfg["Z_MOTOR_PV"]
        step = self.cfg.get("AUTOFOCUS_STEP", "0.2")
        subprocess.Popen([sys.executable, script, cam, zpv, step])

    # ── Position management ───────────────────────────────────────────────

    def addPosition(self):
        pos = {
            "X": round(self.x_motor.get_rbv(), 3),
            "Y": round(self.y_motor.get_rbv(), 3),
            "Z": round(self.z_motor.get_rbv(), 3),
        }
        selected = self.pos_list.selectedItems()
        if selected:
            idx = self.pos_list.row(selected[-1]) + 1
            self.positions.insert(idx, pos)
        else:
            self.positions.append(pos)
        self._refresh_pos_list()

    def removePositions(self):
        rows = sorted(
            [self.pos_list.row(i) for i in self.pos_list.selectedItems()],
            reverse=True,
        )
        for r in rows:
            self.positions.pop(r)
        self._refresh_pos_list()

    def moveUp(self):
        rows = [self.pos_list.row(i) for i in self.pos_list.selectedItems()]
        for r in sorted(rows):
            if r > 0:
                self.positions.insert(r - 1, self.positions.pop(r))
        self._refresh_pos_list()

    def moveDown(self):
        rows = sorted(
            [self.pos_list.row(i) for i in self.pos_list.selectedItems()],
            reverse=True,
        )
        for r in rows:
            if r < len(self.positions) - 1:
                self.positions.insert(r + 1, self.positions.pop(r))
        self._refresh_pos_list()

    def _refresh_pos_list(self):
        self.pos_list.clear()
        for i, p in enumerate(self.positions):
            self.pos_list.addItem(
                f"{i:3d}:  X={p['X']:8.3f}  Y={p['Y']:8.3f}  Z={p['Z']:8.3f}"
            )
        self._plot_positions()

    def moveOnListClicked(self, item):
        row = self.pos_list.row(item)
        p = self.positions[row]
        self.x_motor.move_to(p["X"])
        self.y_motor.move_to(p["Y"])
        self.z_motor.move_to(p["Z"])

    def savePositions(self):
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save Positions", "", "Position Files (*.pos)"
        )
        if not fname:
            return
        arr = self._positions_to_array()
        np.savetxt(fname, arr, fmt="%.3f", header="X Y Z")

    def openPositions(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Positions", "", "Position Files (*.pos)"
        )
        if not fname:
            return
        self.positions = []
        with open(fname) as f:
            lines = f.readlines()

        header = lines[0].lstrip('#').strip().split()
        old_format = any('ca://' in h for h in header)  # detect legacy PyDM keys

        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            vals = [float(v) for v in line.split()]
            if len(vals) < 2:
                continue
            # Both old and new format store X, Y, Z in columns 0, 1, 2
            self.positions.append({
                "X": round(vals[0], 3),
                "Y": round(vals[1], 3),
                "Z": round(vals[2], 3) if len(vals) > 2 else 0.0,
            })
        self._refresh_pos_list()

    def _positions_to_array(self) -> np.ndarray:
        if not self.positions:
            return np.empty((0, 3))
        return np.array([[p["X"], p["Y"], p["Z"]] for p in self.positions])

    def _plot_positions(self):
        arr = self._positions_to_array()
        if arr.size == 0:
            self._pos_scatter.setData([], [])
            return
        self._pos_scatter.setData(arr[:, 0], arr[:, 1])

    # ── Blender interpolation ─────────────────────────────────────────────

    def blenderInterpolate(self):
        if not self.positions:
            QMessageBox.warning(self, "No Positions", "Add sample positions first.")
            return

        try:
            spacing = float(self.interp_edit.text())   # BUG FIX: was eval()
        except ValueError:
            QMessageBox.warning(self, "Value Error", "Spacing must be a number.")
            return

        local_mount  = self.cfg.get("LOCAL_MOUNT", "")
        remote_mount = self.cfg.get("REMOTE_MOUNT", "")

        data_dir = os.path.join(_DIR, "Data")
        os.makedirs(data_dir, exist_ok=True)
        temp_pos = os.path.join(data_dir, "temp.pos")

        arr = self._positions_to_array()
        np.savetxt(temp_pos, arr, fmt="%.3f", header="X Y Z")

        lifname = os.path.abspath(temp_pos)
        ifname  = lifname.replace(local_mount, remote_mount).replace('\\', '/')

        lofname, _ = QFileDialog.getSaveFileName(
            self, "Save Interpolated Output", data_dir,
            "CSV Files (*.csv);;Text Files (*.txt)"
        )
        if not lofname:
            _safe_remove(temp_pos)
            return

        if not os.path.splitext(lofname)[1]:
            lofname += ".csv"
        ofname = lofname.replace(local_mount, remote_mount).replace('\\', '/')

        blender_script = self.cfg.get("BLENDER_SCRIPT", "")
        hostname = self.cfg.get("BLENDER_HOST", "")
        username = self.cfg.get("BLENDER_USER", "")
        key_file = self.cfg.get("BLENDER_KEY", "")
        cmd = ['blender', '--background', '--python', blender_script,
               '--', ifname, ofname, f"{spacing:.2f}"]

        try:
            local_ip = socket.gethostbyaddr(socket.gethostname())[2][0]
        except Exception:
            local_ip = ""

        success = False
        if local_ip != hostname:
            success = self._run_blender_ssh(hostname, username, key_file, cmd)
        else:
            success = self._run_blender_local(cmd)

        _safe_remove(temp_pos)

        if not success:
            return

        if os.path.exists(lofname):
            try:
                data = np.loadtxt(lofname, comments='#', delimiter=',')
                if data.ndim == 2 and data.shape[1] >= 2:
                    self._interp_line.setData(data[:, 0], data[:, 1])
            except Exception as e:
                QMessageBox.warning(self, "Load Error", f"Could not load output:\n{e}")

    def _run_blender_ssh(self, hostname, username, key_file, cmd) -> bool:
        if not PARAMIKO_AVAILABLE:
            QMessageBox.critical(self, "SSH Error",
                                 "paramiko is not installed.\nRun: pip install paramiko")
            return False
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(hostname, port=22, username=username, key_filename=key_file)
            stdin, stdout, stderr = client.exec_command(" ".join(cmd))
            out = stdout.read().decode()
            err = stderr.read().decode()
            if out:
                print(out)
            if err:
                print("stderr:", err)
            return True
        except Exception as e:
            QMessageBox.critical(self, "SSH Error", str(e))
            return False
        finally:
            client.close()

    def _run_blender_local(self, cmd) -> bool:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()
            print(out.decode())
            if err:
                print("stderr:", err.decode())
            return True
        except Exception as e:
            QMessageBox.critical(self, "Blender Error", str(e))
            return False

    # ── Setup tab handlers ────────────────────────────────────────────────

    def applySetup(self):
        for key, edit in self._setup_edits.items():
            self.cfg[key] = edit.text().strip()
        self._apply_config()

    def saveSetup(self):
        self.applySetup()
        self._save_config()
        QMessageBox.information(self, "Saved", "Configuration saved to file.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_remove(path: str):
    try:
        os.remove(path)
    except Exception:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SampleView()
    w.resize(1400, 900)
    w.show()
    sys.exit(app.exec())
