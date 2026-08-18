"""
sample_station.py — ASWAXS Sample Station
Combines live camera/motor control with rich sample position planning.
"""

import os, sys, json, csv, re, time, socket, subprocess, atexit, tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QCheckBox, QTabWidget,
    QFileDialog, QMessageBox, QSplitter, QApplication,
    QAbstractItemView, QDialog, QDialogButtonBox, QFrame,
    QScrollArea, QTableWidget, QTableWidgetItem, QComboBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QSizePolicy,
    QMainWindow, QMenu,
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, pyqtSlot, QObject, QEvent,
    QItemSelection, QItemSelectionModel, QRectF,
    QThread, QTimer,
)
from PyQt6.QtGui import QFont, QPainter, QColor, QPen, QBrush, QKeySequence, QShortcut

pg.setConfigOption('imageAxisOrder', 'row-major')

try:
    from epics import Motor, PV
    import epics.ca as _ca
    _ca.initialize_libca()
    EPICS_AVAILABLE = True
except Exception as _epics_err:
    EPICS_AVAILABLE = False
    Motor = None
    PV = None
    print(f"Warning: EPICS unavailable ({_epics_err}) — running in offline mode")

# ── Constants ──────────────────────────────────────────────────────────────────

_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(_DIR, "sample_station_config.json")
CALIB_FILE  = os.path.join(_DIR, "Data", "camera_calib.txt")

DEFAULT_CONFIG = {
    "X_MOTOR_PV":       "15IDD:m19",
    "Y_MOTOR_PV":       "15IDD:m18",
    "Z_MOTOR_PV":       "15IDD:m7",
    "CAMERA_PREFIX":    "Teslong:cam1:",
    "IMAGE_PREFIX":     "Teslong:image1:",
    "AUTOFOCUS_STEP":   "0.2",
    "AUTOFOCUS_SCRIPT": os.path.join(_DIR, "autofocus.py"),
    "BLENDER_HOST":     "164.54.169.92",
    "BLENDER_USER":     "chem_epics",
    "BLENDER_KEY":      "/home/chem_epics/.ssh/mykey",
    "BLENDER_EXE":      "blender",
    "BLENDER_SCRIPT":   "/home/chem_epics/cars6/Data/chemmat/ASWAXS/ASWAXS/Scripts/Blender_Macro.py",
    "LOCAL_MOUNT":      "/home/chem_epics/cars6/Data",
    "REMOTE_MOUNT":     "/home/chem_epics/cars6/Data",
}

# ── Position data model ────────────────────────────────────────────────────────

POSITION_FIELDS = ["name", "x", "y", "z", "role", "layout", "group", "solvent_group", "note"]
NUMERIC_FIELDS  = {"x", "y", "z"}
DEFAULT_ROLE    = "Sample"
DEFAULT_LAYOUT  = "freeform"
ROLE_PRESETS    = ["Sample", "Solvent", "GC", "Air", "Empty", "Standard",
                   "Background", "Inlet", "Outlet", "Channel", "Observation", "Skip"]
ROLE_COLORS = {
    "Sample":       "#4C78A8",
    "Solvent":      "#54A24B",
    "Empty":        "#BAB0AC",
    "Standard":     "#F58518",
    "Air":          "#E45756",
    "GC":           "#B279A2",
    "Background":   "#72B7B2",
    "Inlet":        "#1F77B4",
    "Outlet":       "#D62728",
    "Channel":      "#59A14F",
    "Observation":  "#EDC948",
    "Skip":         "#D8DCE3",
    "Interpolated": "#A855F7",
}


@dataclass
class PositionRecord:
    name: str = ""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    role: str = DEFAULT_ROLE
    layout: str = DEFAULT_LAYOUT
    group: str = ""
    solvent_group: str = ""
    note: str = ""

    @classmethod
    def from_mapping(cls, data: dict, index: int | None = None) -> "PositionRecord":
        normalized = {f: data.get(f, "") for f in POSITION_FIELDS}
        for f in NUMERIC_FIELDS:
            normalized[f] = _flt(normalized.get(f))
        if not str(normalized["name"]).strip():
            normalized["name"] = f"pos_{index+1}" if index is not None else "pos"
        for f in ("role", "layout"):
            if not str(normalized[f]).strip():
                normalized[f] = DEFAULT_ROLE if f == "role" else DEFAULT_LAYOUT
        return cls(
            name=str(normalized["name"]).strip(),
            x=float(normalized["x"]), y=float(normalized["y"]), z=float(normalized["z"]),
            role=str(normalized["role"]).strip(), layout=str(normalized["layout"]).strip(),
            group=str(normalized["group"]).strip(),
            solvent_group=str(normalized["solvent_group"]).strip(),
            note=str(normalized["note"]).strip(),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _flt(v: Any) -> float:
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def normalize_positions(rows: list) -> list:
    return [r.to_dict() if isinstance(r, PositionRecord)
            else PositionRecord.from_mapping(r, index=i).to_dict()
            for i, r in enumerate(rows)]


def blank_position(index: int = 0, *, layout: str = DEFAULT_LAYOUT) -> dict:
    return PositionRecord(name=f"pos_{index+1}", layout=layout).to_dict()


# ── I/O helpers ────────────────────────────────────────────────────────────────

CSV_ALIASES = {
    "s_x": "x", "sx": "x", "sp_x": "x", "sample_x": "x", "motor_x": "x",
    "vertex_x": "x", "vertexx": "x",
    "s_y": "y", "sy": "y", "sp_y": "y", "sample_y": "y", "motor_y": "y",
    "vertex_y": "y", "vertexy": "y",
    "s_z": "z", "sz": "z", "sp_z": "z", "sample_z": "z", "motor_z": "z",
    "vertex_z": "z", "vertexz": "z",
    "comment": "note", "notes": "note",
    "solvent": "solvent_group", "solventgroup": "solvent_group",
    "solvent_group_index": "solvent_group",
}


def _canonical_field(v: str) -> str:
    k = v.strip().lstrip("#").lower().replace(" ", "_").replace("-", "_")
    return CSV_ALIASES.get(k, k)


def _split_row(v: str) -> list:
    if "," in v:
        rows = list(csv.reader([v]))[0]
        return [x.strip() for x in rows]
    return [x for x in re.split(r'[\t,; ]+', v.strip()) if x]


def _layout_from_header(header: list) -> str:
    cf = [_canonical_field(h) for h in header]
    has_xyz    = any(f in cf for f in ("x", "y", "z"))
    has_normal = any(f.startswith("normal") for f in cf)
    if has_xyz and has_normal:
        return "blender_interpolated"
    return "freeform"


def _role_from_header(header: list) -> str:
    cf = [_canonical_field(h) for h in header]
    if any(f.startswith("normal") for f in cf):
        return "Interpolated"
    return "Sample"


def _load_csv(path) -> list:
    result = []
    header = None
    default_layout = "freeform"
    default_role   = "Sample"
    with open(path, newline='', encoding='utf-8', errors='replace') as f:
        for line in f:
            stripped = line.rstrip('\n').strip()
            if not stripped:
                continue
            if stripped.startswith('#') and any(c.isalpha() for c in stripped[1:]):
                raw_parts = _split_row(stripped.lstrip('#').strip())
                header = [_canonical_field(h) for h in raw_parts]
                default_layout = _layout_from_header(raw_parts)
                default_role   = _role_from_header(raw_parts)
                continue
            if stripped.startswith('#'):
                continue
            parts = _split_row(stripped)
            if not parts:
                continue
            if header:
                row = {header[i]: parts[i] for i in range(min(len(header), len(parts)))}
            else:
                row = {}
                for i, f in enumerate(["x", "y", "z"]):
                    if i < len(parts):
                        row[f] = parts[i]
            row.setdefault("layout", default_layout)
            row.setdefault("role",   default_role)
            result.append(row)
    return normalize_positions(result)


def _save_csv(path, positions) -> None:
    positions = normalize_positions(positions)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=POSITION_FIELDS)
        writer.writeheader()
        for pos in positions:
            writer.writerow(pos)


def _load_json(path) -> list:
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        rows = data.get("positions", [])
    else:
        rows = data
    return normalize_positions(rows)


def _save_json(path, positions) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(
            {"schema": "aswaxs-sample-position-v1",
             "positions": normalize_positions(positions)},
            f, indent=2,
        )


def _load_pos(path) -> list:
    result = []
    header = None
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('#'):
                header = stripped.lstrip('#').split()
                continue
            parts = (stripped.split() if ',' not in stripped
                     else [x.strip() for x in stripped.split(',')])
            if not parts:
                continue
            row: dict = {}
            if header and len(header) == len(parts):
                row = {_canonical_field(header[i]): parts[i] for i in range(len(parts))}
            else:
                for i, f in enumerate(["x", "y", "z"]):
                    if i < len(parts):
                        row[f] = parts[i]
            row.setdefault("layout", "pos")
            result.append(row)
    return normalize_positions(result)


def _save_pos(path, positions) -> None:
    positions = normalize_positions(positions)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("# x y z\n")
        for pos in positions:
            f.write(f"{float(pos['x']):.6f} {float(pos['y']):.6f} {float(pos['z']):.6f}\n")


def load_positions(path) -> list:
    path = str(path)
    suffix = os.path.splitext(path)[1].lower()
    if suffix == '.json':
        return _load_json(path)
    elif suffix == '.pos':
        return _load_pos(path)
    else:
        return _load_csv(path)


def save_positions(path, positions: list) -> None:
    path = str(path)
    suffix = os.path.splitext(path)[1].lower()
    if suffix == '.json':
        _save_json(path, positions)
    elif suffix == '.pos':
        _save_pos(path, positions)
    else:
        _save_csv(path, positions)


def export_bluesky_csv(path, positions: list) -> None:
    positions = normalize_positions(positions)
    fieldnames = ["s_x", "s_y", "s_z", "name", "role", "layout",
                  "group", "solvent_group", "note"]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pos in positions:
            writer.writerow({
                "s_x": pos["x"], "s_y": pos["y"], "s_z": pos["z"],
                "name": pos["name"], "role": pos["role"], "layout": pos["layout"],
                "group": pos["group"], "solvent_group": pos["solvent_group"],
                "note": pos["note"],
            })


def export_reducer_pairs_csv(path, positions: list) -> None:
    positions = normalize_positions(positions)
    fieldnames = ["name", "sample_group", "solvent_group",
                  "sample_x", "sample_y", "sample_z", "role"]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pos in positions:
            if str(pos.get("role", "")).strip() == "Sample":
                writer.writerow({
                    "name":          pos["name"],
                    "sample_group":  pos.get("group", ""),
                    "solvent_group": pos["solvent_group"],
                    "sample_x":      pos["x"],
                    "sample_y":      pos["y"],
                    "sample_z":      pos["z"],
                    "role":          pos["role"],
                })


# ── Thread-safe EPICS → Qt bridge ─────────────────────────────────────────────

class _PVBridge(QObject):
    changed    = pyqtSignal(str, object)
    conn_state = pyqtSignal(str, bool)   # pvname, connected

    def __call__(self, pvname=None, value=None, **_kw):
        if value is not None:
            self.changed.emit(str(pvname or ""), value)

    def conn_cb(self, pvname=None, conn=True, **_kw):
        self.conn_state.emit(str(pvname or ""), bool(conn))


# ── Single-motor control panel ─────────────────────────────────────────────────

class MotorPanel(QObject):
    """Motor EPICS logic — exposes sub-widgets for placement in a shared QGridLayout."""

    def __init__(self, label="Motor", parent=None):
        super().__init__(parent)
        self._label = label
        self._base = ""
        self._pvs: dict[str, PV] = {}
        self._motor = None
        self._bridge = _PVBridge()
        self._bridge.changed.connect(self._on_pv)
        self._bridge.conn_state.connect(self._on_conn)
        self._build_widgets()

    def _build_widgets(self):
        # Axis badge — doubles as PV connection indicator
        self.axis_lbl = QLabel(self._label[0])
        self.axis_lbl.setFixedWidth(28)
        self.axis_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.axis_lbl.setObjectName("axisLabel")
        self._set_conn("idle")

        # DESC readback
        self.desc_lbl = QLabel("—")
        self.desc_lbl.setFixedWidth(110)
        self.desc_lbl.setObjectName("descLabel")

        # RBV tag + value
        self.rbv_tag = QLabel("RBV")
        self.rbv_tag.setFixedWidth(28)
        self.rbv_tag.setObjectName("fieldTag")

        self.rbv_lbl = QLabel("—")
        self.rbv_lbl.setFixedWidth(75)
        self.rbv_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.rbv_lbl.setObjectName("rbvLabel")

        # SP tag + edit
        self.sp_tag = QLabel("SP")
        self.sp_tag.setFixedWidth(22)
        self.sp_tag.setObjectName("fieldTag")

        self.sp_edit = QLineEdit()
        self.sp_edit.setFixedWidth(75)
        self.sp_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.sp_edit.returnPressed.connect(self._send_sp)

        # MOVN indicator
        self.movn_lbl = QLabel("⬤")
        self.movn_lbl.setFixedWidth(18)
        self.movn_lbl.setObjectName("movnIdle")

        # Step tag + edit
        self.step_tag = QLabel("Step")
        self.step_tag.setFixedWidth(36)
        self.step_tag.setObjectName("fieldTag")

        self.step_edit = QLineEdit("0.100")
        self.step_edit.setFixedWidth(58)
        self.step_edit.returnPressed.connect(self._send_twv)

        # Tweak buttons
        self.rev_btn = QPushButton("◀")
        self.rev_btn.setFixedSize(26, 24)
        self.rev_btn.setObjectName("tweakBtn")
        self.rev_btn.clicked.connect(self._tweak_rev)

        self.fwd_btn = QPushButton("▶")
        self.fwd_btn.setFixedSize(26, 24)
        self.fwd_btn.setObjectName("tweakBtn")
        self.fwd_btn.clicked.connect(self._tweak_fwd)

    def place_in_grid(self, grid, row: int):
        """Place all sub-widgets into an external QGridLayout at the given row."""
        A = Qt.AlignmentFlag
        grid.addWidget(self.axis_lbl,  row, 0, A.AlignCenter)
        grid.addWidget(self.desc_lbl,  row, 1)
        grid.addWidget(self.rbv_tag,   row, 2, A.AlignRight | A.AlignVCenter)
        grid.addWidget(self.rbv_lbl,   row, 3)
        grid.addWidget(self.sp_tag,    row, 4, A.AlignRight | A.AlignVCenter)
        grid.addWidget(self.sp_edit,   row, 5)
        grid.addWidget(self.movn_lbl,  row, 6, A.AlignCenter)
        grid.addWidget(self.step_tag,  row, 7, A.AlignRight | A.AlignVCenter)
        grid.addWidget(self.step_edit, row, 8)
        grid.addWidget(self.rev_btn,   row, 9)
        grid.addWidget(self.fwd_btn,   row, 10)

    # ── public API ─────────────────────────────────────────────────────────

    def connect(self, base_pv: str):
        self._disconnect()
        self._base = base_pv
        if not (EPICS_AVAILABLE and base_pv):
            self._set_conn("idle")
            return
        self._set_conn("connecting")
        try:
            monitored   = ("DESC", "RBV", "VAL", "MOVN", "TWV")
            unmonitored = ("TWR", "TWF")
            for f in monitored:
                kw = dict(callback=self._bridge, auto_monitor=True)
                if f == "RBV":
                    kw["connection_callback"] = self._bridge.conn_cb
                self._pvs[f] = PV(f"{base_pv}.{f}", **kw)
            for f in unmonitored:
                self._pvs[f] = PV(f"{base_pv}.{f}")
            self._motor = Motor(base_pv)
        except Exception as e:
            print(f"Motor({base_pv}) connection failed: {e}")
            self._set_conn("lost")

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

    # ── private ────────────────────────────────────────────────────────────

    def _disconnect(self):
        for pv in self._pvs.values():
            try:
                pv.disconnect()
            except Exception:
                pass
        self._pvs.clear()
        self._motor = None
        self._set_conn("idle")

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

    _CONN_STYLE = {
        "idle":       "background:#9ca3af; color:white;",
        "connecting": "background:#d97706; color:white;",
        "ok":         "background:#16a34a; color:white;",
        "lost":       "background:#dc2626; color:white;",
    }
    _CONN_TIP = {
        "idle":       "No PV configured",
        "connecting": "Connecting…",
        "ok":         "Connected",
        "lost":       "Connection lost",
    }

    def _set_conn(self, state: str):
        style = self._CONN_STYLE.get(state, self._CONN_STYLE["idle"])
        base_style = (
            f"{style} border-radius:4px; font-weight:bold;"
            " font-size:9pt; padding:1px 2px;"
        )
        self.axis_lbl.setStyleSheet(base_style)
        self.axis_lbl.setToolTip(
            f"{self._label} — {self._CONN_TIP.get(state, '')}"
            + (f"\nPV: {self._base}" if self._base else "")
        )

    @pyqtSlot(str, bool)
    def _on_conn(self, pvname: str, conn: bool):
        if pvname == f"{self._base}.RBV":
            self._set_conn("ok" if conn else "lost")

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


# ── Off-thread camera image processor ─────────────────────────────────────────

class _ImageProcessor(QObject):
    """Processes raw camera frames off the Qt main thread."""
    processed = pyqtSignal(object, object, float, int, int)
    # emits: rgb_array, gray_array, focus_score, width, height

    @pyqtSlot(object, int, int)
    def process(self, raw_value, width, height):
        try:
            raw = np.asarray(raw_value, dtype=np.uint8)
            if width <= 0 or height <= 0 or raw.size == 0:
                return
            n_px = width * height
            if raw.size == n_px:
                # ── Mono8 ──────────────────────────────────────────────────
                gray = raw.reshape((height, width))
                if CV2_AVAILABLE:
                    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                    fp  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                else:
                    rgb = np.stack([gray, gray, gray], axis=-1)
                    fp  = 0.0
            elif raw.size == n_px * 3:
                # ── RGB8 / BGR8 ────────────────────────────────────────────
                bgr = raw.reshape((height, width, 3))
                if CV2_AVAILABLE:
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    fp   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                else:
                    gray = bgr.mean(axis=2).astype(np.uint8)
                    rgb  = bgr[:, :, ::-1].copy()
                    fp   = 0.0
            else:
                print(f"[_ImageProcessor] unexpected size {raw.size} for {width}×{height}")
                return
            self.processed.emit(rgb, gray, fp, width, height)
        except Exception as e:
            print(f"[_ImageProcessor] {e}")


# ── Off-thread Blender SSH worker ──────────────────────────────────────────────

class _BlenderWorker(QObject):
    """Runs Blender SSH job off the main thread."""
    finished = pyqtSignal(object)   # list[dict] | None
    error    = pyqtSignal(str)

    def __init__(self, cfg: dict, positions: list, spacing: float, data_dir: str):
        super().__init__()
        self._cfg      = cfg
        self._positions = positions
        self._spacing  = spacing
        self._data_dir = data_dir

    @pyqtSlot()
    def run(self):
        try:
            result = self._do_ssh()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def _do_ssh(self):
        import paramiko as _pm
        cfg          = self._cfg
        local_mount  = cfg.get("LOCAL_MOUNT", "")
        remote_mount = cfg.get("REMOTE_MOUNT", "")

        temp_pos = os.path.join(self._data_dir, "temp_blender_in.pos")
        with open(temp_pos, 'w') as f:
            f.write("# x y z\n")
            for p in normalize_positions(self._positions):
                f.write(f"{float(p['x']):.6f} {float(p['y']):.6f} {float(p['z']):.6f}\n")

        temp_out = os.path.join(self._data_dir, "temp_blender_out.csv")
        lifname  = os.path.abspath(temp_pos)
        lofname  = os.path.abspath(temp_out)
        ifname   = lifname.replace(local_mount, remote_mount).replace('\\', '/')
        ofname   = lofname.replace(local_mount, remote_mount).replace('\\', '/')

        blender_exe    = cfg.get("BLENDER_EXE", "blender") or "blender"
        blender_script = cfg.get("BLENDER_SCRIPT", "")
        cmd = " ".join([blender_exe, '--background', '--python', blender_script,
                        '--', ifname, ofname, f"{self._spacing:.2f}"])

        client = _pm.SSHClient()
        client.set_missing_host_key_policy(_pm.AutoAddPolicy())
        try:
            client.connect(cfg.get("BLENDER_HOST",""), port=22,
                           username=cfg.get("BLENDER_USER",""),
                           key_filename=cfg.get("BLENDER_KEY",""))
            _stdin, stdout, stderr = client.exec_command(cmd)
            out = stdout.read().decode(); err = stderr.read().decode()
            if out: print(out)
            if err: print("stderr:", err)
            if not os.path.exists(lofname):
                try:
                    sftp = client.open_sftp()
                    sftp.get(ofname, lofname)
                    sftp.close()
                except Exception:
                    pass
        finally:
            client.close()

        if not os.path.exists(lofname):
            return None

        parsed_rows = []
        with open(lofname, newline='', encoding='utf-8', errors='replace') as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                parts = [x.strip() for x in stripped.split(',')]
                if len(parts) < 3:
                    continue
                try:
                    parsed_rows.append({
                        "x": float(parts[0]),
                        "y": float(parts[1]),
                        "z": float(parts[2]),
                    })
                except ValueError:
                    continue
        return parsed_rows if parsed_rows else None


# ── Calibration dialog ─────────────────────────────────────────────────────────

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


# ── Position map widget ────────────────────────────────────────────────────────

class PositionMapWidget(QWidget):
    pointSelected      = pyqtSignal(int)
    pointAddRequested  = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._positions        = []
        self._selected_row     = -1
        self._show_arrows      = False
        self._show_names       = True
        self._add_points_enabled = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.setLabel("bottom", "x (mm)")
        self.plot.setLabel("left",   "y (mm)")
        self.plot.showGrid(x=True, y=True, alpha=0.22)
        self.plot.enableAutoRange(False)
        self.plot.setRange(xRange=[-10, 10], yRange=[-10, 10])
        lay.addWidget(self.plot)

        self.plot.scene().sigMouseClicked.connect(self._plot_clicked)

        self._label_items     = []
        self._arrow_items     = []
        self._scatter_items   = []
        self._selection_item  = None

    def set_positions(self, positions):
        self._positions = list(positions)
        self._redraw()

    def set_selected_row(self, row: int):
        self._selected_row = row
        self._draw_selection()

    def set_sequence_arrows_visible(self, v: bool):
        self._show_arrows = v
        self._redraw()

    def set_names_visible(self, v: bool):
        self._show_names = v
        self._redraw()

    def set_add_points_enabled(self, v: bool):
        self._add_points_enabled = v
        if v:
            self.plot.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.plot.unsetCursor()

    def _redraw(self):
        for item in self._scatter_items + self._label_items + self._arrow_items:
            try:
                self.plot.removeItem(item)
            except Exception:
                pass
        if self._selection_item:
            try:
                self.plot.removeItem(self._selection_item)
            except Exception:
                pass
        self._scatter_items  = []
        self._label_items    = []
        self._arrow_items    = []
        self._selection_item = None

        if not self._positions:
            return

        # Group by role
        role_groups: dict[str, dict] = {}
        for i, pos in enumerate(self._positions):
            role = str(pos.get("role", "Sample") or "Sample")
            if role not in role_groups:
                role_groups[role] = {"xs": [], "ys": [], "indices": []}
            role_groups[role]["xs"].append(float(pos.get("x", 0)))
            role_groups[role]["ys"].append(float(pos.get("y", 0)))
            role_groups[role]["indices"].append(i)

        for role, data in role_groups.items():
            hex_color = ROLE_COLORS.get(role, "#888888")
            qc = QColor(hex_color)
            scatter = pg.ScatterPlotItem(
                x=data["xs"], y=data["ys"],
                size=12,
                brush=pg.mkBrush(qc),
                pen=pg.mkPen(qc.darker(140), width=1),
                data=data["indices"],
            )
            scatter.sigClicked.connect(self._scatter_clicked)
            self.plot.addItem(scatter)
            self._scatter_items.append(scatter)

        if self._show_names:
            for pos in self._positions:
                name = str(pos.get("name", ""))
                if name:
                    text = pg.TextItem(name, color="#333333", anchor=(0, 1))
                    text.setPos(float(pos.get("x", 0)), float(pos.get("y", 0)))
                    self.plot.addItem(text)
                    self._label_items.append(text)

        if self._show_arrows and len(self._positions) > 1:
            self._draw_sequence_tube()

        self._draw_selection()
        self.plot.autoRange(padding=0.15)

    def _scatter_clicked(self, _scatter, points):
        if points:
            idx = points[0].data()
            if idx is not None:
                self.pointSelected.emit(int(idx))

    def _plot_clicked(self, event):
        if not self._add_points_enabled:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos    = event.scenePos()
        mapped = self.plot.plotItem.vb.mapSceneToView(pos)
        self.pointAddRequested.emit(mapped.x(), mapped.y())

    def _draw_selection(self):
        if self._selection_item:
            try:
                self.plot.removeItem(self._selection_item)
            except Exception:
                pass
            self._selection_item = None
        if self._selected_row < 0 or self._selected_row >= len(self._positions):
            return
        pos = self._positions[self._selected_row]
        x   = float(pos.get("x", 0))
        y   = float(pos.get("y", 0))
        self._selection_item = pg.ScatterPlotItem(
            x=[x], y=[y],
            size=22,
            brush=pg.mkBrush(None),
            pen=pg.mkPen("#1e3a5f", width=2),
            symbol='o',
        )
        self.plot.addItem(self._selection_item)

    def _draw_sequence_tube(self):
        xs = [float(p.get("x", 0)) for p in self._positions]
        ys = [float(p.get("y", 0)) for p in self._positions]
        outer = pg.PlotCurveItem(xs, ys, pen=pg.mkPen("#93c5fd", width=6))
        inner = pg.PlotCurveItem(xs, ys, pen=pg.mkPen("#3b82f6", width=2))
        self.plot.addItem(outer)
        self.plot.addItem(inner)
        self._arrow_items.extend([outer, inner])
        for i in range(1, len(xs)):
            mx = (xs[i - 1] + xs[i]) / 2
            my = (ys[i - 1] + ys[i]) / 2
            spot = pg.ScatterPlotItem(
                [mx], [my], size=8,
                brush=pg.mkBrush("#2563eb"),
                pen=pg.mkPen(None),
            )
            self.plot.addItem(spot)
            self._arrow_items.append(spot)


# ── Rack canvas ────────────────────────────────────────────────────────────────

_RACK_ROLES  = ["Sample", "Solvent", "GC", "Air", "Empty", "Skip"]
_RACK_COLORS = {
    "Sample":  QColor("#5b8ff9"),
    "Solvent": QColor("#5ad8a6"),
    "GC":      QColor("#f6bd16"),
    "Air":     QColor("#e86452"),
    "Empty":   QColor("#6dc8ec"),
    "Skip":    QColor("#d8dce3"),
}


class RackCanvas(QWidget):
    selectedChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(230)
        self.setMouseTracking(True)
        self.positions     = []
        self.selected_index = 0

    def set_positions(self, positions):
        self.positions      = positions
        self.selected_index = min(self.selected_index, max(0, len(positions) - 1))
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.palette().base())
        if not self.positions:
            return
        margin_x    = 32
        rack_top    = 92
        rack_height = 96
        painter.setPen(QPen(QColor("#111111"), 4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(margin_x, rack_top, self.width() - 2 * margin_x, rack_height)
        slot_w = 18
        slot_h = 116
        y      = rack_top - 42
        for i, pos in enumerate(self.positions):
            cx   = self._cx(i)
            rect = QRectF(cx - slot_w / 2, y, slot_w, slot_h)
            role  = str(pos.get("role", "Skip") or "Skip")
            color = _RACK_COLORS.get(role, _RACK_COLORS["Skip"])
            painter.setBrush(QBrush(color.lighter(145)))
            pen = QPen(QColor("#111111"), 3)
            if i == self.selected_index:
                pen = QPen(QColor("#000000"), 5)
            painter.setPen(pen)
            painter.drawRoundedRect(rect, 6, 6)
            painter.setPen(QColor("#222222"))
            painter.setFont(QFont("Arial", 8))
            painter.drawText(
                QRectF(cx - 26, y + slot_h + 5, 52, 18),
                Qt.AlignmentFlag.AlignCenter,
                str(i + 1),
            )
            label = str(pos.get("name", "")).strip() or role
            painter.drawText(
                QRectF(cx - 42, y - 26, 84, 22),
                Qt.AlignmentFlag.AlignCenter,
                label[:12],
            )

    def mousePressEvent(self, event):
        if not self.positions:
            return
        dist, idx = min(
            (abs(event.position().x() - self._cx(i)), i)
            for i in range(len(self.positions))
        )
        if dist <= 28:
            self.selected_index = idx
            self.selectedChanged.emit(idx)
            self.update()

    def _cx(self, index):
        n     = max(1, len(self.positions))
        left  = 62
        right = max(63, self.width() - 62)
        return (left + right) / 2 if n == 1 else left + (right - left) * index / (n - 1)


# ── Rack builder dialog ────────────────────────────────────────────────────────

class RackBuilderDialog(QDialog):
    def __init__(self, positions=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rack Builder")
        self.resize(960, 620)
        self._positions  = normalize_positions(list(positions)) if positions else []
        self._updating   = False
        self._build_ui()
        if not self._positions:
            self._seed_positions(13)
        else:
            self._refresh_table()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(6)

        # Top controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Slots:"))
        self._count_spin = QSpinBox()
        self._count_spin.setRange(1, 200)
        self._count_spin.setValue(13)
        self._count_spin.valueChanged.connect(self.set_group_count)
        ctrl.addWidget(self._count_spin)

        ctrl.addSpacing(12)
        ctrl.addWidget(QLabel("Start X (mm):"))
        self._start_x = QDoubleSpinBox()
        self._start_x.setRange(-999, 999)
        self._start_x.setDecimals(3)
        self._start_x.setValue(0.0)
        ctrl.addWidget(self._start_x)

        ctrl.addWidget(QLabel("Spacing (mm):"))
        self._spacing = QDoubleSpinBox()
        self._spacing.setRange(-999, 999)
        self._spacing.setDecimals(3)
        self._spacing.setValue(3.0)
        ctrl.addWidget(self._spacing)

        ctrl.addWidget(QLabel("Y (mm):"))
        self._fixed_y = QDoubleSpinBox()
        self._fixed_y.setRange(-999, 999)
        self._fixed_y.setDecimals(3)
        self._fixed_y.setValue(0.0)
        ctrl.addWidget(self._fixed_y)

        ctrl.addWidget(QLabel("Z (mm):"))
        self._fixed_z = QDoubleSpinBox()
        self._fixed_z.setRange(-999, 999)
        self._fixed_z.setDecimals(3)
        self._fixed_z.setValue(0.0)
        ctrl.addWidget(self._fixed_z)

        apply_coord_btn = QPushButton("Apply Coordinates")
        apply_coord_btn.setObjectName("actionBtn")
        apply_coord_btn.clicked.connect(self._apply_coordinates)
        ctrl.addWidget(apply_coord_btn)
        ctrl.addStretch()
        main.addLayout(ctrl)

        # Role quick-assign buttons
        role_row = QHBoxLayout()
        role_row.addWidget(QLabel("Set selected →"))
        for role in _RACK_ROLES:
            btn = QPushButton(role)
            btn.setFixedHeight(24)
            qc = _RACK_COLORS.get(role, QColor("#aaa"))
            btn.setStyleSheet(
                f"QPushButton{{background:{qc.name()};border:1px solid #888;"
                f"border-radius:3px;padding:1px 6px;}}"
                f"QPushButton:hover{{background:{qc.lighter(115).name()};}}"
            )
            btn.clicked.connect(lambda checked, r=role: self._set_selected_role(r))
            role_row.addWidget(btn)
        role_row.addStretch()
        main.addLayout(role_row)

        # Splitter: table | editor | canvas
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Table
        self._table = QTableWidget(0, len(POSITION_FIELDS))
        self._table.setHorizontalHeaderLabels(POSITION_FIELDS)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemChanged.connect(self._table_item_changed)
        self._table.selectionModel().selectionChanged.connect(self._table_selection_changed)
        splitter.addWidget(self._table)

        # Editor panel
        editor_w = QWidget()
        editor_w.setFixedWidth(220)
        ef = QFormLayout(editor_w)
        ef.setSpacing(6)
        self._ed_name  = QLineEdit()
        self._ed_x     = QDoubleSpinBox(); self._ed_x.setRange(-9999, 9999); self._ed_x.setDecimals(4)
        self._ed_y     = QDoubleSpinBox(); self._ed_y.setRange(-9999, 9999); self._ed_y.setDecimals(4)
        self._ed_z     = QDoubleSpinBox(); self._ed_z.setRange(-9999, 9999); self._ed_z.setDecimals(4)
        self._ed_role  = QComboBox(); self._ed_role.addItems(ROLE_PRESETS)
        self._ed_group = QLineEdit()
        self._ed_solvent_group = QLineEdit()
        self._ed_note  = QLineEdit()
        ef.addRow("Name:",          self._ed_name)
        ef.addRow("X (mm):",        self._ed_x)
        ef.addRow("Y (mm):",        self._ed_y)
        ef.addRow("Z (mm):",        self._ed_z)
        ef.addRow("Role:",          self._ed_role)
        ef.addRow("Group:",         self._ed_group)
        ef.addRow("Solvent group:", self._ed_solvent_group)
        ef.addRow("Note:",          self._ed_note)
        apply_btn = QPushButton("Apply to selected")
        apply_btn.clicked.connect(self._apply_editor)
        ef.addRow(apply_btn)
        splitter.addWidget(editor_w)

        # Rack canvas
        self._canvas = RackCanvas()
        self._canvas.selectedChanged.connect(self._select_row)
        splitter.addWidget(self._canvas)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 2)
        main.addWidget(splitter, 1)

        # Dialog buttons
        bbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        main.addWidget(bbox)

    # ── Data management ────────────────────────────────────────────────────

    def _seed_positions(self, n: int):
        self._positions = [self._blank_position(i) for i in range(n)]
        self._count_spin.setValue(n)
        self._refresh_table()

    def _blank_position(self, i: int) -> dict:
        return PositionRecord(
            name=f"rack_{i+1}", x=0.0, y=0.0, z=0.0,
            role="Sample", layout="rack",
        ).to_dict()

    def set_group_count(self, n: int):
        current = len(self._positions)
        if n > current:
            for i in range(current, n):
                self._positions.append(self._blank_position(i))
        elif n < current:
            self._positions = self._positions[:n]
        self._refresh_table()

    def _apply_coordinates(self):
        start   = self._start_x.value()
        spacing = self._spacing.value()
        y_val   = self._fixed_y.value()
        z_val   = self._fixed_z.value()
        saved_rows = self._selected_rows()
        for i, pos in enumerate(self._positions):
            pos["x"] = round(start + i * spacing, 4)
            pos["y"] = round(y_val, 4)
            pos["z"] = round(z_val, 4)
        self._refresh_table()
        self._restore_selection(saved_rows, saved_rows[0] if saved_rows else 0)

    def _refresh_table(self):
        self._updating = True
        self._table.setRowCount(len(self._positions))
        for r, pos in enumerate(self._positions):
            for c, field in enumerate(POSITION_FIELDS):
                val = pos.get(field, "")
                if field == "role":
                    combo = QComboBox()
                    combo.addItems(ROLE_PRESETS)
                    if str(val) in ROLE_PRESETS:
                        combo.setCurrentText(str(val))
                    combo.currentTextChanged.connect(
                        lambda v, row=r: self._set_rows_role([row], v)
                    )
                    self._table.setCellWidget(r, c, combo)
                else:
                    item = QTableWidgetItem(str(val))
                    self._table.setItem(r, c, item)
        self._canvas.set_positions(self._positions)
        self._updating = False

    def _table_item_changed(self, item):
        if self._updating:
            return
        r     = item.row()
        c     = item.column()
        field = POSITION_FIELDS[c]
        if r >= len(self._positions):
            return
        val = item.text()
        if field in NUMERIC_FIELDS:
            val = _flt(val)
        self._positions[r][field] = val
        self._canvas.set_positions(self._positions)

    def _table_selection_changed(self, _sel, _desel):
        rows = self._selected_rows()
        if rows:
            self._load_editor_for_row(rows[0])
            self._canvas.selected_index = rows[0]
            self._canvas.update()

    def _selected_rows(self) -> list:
        return sorted(set(idx.row() for idx in self._table.selectionModel().selectedRows()))

    def _select_row(self, row: int):
        self._table.selectRow(row)
        self._load_editor_for_row(row)

    def _load_editor_for_row(self, row: int):
        if row < 0 or row >= len(self._positions):
            return
        pos = self._positions[row]
        self._set_editor_values(
            name=str(pos.get("name", "")),
            x=float(pos.get("x", 0)), y=float(pos.get("y", 0)), z=float(pos.get("z", 0)),
            role=str(pos.get("role", "Sample")),
            group=str(pos.get("group", "")),
            solvent_group=str(pos.get("solvent_group", "")),
            note=str(pos.get("note", "")),
        )

    def _set_editor_values(self, *, name="", x=0.0, y=0.0, z=0.0,
                           role="Sample", group="", solvent_group="", note=""):
        self._ed_name.setText(name)
        self._ed_x.setValue(x)
        self._ed_y.setValue(y)
        self._ed_z.setValue(z)
        if role in ROLE_PRESETS:
            self._ed_role.setCurrentText(role)
        self._ed_group.setText(group)
        self._ed_solvent_group.setText(solvent_group)
        self._ed_note.setText(note)

    def _apply_editor(self):
        rows = self._selected_rows()
        if not rows:
            return
        saved = rows[:]
        for r in rows:
            if r >= len(self._positions):
                continue
            pos = self._positions[r]
            pos["name"]          = self._ed_name.text().strip()
            pos["x"]             = self._ed_x.value()
            pos["y"]             = self._ed_y.value()
            pos["z"]             = self._ed_z.value()
            pos["role"]          = self._ed_role.currentText()
            pos["group"]         = self._ed_group.text().strip()
            pos["solvent_group"] = self._ed_solvent_group.text().strip()
            pos["note"]          = self._ed_note.text().strip()
        self._refresh_table()
        self._restore_selection(saved, saved[0])

    def _set_selected_role(self, role: str):
        rows = self._selected_rows()
        self._set_rows_role(rows, role)

    def _set_rows_role(self, rows: list, role: str):
        saved = self._selected_rows()
        for r in rows:
            if r < len(self._positions):
                self._positions[r]["role"] = role
        self._refresh_table()
        self._restore_selection(saved, saved[0] if saved else 0)

    def _restore_selection(self, rows: list, primary: int):
        sm = self._table.selectionModel()
        sm.clearSelection()
        for r in rows:
            idx = self._table.model().index(r, 0)
            sm.select(
                idx,
                QItemSelectionModel.SelectionFlag.Select |
                QItemSelectionModel.SelectionFlag.Rows,
            )
        if 0 <= primary < self._table.rowCount():
            self._table.scrollToItem(self._table.item(primary, 0))

    def _sync_from_table(self):
        self._updating = True
        for r in range(self._table.rowCount()):
            if r >= len(self._positions):
                break
            pos = self._positions[r]
            for c, field in enumerate(POSITION_FIELDS):
                if field == "role":
                    w = self._table.cellWidget(r, c)
                    if w:
                        pos[field] = w.currentText()
                else:
                    item = self._table.item(r, c)
                    if item:
                        val = item.text()
                        pos[field] = _flt(val) if field in NUMERIC_FIELDS else val
        self._updating = False

    def _nearest_solvent(self, row: int) -> int:
        solvents = [i for i, p in enumerate(self._positions)
                    if str(p.get("role", "")).strip() == "Solvent"]
        if not solvents:
            return -1
        px = float(self._positions[row].get("x", 0))
        return min(solvents, key=lambda i: abs(float(self._positions[i].get("x", 0)) - px))

    @property
    def result_positions(self) -> list:
        self._sync_from_table()
        return normalize_positions(self._positions)


# ── Sample position tab ────────────────────────────────────────────────────────

class SamplePositionTab(QWidget):
    positionsChanged = pyqtSignal(list)

    def __init__(self, station=None, parent=None):
        super().__init__(parent)
        self._station          = station
        self._positions        = []
        self._updating         = False
        self._current_path     = None
        self._pending_role_rows: list[int] = []
        self._build_ui()
        self.set_positions([])  # start empty; use Templates menu or Capture to add positions

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(4, 4, 4, 4)
        main.setSpacing(4)

        # ── Top toolbar (file + capture + template + view) ──────────────
        top = QHBoxLayout()
        top.setSpacing(4)

        new_btn = QPushButton("New")
        new_btn.clicked.connect(self._new)
        top.addWidget(new_btn)

        open_btn = QPushButton("Open…")
        open_btn.clicked.connect(self._open)
        top.addWidget(open_btn)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save)
        top.addWidget(save_btn)

        saveas_btn = QPushButton("Save As…")
        saveas_btn.clicked.connect(self._save_as)
        top.addWidget(saveas_btn)

        exp_bs_btn = QPushButton("Export Bluesky CSV")
        exp_bs_btn.clicked.connect(self._export_bluesky)
        top.addWidget(exp_bs_btn)

        exp_rd_btn = QPushButton("Export Reducer Pairs")
        exp_rd_btn.clicked.connect(self._export_reducer)
        top.addWidget(exp_rd_btn)

        sep0 = QFrame(); sep0.setFrameShape(QFrame.Shape.VLine)
        sep0.setObjectName("toolSep"); top.addWidget(sep0)

        self.capture_btn = QPushButton("📍 Capture from Stage")
        self.capture_btn.setObjectName("captureBtn")
        self.capture_btn.setToolTip("Read current motor positions and add entry")
        self.capture_btn.clicked.connect(self._capture_from_stage)
        top.addWidget(self.capture_btn)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.Shape.VLine)
        sep1.setObjectName("toolSep"); top.addWidget(sep1)

        top.addWidget(QLabel("Template:"))
        self.template_combo = QComboBox()
        self.template_combo.addItems(["Freeform", "Capillary Linear", "Rack Builder", "Chip"])
        top.addWidget(self.template_combo)

        apply_tmpl_btn = QPushButton("Apply")
        apply_tmpl_btn.clicked.connect(self._apply_template)
        top.addWidget(apply_tmpl_btn)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setObjectName("toolSep"); top.addWidget(sep2)

        self.show_arrows_cb = QCheckBox("Arrows")
        self.show_arrows_cb.setChecked(False)
        self.show_arrows_cb.toggled.connect(
            lambda v: self.map_widget.set_sequence_arrows_visible(v))
        top.addWidget(self.show_arrows_cb)

        self.show_names_cb = QCheckBox("Names")
        self.show_names_cb.setChecked(True)
        self.show_names_cb.toggled.connect(
            lambda v: self.map_widget.set_names_visible(v))
        top.addWidget(self.show_names_cb)

        self.click_add_cb = QCheckBox("Click-add")
        self.click_add_cb.setChecked(False)
        self.click_add_cb.toggled.connect(
            lambda v: self.map_widget.set_add_points_enabled(v))
        top.addWidget(self.click_add_cb)

        top.addStretch()
        main.addLayout(top)

        # ── Row management bar ──────────────────────────────────────────
        row_bar = QHBoxLayout()
        row_bar.setSpacing(4)

        add_btn = QPushButton("＋ Add")
        add_btn.setObjectName("greenBtn")
        add_btn.clicked.connect(self._add_row)
        row_bar.addWidget(add_btn)

        del_btn = QPushButton("－ Delete")
        del_btn.setObjectName("redBtn")
        del_btn.clicked.connect(self._delete_selected)
        row_bar.addWidget(del_btn)

        dup_btn = QPushButton("Duplicate")
        dup_btn.clicked.connect(self._duplicate_selected)
        row_bar.addWidget(dup_btn)

        row_bar.addSpacing(8)
        row_bar.addWidget(QLabel("Selected role:"))
        self.bulk_role_combo = QComboBox()
        self.bulk_role_combo.addItems(ROLE_PRESETS)
        row_bar.addWidget(self.bulk_role_combo)

        assign_btn = QPushButton("Assign")
        assign_btn.clicked.connect(self._assign_role)
        row_bar.addWidget(assign_btn)

        sep3 = QFrame(); sep3.setFrameShape(QFrame.Shape.VLine)
        sep3.setObjectName("toolSep"); row_bar.addWidget(sep3)

        row_bar.addWidget(QLabel("Blender spacing (mm):"))
        self.interp_edit = QLineEdit("1.0")
        self.interp_edit.setFixedWidth(55)
        row_bar.addWidget(self.interp_edit)

        self.blender_btn = QPushButton("⚙ Run Blender")
        self.blender_btn.setObjectName("actionBtn")
        self.blender_btn.clicked.connect(self._run_blender)
        row_bar.addWidget(self.blender_btn)

        row_bar.addStretch()
        main.addLayout(row_bar)

        # ── Splitter: table | map ───────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.table = QTableWidget(0, len(POSITION_FIELDS))
        self.table.setHorizontalHeaderLabels(POSITION_FIELDS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.installEventFilter(self)
        self.table.setDragEnabled(True)
        self.table.setAcceptDrops(True)
        self.table.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.table.setDropIndicatorShown(True)
        self.table.setDragDropOverwriteMode(False)
        self.table.model().rowsMoved.connect(self._on_rows_moved)
        self.table.itemChanged.connect(self._item_changed)
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)
        splitter.addWidget(self.table)

        self.map_widget = PositionMapWidget()
        self.map_widget.pointSelected.connect(self._select_row)
        self.map_widget.pointAddRequested.connect(self._add_from_map)
        splitter.addWidget(self.map_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        main.addWidget(splitter, 1)

    # ── Public API ─────────────────────────────────────────────────────────

    def positions(self) -> list:
        return normalize_positions(self._positions)

    def set_positions(self, positions):
        self._positions = normalize_positions(positions)
        self._refresh_table()
        self.map_widget.set_positions(self._positions)
        if self._positions:
            self.table.selectRow(0)
        self.positionsChanged.emit(self.positions())

    def load_positions(self, path):
        self.set_positions(load_positions(path))
        self._current_path = Path(path)

    def save_positions(self, path):
        save_positions(path, self.positions())
        self._current_path = Path(path)

    # ── Capture from stage ─────────────────────────────────────────────────

    def _capture_from_stage(self):
        if self._station is None:
            QMessageBox.warning(self, "No Station", "Not connected to station.")
            return
        try:
            x = float(self._station.x_motor.rbv_lbl.text())
            y = float(self._station.y_motor.rbv_lbl.text())
            z = float(self._station.z_motor.rbv_lbl.text())
        except (ValueError, AttributeError):
            QMessageBox.warning(self, "No Readback", "Motor RBV not available.")
            return
        idx = len(self._positions)
        self._positions.append(PositionRecord(
            name=f"pos_{idx+1}", x=x, y=y, z=z,
            role="Sample", layout="freeform",
        ).to_dict())
        self.set_positions(self._positions)
        self._select_row(idx)

    # ── Blender ────────────────────────────────────────────────────────────

    def _run_blender(self):
        if self._station is None:
            QMessageBox.warning(self, "Blender", "No station connected.")
            return
        if not PARAMIKO_AVAILABLE:
            QMessageBox.critical(self, "SSH Error",
                                 "paramiko is not installed.\nRun: pip install paramiko")
            return
        try:
            spacing = float(self.interp_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Blender", "Enter a valid number for spacing.")
            return

        data_dir = os.path.join(_DIR, "Data")
        os.makedirs(data_dir, exist_ok=True)
        self.blender_btn.setEnabled(False)
        self.blender_btn.setText("Running…")

        worker = _BlenderWorker(self._station.cfg, self._positions, spacing, data_dir)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(lambda result: self._station._on_blender_done(result, thread))
        worker.error.connect(lambda msg: self._station._on_blender_error(msg, thread))
        thread.start()

    def _apply_blender_result(self, result):
        reply = QMessageBox.question(
            self, "Blender Result",
            f"Blender returned {len(result)} interpolated points.\n"
            "Yes = Append to existing  |  No = Replace",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No |
            QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return
        interp = [
            PositionRecord(
                name=f"interp_{i+1}",
                x=float(r.get("x", 0)), y=float(r.get("y", 0)), z=float(r.get("z", 0)),
                role="Interpolated", layout="blender_interpolated",
            ).to_dict()
            for i, r in enumerate(result)
        ]
        new = (self._positions + interp) if reply == QMessageBox.StandardButton.Yes else interp
        self.set_positions(new)

    # ── Table management ───────────────────────────────────────────────────

    def _refresh_table(self):
        self._updating = True
        self.table.setRowCount(len(self._positions))
        for r, pos in enumerate(self._positions):
            for c, field in enumerate(POSITION_FIELDS):
                val = pos.get(field, "")
                if field == "role":
                    combo = QComboBox()
                    combo.addItems(ROLE_PRESETS)
                    if str(val) in ROLE_PRESETS:
                        combo.setCurrentText(str(val))
                    combo.currentTextChanged.connect(
                        lambda v, row=r: self._role_changed(row, v)
                    )
                    self.table.setCellWidget(r, c, combo)
                else:
                    item = QTableWidgetItem(str(val))
                    self.table.setItem(r, c, item)
        self._updating = False

    def eventFilter(self, watched, event):
        if watched is self.table:
            if event.type() == QEvent.Type.KeyPress:
                if event.key() == Qt.Key.Key_Delete:
                    self._delete_selected()
                    return True
            if event.type() in (QEvent.Type.FocusIn, QEvent.Type.MouseButtonPress):
                self._pending_role_rows = self._selected_rows()
        return super().eventFilter(watched, event)

    def _item_changed(self, item):
        if self._updating:
            return
        row   = item.row()
        col   = item.column()
        if row >= len(self._positions):
            return
        field = POSITION_FIELDS[col]
        val   = item.text()
        if field in NUMERIC_FIELDS:
            val = _flt(val)
        self._positions[row][field] = val
        self._commit(selected_row=row)

    def _role_changed(self, row: int, value: str):
        if self._updating:
            return
        rows_to_update = self._pending_role_rows if self._pending_role_rows else [row]
        for r in rows_to_update:
            if r < len(self._positions):
                self._positions[r]["role"] = value
        self._pending_role_rows = []
        self._commit()

    def _commit(self, selected_row: int | None = None):
        self._positions = normalize_positions(self._positions)
        self.map_widget.set_positions(self._positions)
        if selected_row is not None:
            self.map_widget.set_selected_row(selected_row)
        self.positionsChanged.emit(self.positions())

    def _selection_changed(self):
        rows = self._selected_rows()
        if rows:
            self.map_widget.set_selected_row(rows[0])
        self._pending_role_rows = rows

    def _select_row(self, row: int):
        self.table.selectRow(row)
        self.map_widget.set_selected_row(row)

    def _selected_rows(self) -> list:
        return sorted(set(idx.row() for idx in self.table.selectionModel().selectedRows()))

    def _assign_role(self):
        role = self.bulk_role_combo.currentText()
        rows = self._selected_rows()
        for r in rows:
            self._positions[r]["role"] = role
        saved = rows[:]
        self._refresh_table()
        self._restore_selection(saved, saved[0] if saved else 0)
        self.positionsChanged.emit(self.positions())

    # ── Row operations ─────────────────────────────────────────────────────

    def _add_row(self):
        rows = self._selected_rows()
        idx  = rows[-1] + 1 if rows else len(self._positions)
        self._positions.insert(idx, blank_position(idx))
        # renumber names
        self._positions = normalize_positions(self._positions)
        self.set_positions(self._positions)
        self._select_row(idx)

    def _add_from_map(self, x: float, y: float):
        rows = self._selected_rows()
        z    = float(self._positions[rows[0]].get("z", 0)) if rows else 0.0
        idx  = len(self._positions)
        self._positions.append(PositionRecord(
            name=f"pos_{idx+1}", x=x, y=y, z=z,
            role="Sample", layout="freeform",
        ).to_dict())
        self.set_positions(self._positions)
        self._select_row(idx)

    def _delete_selected(self):
        rows = sorted(self._selected_rows(), reverse=True)
        for r in rows:
            if 0 <= r < len(self._positions):
                self._positions.pop(r)
        self.set_positions(self._positions)

    def _duplicate_selected(self):
        rows = self._selected_rows()
        if not rows:
            return
        r   = rows[-1]
        pos = dict(self._positions[r])
        pos["name"] = str(pos.get("name", "")) + "_copy"
        self._positions.insert(r + 1, pos)
        self.set_positions(self._positions)
        self._select_row(r + 1)

    def _move_selected(self, delta: int):
        rows = self._selected_rows()
        if not rows:
            return
        row = rows[0]
        new = row + delta
        if 0 <= new < len(self._positions):
            self._positions.insert(new, self._positions.pop(row))
            self.set_positions(self._positions)
            self._select_row(new)

    def _on_rows_moved(self, _src_parent, src_start: int, src_end: int,
                       _dst_parent, dst_row: int):
        """Sync self._positions after a drag-and-drop row reorder."""
        moved = [self._positions.pop(src_start) for _ in range(src_end - src_start + 1)]
        insert_at = dst_row if dst_row <= src_start else dst_row - (src_end - src_start + 1)
        for i, p in enumerate(moved):
            self._positions.insert(insert_at + i, p)
        self.map_widget.set_positions(self._positions)
        self.positionsChanged.emit(self.positions())

    # ── File operations ────────────────────────────────────────────────────

    def _new(self):
        if self._confirm_replace():
            self.set_positions([])

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Positions", "",
            "All supported (*.csv *.json *.pos);;CSV (*.csv);;JSON (*.json);;POS (*.pos)",
        )
        if not path:
            return
        if not self._confirm_replace():
            return
        try:
            self.load_positions(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _save(self):
        if self._current_path:
            try:
                self.save_positions(str(self._current_path))
            except Exception as e:
                QMessageBox.critical(self, "Save Error", str(e))
        else:
            self._save_as()

    def _save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Positions", "",
            "CSV (*.csv);;JSON (*.json);;POS (*.pos)",
        )
        if not path:
            return
        try:
            self.save_positions(path)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _export_bluesky(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Bluesky CSV", "", "CSV (*.csv)")
        if path:
            try:
                export_bluesky_csv(path, self.positions())
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def _export_reducer(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Reducer Pairs", "", "CSV (*.csv)")
        if path:
            try:
                export_reducer_pairs_csv(path, self.positions())
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    # ── Templates ──────────────────────────────────────────────────────────

    def _rack_builder_action(self):
        dlg = RackBuilderDialog(self._positions if self._positions else None, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            if self._confirm_replace():
                self.set_positions(dlg.result_positions())

    def _apply_template(self):
        tmpl = self.template_combo.currentText()
        if tmpl == "Freeform":
            if not self._confirm_replace():
                return
            self.set_positions([blank_position(i) for i in range(5)])
        elif tmpl == "Capillary Linear":
            result = self._capillary_dialog()
            if result is not None:
                if not self._confirm_replace():
                    return
                self.set_positions(result)
        elif tmpl == "Rack Builder":
            dlg = RackBuilderDialog(self._positions if self._positions else None, parent=self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                self.set_positions(dlg.result_positions)
        elif tmpl == "Chip":
            if not self._confirm_replace():
                return
            self.set_positions([blank_position(i, layout="chip") for i in range(10)])

    def _capillary_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Capillary Linear Template")
        form = QFormLayout(dlg)

        count_spin = QSpinBox()
        count_spin.setRange(1, 200)
        count_spin.setValue(13)
        form.addRow("Count:", count_spin)

        spacing_spin = QDoubleSpinBox()
        spacing_spin.setRange(-999, 999)
        spacing_spin.setDecimals(3)
        spacing_spin.setValue(1.0)
        form.addRow("Spacing (mm):", spacing_spin)

        start_x = QDoubleSpinBox(); start_x.setRange(-9999, 9999); start_x.setDecimals(3)
        start_y = QDoubleSpinBox(); start_y.setRange(-9999, 9999); start_y.setDecimals(3)
        start_z = QDoubleSpinBox(); start_z.setRange(-9999, 9999); start_z.setDecimals(3)
        form.addRow("Start X (mm):", start_x)
        form.addRow("Start Y (mm):", start_y)
        form.addRow("Start Z (mm):", start_z)

        axis_combo = QComboBox()
        axis_combo.addItems(["x", "y"])
        form.addRow("Axis:", axis_combo)

        bbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bbox.accepted.connect(dlg.accept)
        bbox.rejected.connect(dlg.reject)
        form.addRow(bbox)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None

        n       = count_spin.value()
        spacing = spacing_spin.value()
        sx      = start_x.value()
        sy      = start_y.value()
        sz      = start_z.value()
        axis    = axis_combo.currentText()
        result  = []
        for i in range(n):
            x = sx + (i * spacing if axis == "x" else 0.0)
            y = sy + (i * spacing if axis == "y" else 0.0)
            result.append(PositionRecord(
                name=f"cap_{i+1}", x=x, y=y, z=sz,
                role="Sample", layout="capillary_linear",
            ).to_dict())
        return result

    def _confirm_replace(self) -> bool:
        if not self._positions:
            return True
        reply = QMessageBox.question(
            self, "Replace positions?",
            "This will replace all current positions. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _restore_selection(self, rows: list, primary: int):
        sm = self.table.selectionModel()
        sm.clearSelection()
        for r in rows:
            idx = self.table.model().index(r, 0)
            sm.select(
                idx,
                QItemSelectionModel.SelectionFlag.Select |
                QItemSelectionModel.SelectionFlag.Rows,
            )
        if 0 <= primary < self.table.rowCount():
            self.table.scrollToItem(self.table.item(primary, 0))


# ── Setup dialog ──────────────────────────────────────────────────────────────

class SetupDialog(QDialog):
    """Standalone settings window opened from the Setup menu."""

    _SECTIONS = [
        ("motors", "Motor PVs", [
            ("X_MOTOR_PV",      "X Motor PV:"),
            ("Y_MOTOR_PV",      "Y Motor PV:"),
            ("Z_MOTOR_PV",      "Z Motor PV:"),
        ]),
        ("camera", "Camera", [
            ("CAMERA_PREFIX",    "Camera Prefix:"),
            ("IMAGE_PREFIX",     "Image Prefix:"),
            ("AUTOFOCUS_STEP",   "Autofocus Step (mm):"),
            ("AUTOFOCUS_SCRIPT", "Autofocus Script:"),
        ]),
        ("ssh", "SSH / Blender", [
            ("BLENDER_HOST",    "Host:"),
            ("BLENDER_USER",    "User:"),
            ("BLENDER_KEY",     "SSH Key Path:"),
            ("BLENDER_EXE",     "Blender Executable:"),
            ("BLENDER_SCRIPT",  "Blender Script (Macro):"),
            ("LOCAL_MOUNT",     "Local Mount:"),
            ("REMOTE_MOUNT",    "Remote Mount:"),
        ]),
    ]

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Setup — ASWAXS Sample Station")
        self.resize(520, 480)
        self.setModal(True)
        self._edits: dict[str, QLineEdit] = {}
        self._group_widgets: dict[str, QGroupBox] = {}
        self._build_ui(cfg)

    def _build_ui(self, cfg: dict):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(0, 0, 6, 0)
        v.setSpacing(10)

        for group_key, title, fields in self._SECTIONS:
            box = QGroupBox(title)
            fl  = QFormLayout(box)
            fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            fl.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            for cfg_key, label in fields:
                edit = QLineEdit(str(cfg.get(cfg_key, "")))
                fl.addRow(label, edit)
                self._edits[cfg_key] = edit
            v.addWidget(box)
            self._group_widgets[group_key] = box

        v.addStretch()
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        # Button box
        bbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel |
            QDialogButtonBox.StandardButton.RestoreDefaults,
        )
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        bbox.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self._restore_defaults)
        root.addWidget(bbox)

    def _restore_defaults(self):
        reply = QMessageBox.question(self, "Reset",
            "Reset all fields to factory defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            for key, edit in self._edits.items():
                edit.setText(str(DEFAULT_CONFIG.get(key, "")))

    def scroll_to(self, group_key: str):
        box = self._group_widgets.get(group_key)
        if box:
            # Defer so the dialog has finished laying out
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, lambda: box.setFocus())

    def values(self) -> dict:
        return {key: edit.text().strip() for key, edit in self._edits.items()}


# ── Main station widget ────────────────────────────────────────────────────────

class SampleStation(QMainWindow):
    _process_frame = pyqtSignal(object, int, int)   # raw_value, width, height → triggers _ImageProcessor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ASWAXS Sample Station")

        # Config
        self.cfg = self._load_config()

        # Camera / calibration state
        self.cf            = 0.002450
        self.roisize       = 60
        self.offsetFactor  = 1.0 / 3.3538
        self.x_offset      = 0.0
        self.y_offset      = 0.0
        self.positions: list[dict] = []
        self.calibration_flag = False
        self.calib_chosen  = 1
        self.calib_pos     = [[0, 0], [1, 1]]
        self.pos1          = [0, 0]
        self.pos2          = [1, 1]
        self.image: np.ndarray | None = None
        self.image_width   = 1280
        self.image_height        = 960
        self.image_height_sensor = 960   # from ArraySizeY_RBV
        self.image_cx      = 640
        self.image_cy      = 480
        self.cursor_x      = 0
        self.cursor_y      = 0
        self.beam_x        = 640
        self.beam_y        = 480
        self._acquire_pv: PV | None = None
        self._center_initialized = False
        self._last_frame_time  = 0.0
        self._cam_fps_limit    = 15        # max display frames per second

        # Camera image processing thread
        self._img_thread    = QThread(self)
        self._img_processor = _ImageProcessor()
        self._img_processor.moveToThread(self._img_thread)
        self._process_frame.connect(self._img_processor.process)
        self._img_processor.processed.connect(self._on_image_processed)
        self._img_thread.start()

        self._procs: list = []
        atexit.register(self._cleanup_procs)

        self._build_ui()
        self._apply_style()
        self._load_calib_file()
        self._init_overlays()
        self._apply_config()

    # ── Config ─────────────────────────────────────────────────────────────

    def _load_config(self) -> dict:
        cfg = dict(DEFAULT_CONFIG)
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE) as f:
                    cfg.update(json.load(f))
            except Exception as e:
                print(f"Config load error: {e}")
        return cfg

    def _save_config(self):
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.cfg, f, indent=2)
            QMessageBox.information(self, "Saved", "Configuration saved to file.")
        except Exception as e:
            QMessageBox.warning(self, "Config Error", f"Could not save:\n{e}")

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)
        v.addWidget(self._build_motor_bar())
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_camera_tab(),    "Camera")
        self.tabs.addTab(self._build_positions_tab(), "Sample Positions")
        v.addWidget(self.tabs, 1)
        self._build_menu_bar()
        self._connect_signals()

    def _build_menu_bar(self):
        mb = self.menuBar()

        # ── File ────────────────────────────────────────────────────────────
        fm = mb.addMenu("&File")
        fm.addAction("New Positions",        lambda: self.pos_tab._new())
        fm.addAction("Open Positions…",      lambda: self.pos_tab._open())
        fm.addAction("Save Positions",       lambda: self.pos_tab._save())
        fm.addAction("Save Positions As…",   lambda: self.pos_tab._save_as())
        fm.addSeparator()
        ex = fm.addMenu("Export")
        ex.addAction("Bluesky CSV…",         lambda: self.pos_tab._export_bluesky())
        ex.addAction("Reducer Pairs CSV…",   lambda: self.pos_tab._export_reducer())
        fm.addSeparator()
        fm.addAction("Exit",                 self.close)

        # ── Acquisition ─────────────────────────────────────────────────────
        am = mb.addMenu("&Acquisition")
        am.addAction("▶  Start Camera",      self._start_camera)
        am.addAction("■  Stop Camera",       self._stop_camera)
        am.addSeparator()
        am.addAction("Autofocus",            self._run_autofocus)
        am.addAction("Calibrate Camera…",    self._open_calib_dialog)

        # ── Positions ────────────────────────────────────────────────────────
        pm = mb.addMenu("&Positions")
        pm.addAction("📍 Capture from Stage", lambda: self.pos_tab._capture_from_stage())
        pm.addSeparator()
        pm.addAction("Add Row",              lambda: self.pos_tab._add_row())
        pm.addAction("Delete Selected",      lambda: self.pos_tab._delete_selected())
        pm.addAction("Duplicate",            lambda: self.pos_tab._duplicate_selected())
        pm.addAction("Move Up",              lambda: self.pos_tab._move_selected(-1))
        pm.addAction("Move Down",            lambda: self.pos_tab._move_selected(1))
        pm.addSeparator()
        tm = pm.addMenu("Templates")
        tm.addAction("Freeform",             lambda: self.pos_tab.set_positions([blank_position(i) for i in range(5)]))
        tm.addAction("Capillary Linear…",    lambda: self.pos_tab._capillary_dialog())
        tm.addAction("Rack Builder…",        lambda: self.pos_tab._rack_builder_action())
        tm.addAction("Chip Manual Map",      lambda: self.pos_tab.set_positions([blank_position(i, layout="chip") for i in range(10)]))

        # ── Blender ──────────────────────────────────────────────────────────
        bm = mb.addMenu("&Blender")
        bm.addAction("⚙  Run Interpolation", lambda: self.pos_tab._run_blender())
        bm.addSeparator()
        bm.addAction("Blender Settings…",    self._open_blender_settings_dialog)

        # ── Setup ────────────────────────────────────────────────────────────
        sm = mb.addMenu("&Setup")
        sm.addAction("Open Setup…",          self._open_setup_dialog)
        sm.addSeparator()
        sm.addAction("Save Config",          self._save_config)
        sm.addAction("Reset to Defaults",    self._reset_config)

        # ── Help ─────────────────────────────────────────────────────────────
        hm = mb.addMenu("&Help")
        hm.addAction("About",               self._show_about)

    def _open_calib_dialog(self):
        dlg = _CalibDialog(self.cf, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.cf = dlg.factor()
            if hasattr(self, 'cf_edit'):
                self.cf_edit.setText(f"{self.cf:.6f}")

    def _open_setup_dialog(self, focus_group: str | None = None):
        dlg = SetupDialog(self.cfg, self)
        dlg.setWindowTitle("Setup — ASWAXS Sample Station")
        if focus_group:
            dlg.scroll_to(focus_group)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.cfg.update(dlg.values())
            self._save_config()
            self._apply_config()

    def _open_blender_settings_dialog(self):
        self._open_setup_dialog(focus_group="ssh")

    def _reset_config(self):
        reply = QMessageBox.question(self, "Reset Config",
            "Reset all settings to factory defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.cfg = dict(DEFAULT_CONFIG)
        self._apply_config()
        QMessageBox.information(self, "Reset", "Settings reset to defaults and applied.")

    def _show_about(self):
        QMessageBox.about(self, "About ASWAXS Sample Station",
            "<b>ASWAXS Sample Station</b><br>"
            "Version 1.0<br><br>"
            "Combines live camera/motor control with rich sample position planning.<br><br>"
            "Features:<br>"
            "• EPICS motor control (X/Y/Z)<br>"
            "• Live camera with click-to-move<br>"
            "• Rich position management (name, role, group, solvent)<br>"
            "• Role-colored position map<br>"
            "• Capillary rack builder<br>"
            "• Blender path interpolation via SSH<br>"
            "• Export to Bluesky CSV and Reducer Pairs CSV")

    def _build_motor_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("motorBar")
        outer = QVBoxLayout(bar)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(0)

        self.x_motor = MotorPanel("X")
        self.y_motor = MotorPanel("Y")
        self.z_motor = MotorPanel("Z (Focus)")

        # Single shared grid — all motor sub-widgets share column definitions
        grid_w = QWidget()
        grid = QGridLayout(grid_w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(0)
        grid.setColumnStretch(11, 1)   # spacer between motor cols and right extras

        # Motor rows at grid rows 0, 2, 4; separators at 1, 3
        self.x_motor.place_in_grid(grid, 0)
        self.y_motor.place_in_grid(grid, 2)
        self.z_motor.place_in_grid(grid, 4)

        sep0 = QFrame(); sep0.setFrameShape(QFrame.Shape.HLine); sep0.setObjectName("motorSep")
        sep1 = QFrame(); sep1.setFrameShape(QFrame.Shape.HLine); sep1.setObjectName("motorSep")
        grid.addWidget(sep0, 1, 0, 1, 13)
        grid.addWidget(sep1, 3, 0, 1, 13)

        # Right-side extras in column 12
        # X row: ROI
        roi_w = QWidget()
        roi_h = QHBoxLayout(roi_w)
        roi_h.setContentsMargins(0, 0, 0, 0)
        roi_h.setSpacing(4)
        roi_h.addWidget(QLabel("ROI:"))
        self.roi_edit = QLineEdit("60")
        self.roi_edit.setFixedWidth(44)
        roi_h.addWidget(self.roi_edit)
        grid.addWidget(roi_w, 0, 12)

        # Y row: offset label
        self.offset_label = QLabel("Offset: X=0.000000, Y=0.000000")
        self.offset_label.setObjectName("offsetLabel")
        self.offset_label.setVisible(False)
        grid.addWidget(self.offset_label, 2, 12)

        # Z row: action buttons
        btns_w = QWidget()
        btns_h = QHBoxLayout(btns_w)
        btns_h.setContentsMargins(0, 0, 0, 0)
        btns_h.setSpacing(4)
        self.calc_offset_btn = QPushButton("Calc Offset")
        self.calc_offset_btn.setObjectName("actionBtn")
        self.calc_offset_btn.setFixedHeight(26)
        self.calc_offset_btn.setVisible(False)
        self.center_x_btn = QPushButton("Cen X")
        self.center_x_btn.setObjectName("actionBtn")
        self.center_x_btn.setFixedHeight(26)
        self.center_x_btn.setVisible(False)
        self.center_y_btn = QPushButton("Cen Y")
        self.center_y_btn.setObjectName("actionBtn")
        self.center_y_btn.setFixedHeight(26)
        self.center_y_btn.setVisible(False)
        btns_h.addWidget(self.calc_offset_btn)
        btns_h.addWidget(self.center_x_btn)
        btns_h.addWidget(self.center_y_btn)
        grid.addWidget(btns_w, 4, 12)

        outer.addWidget(grid_w)
        return bar

    def _build_camera_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        # Top toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

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

        # Camera image
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
        self.pos_tab = SamplePositionTab(station=self)
        return self.pos_tab

    # _build_setup_tab removed — setup is now SetupDialog (see class below)

    # ── Signal wiring ──────────────────────────────────────────────────────

    def _connect_signals(self):
        # Motor bar
        self.calc_offset_btn.clicked.connect(self._calc_offset)
        self.center_x_btn.clicked.connect(self._center_x)
        self.center_y_btn.clicked.connect(self._center_y)
        self.roi_edit.returnPressed.connect(self.roiSizeChanged)

        # Camera tab
        self.cf_edit.returnPressed.connect(self.cfChanged)
        self.calibrate_btn.clicked.connect(self.openCalibration)
        self.autofocus_btn.clicked.connect(self._run_autofocus)
        self.cam_start_btn.clicked.connect(self._start_camera)
        self.cam_stop_btn.clicked.connect(self._stop_camera)

        # Camera mouse events
        scene = self.view_box.scene()
        scene.sigMouseMoved.connect(self._update_cursor_label)
        scene.sigMouseClicked.connect(self._on_camera_click)

    # ── EPICS connection ───────────────────────────────────────────────────

    def _apply_config(self):
        # cfg is already up-to-date (SetupDialog.values() was merged before calling this)
        self._apply_motor_config()
        self._apply_camera_config()

    def _apply_motor_config(self):
        self.x_motor.connect(self.cfg["X_MOTOR_PV"])
        self.y_motor.connect(self.cfg["Y_MOTOR_PV"])
        self.z_motor.connect(self.cfg["Z_MOTOR_PV"])

    def _apply_camera_config(self):
        # Stop any pending connection timeout
        if hasattr(self, '_cam_conn_timer'):
            self._cam_conn_timer.stop()

        for attr in ('_img_pv', '_wid_pv', '_hgt_pv', '_state_pv'):
            old = getattr(self, attr, None)
            if old is not None:
                try:
                    old.disconnect()
                except Exception:
                    pass

        if not EPICS_AVAILABLE:
            if hasattr(self, 'cam_state_lbl'):
                self.cam_state_lbl.setText("● Camera: offline (no EPICS)")
                self.cam_state_lbl.setStyleSheet("color: #9ca3af; font-size: 8.5pt;")
            return

        cam = self.cfg.get("CAMERA_PREFIX", "").strip()
        img = self.cfg.get("IMAGE_PREFIX",  "").strip()
        if not cam or not img:
            self.cam_state_lbl.setText("● Camera: prefix not configured")
            self.cam_state_lbl.setStyleSheet("color: #dc2626; font-size: 8.5pt;")
            return

        try:
            self._img_bridge   = _PVBridge()
            self._wid_bridge   = _PVBridge()
            self._hgt_bridge   = _PVBridge()
            self._state_bridge = _PVBridge()
            self._acq_bridge   = _PVBridge()

            self._img_bridge.changed.connect(self._on_image_data)
            self._wid_bridge.changed.connect(self._on_width_data)
            self._hgt_bridge.changed.connect(self._on_height_data)
            self._state_bridge.changed.connect(self._on_cam_state)
            self._state_bridge.conn_state.connect(self._on_cam_conn)
            self._acq_bridge.conn_state.connect(self._on_acquire_conn)

            self._img_pv   = PV(img + "ArrayData",
                                callback=self._img_bridge, auto_monitor=True)
            self._wid_pv   = PV(cam + "ArraySizeX_RBV",
                                callback=self._wid_bridge, auto_monitor=True)
            self._hgt_pv   = PV(cam + "ArraySizeY_RBV",
                                callback=self._hgt_bridge, auto_monitor=True)
            self._state_pv = PV(cam + "DetectorState_RBV",
                                callback=self._state_bridge,
                                connection_callback=self._state_bridge.conn_cb,
                                auto_monitor=True)

            # Acquire PV — call put(1) immediately (pyepics queues it until connected)
            # Also attach connection_callback as a fallback for reconfigure calls
            if self._acquire_pv is not None:
                try:
                    self._acquire_pv.disconnect()
                except Exception:
                    pass
            self._acquire_pv = PV(cam + "Acquire",
                                  connection_callback=self._acq_bridge.conn_cb)
            atexit.register(self._stop_acquire)
            self._acquire_pv.put(1)   # pyepics queues internally if not yet connected

            # Show connecting state immediately
            self.cam_state_lbl.setText("● Camera: connecting…")
            self.cam_state_lbl.setStyleSheet("color: #d97706; font-size: 8.5pt;")

            # Timeout: if DetectorState_RBV has not connected in 5 s, warn user
            self._cam_conn_timer = QTimer(self)
            self._cam_conn_timer.setSingleShot(True)
            self._cam_conn_timer.timeout.connect(self._on_cam_conn_timeout)
            self._cam_conn_timer.start(5000)

        except Exception as e:
            print(f"Camera PV setup error: {e}")
            self.cam_state_lbl.setText(f"● Camera: error — {e}")
            self.cam_state_lbl.setStyleSheet("color: #dc2626; font-size: 8.5pt;")

    @pyqtSlot(str, bool)
    def _on_cam_conn(self, _pvname: str, conn: bool):
        """DetectorState_RBV connection callback — fires when EPICS actually connects."""
        if conn:
            # Cancel the timeout — we have a live PV
            if hasattr(self, '_cam_conn_timer'):
                self._cam_conn_timer.stop()
        else:
            self.cam_state_lbl.setText("● Camera: disconnected")
            self.cam_state_lbl.setStyleSheet("color: #dc2626; font-size: 8.5pt;")

    @pyqtSlot(str, bool)
    def _on_acquire_conn(self, _pvname: str, conn: bool):
        """Start acquiring only once the Acquire PV has actually connected."""
        if conn and self._acquire_pv is not None:
            self._acquire_pv.put(1)

    def _on_cam_conn_timeout(self):
        """Fired 5 s after _apply_camera_config if camera PV never responded."""
        connected = self._state_pv is not None and getattr(self._state_pv, 'connected', False)
        if not connected:
            cam = self.cfg.get("CAMERA_PREFIX", "?")
            self.cam_state_lbl.setText("● Camera: no response")
            self.cam_state_lbl.setStyleSheet("color: #dc2626; font-size: 8.5pt;")
            self.cam_state_lbl.setToolTip(
                f"PV '{cam}DetectorState_RBV' did not connect after 5 s.\n"
                f"Check CAMERA_PREFIX in Setup."
            )

    def _stop_acquire(self):
        if self._acquire_pv:
            try:
                self._acquire_pv.put(0)
            except Exception:
                pass

    # ── Camera PV callbacks ────────────────────────────────────────────────

    @pyqtSlot(str, object)
    def _on_width_data(self, _pvname: str, value):
        self.image_width = int(value)

    @pyqtSlot(str, object)
    def _on_height_data(self, _pvname: str, value):
        self.image_height_sensor = int(value)

    @pyqtSlot(str, object)
    def _on_cam_state(self, _pvname: str, value):
        text = str(value) if value is not None else "—"
        self.cam_state_lbl.setText(f"● Camera: {text}")
        low = text.lower()
        if any(k in low for k in ("acquire", "idle", "wait")):
            color = "#16a34a"   # green — live
        elif any(k in low for k in ("error", "abort", "fault", "disconnect")):
            color = "#dc2626"   # red — problem
        else:
            color = "#d97706"   # amber — intermediate state
        self.cam_state_lbl.setStyleSheet(f"color: {color}; font-size: 8.5pt;")

    @pyqtSlot(str, object)
    def _on_image_data(self, _pvname: str, value):
        now = time.monotonic()
        if now - self._last_frame_time < 1.0 / self._cam_fps_limit:
            return
        self._last_frame_time = now
        self._process_frame.emit(value, self.image_width, self.image_height_sensor)

    @pyqtSlot(object, object, float, int, int)
    def _on_image_processed(self, rgb, gray, focus_score: float, w: int, h: int):
        try:
            self.image        = gray
            self.image_height = h
            self.image_item.setImage(rgb, autoLevels=False, levels=(0, 255))
            if not self._center_initialized:
                self.image_cx = w // 2
                self.image_cy = h // 2
                self._center_x_line.setValue(w / 2)
                self._center_y_line.setValue(h / 2)
                self._center_initialized = True
            if CV2_AVAILABLE:
                self.focus_lbl.setText(f"{focus_score:.3f}")
            else:
                self.focus_lbl.setText("(cv2 N/A)")
        except Exception as e:
            print(f"[_on_image_processed] {e}")

    def _on_global_pv(self, pvname: str, value):
        pass

    # ── Overlay graphics ───────────────────────────────────────────────────

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

    # ── Camera controls ────────────────────────────────────────────────────

    def _start_camera(self):
        if self._acquire_pv:
            self._acquire_pv.put(1)

    def _stop_camera(self):
        if self._acquire_pv:
            self._acquire_pv.put(0)

    # ── Mouse events ───────────────────────────────────────────────────────

    @pyqtSlot(object)
    def _update_cursor_label(self, pos):
        try:
            coords = self.image_item.mapFromScene(pos)
            x, y   = int(coords.x()), int(coords.y())
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

    def _update_beam_label(self):
        self.beam_lbl.setText(f"BeamX={self.beam_x:4d}, BeamY={self.beam_y:4d}")

    @pyqtSlot(object)
    def _on_camera_click(self, event):
        x, y = self.cursor_x, self.cursor_y
        if self.image is None:
            return

        if event._double and self.click_move_cb.isChecked():
            if 0 <= x < self.image_width and 0 <= y < self.image_height:
                new_x = self.x_motor.get_sp() - self.cf * (x - self.image_width  / 2 - 1)
                new_y = self.y_motor.get_sp() + self.cf * (y - self.image_height / 2 + 1)
                self.x_motor.move_to(new_x)
                self.y_motor.move_to(new_y)
                if self.auto_add_cb.isChecked():
                    # Wait for both motors then auto-add — non-blocking via QTimer
                    def _check_and_add(xm=self.x_motor, ym=self.y_motor,
                                       t=QTimer(self), add=self.addPosition):
                        if not xm.is_moving() and not ym.is_moving():
                            t.stop(); t.deleteLater(); add()
                    _t = QTimer(self); _t.setInterval(50)
                    _t.timeout.connect(lambda: _check_and_add(t=_t))
                    _t.start()

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
            self._update_beam_label()
            self._beam_x_curve.setData(x=[x, x],         y=[y - 10, y + 10])
            self._beam_y_curve.setData(x=[x - 10, x + 10], y=[y, y])
            self._beam_x_curve.show()
            self._beam_y_curve.show()

    # ── Calibration factor ─────────────────────────────────────────────────

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
            with open(CALIB_FILE, 'w') as f:
                f.write(f"#Calibration saved {time.asctime()}\n")
                f.write(f"cf={self.cf:.6f}\n")
        except Exception as e:
            QMessageBox.warning(self, "File Error", f"Could not save calibration:\n{e}")

    # ── Calibration widget ─────────────────────────────────────────────────

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

    # ── ROI / beam centering ───────────────────────────────────────────────

    def roiSizeChanged(self):
        try:
            self.roisize = int(self.roi_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Value Error", "Integer only.")
            self.roi_edit.setText(str(self.roisize))

    def _calc_offset(self):
        if self.image is None:
            QMessageBox.warning(self, "No Image", "Camera not streaming.")
            return
        cx, cy, r = self.image_cx, self.image_cy, self.roisize
        roi1 = self.image[cy - r:cy,       cx - r:cx    ]
        roi2 = self.image[cy - r:cy,       cx:cx + r    ]
        roi3 = self.image[cy:cy + r,       cx:cx + r    ]
        roi4 = self.image[cy:cy + r,       cx - r:cx    ]

        int_max  = max(r_.max() for r_ in (roi1, roi2, roi3, roi4))
        rois     = [np.abs(r_ - int_max) for r_ in (roi1, roi2, roi3, roi4)]
        int_max2 = max(r_.max() for r_ in rois)
        thresh   = 0.1 * int_max2

        s     = [np.sum(r_ > thresh) for r_ in rois]
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

    def _center_x(self):
        self._calc_offset()
        if abs(self.x_offset) > 0.005:
            self.x_motor.move_to(self.x_motor.get_sp() - self.x_offset)
            self._wait_motor_done(self.x_motor, self._calc_offset)

    def _center_y(self):
        self._calc_offset()
        if abs(self.y_offset) > 0.005:
            self.y_motor.move_to(self.y_motor.get_sp() - self.y_offset)
            self._wait_motor_done(self.y_motor, self._calc_offset)

    def _wait_motor_done(self, motor, callback=None):
        """Poll motor MOVN every 50 ms via QTimer — no processEvents blocking."""
        timer = QTimer(self)
        timer.setInterval(50)
        def _check():
            if not motor.is_moving():
                timer.stop()
                timer.deleteLater()
                if callback:
                    callback()
        timer.timeout.connect(_check)
        timer.start()

    # ── Autofocus ──────────────────────────────────────────────────────────

    def _run_autofocus(self):
        script = self.cfg.get("AUTOFOCUS_SCRIPT", "")
        if not os.path.exists(script):
            QMessageBox.warning(self, "Autofocus", f"Script not found:\n{script}")
            return
        cam  = self.cfg["CAMERA_PREFIX"]
        zpv  = self.cfg["Z_MOTOR_PV"]
        step = self.cfg.get("AUTOFOCUS_STEP", "0.2")
        proc = subprocess.Popen([sys.executable, script, cam, zpv, step])
        self._procs.append(proc)

    # ── Position capture ───────────────────────────────────────────────────

    def addPosition(self):
        if hasattr(self, 'pos_tab'):
            self.pos_tab._capture_from_stage()

    # ── Blender SSH callbacks (used by _BlenderWorker thread) ─────────────

    def _on_blender_done(self, result, thread):
        thread.quit(); thread.wait()
        self.pos_tab.blender_btn.setEnabled(True)
        self.pos_tab.blender_btn.setText("Run Blender")
        if result is None:
            QMessageBox.critical(self, "Blender Error",
                                 "Output file not found — Blender may have failed.")
            return
        self.pos_tab._apply_blender_result(result)

    def _on_blender_error(self, msg: str, thread):
        thread.quit(); thread.wait()
        self.pos_tab.blender_btn.setEnabled(True)
        self.pos_tab.blender_btn.setText("Run Blender")
        QMessageBox.critical(self, "SSH Error", msg)

    # ── Blender SSH (returns list of dicts or None) — kept for direct/testing use

    def _run_blender_ssh(self, positions: list, spacing: float):
        if not PARAMIKO_AVAILABLE:
            QMessageBox.critical(self, "SSH Error",
                                 "paramiko is not installed.\nRun: pip install paramiko")
            return None

        local_mount  = self.cfg.get("LOCAL_MOUNT", "")
        remote_mount = self.cfg.get("REMOTE_MOUNT", "")
        data_dir     = os.path.join(_DIR, "Data")
        os.makedirs(data_dir, exist_ok=True)

        # Write temp input pos file
        temp_pos = os.path.join(data_dir, "temp_blender_in.pos")
        try:
            with open(temp_pos, 'w') as f:
                f.write("# x y z\n")
                for p in normalize_positions(positions):
                    f.write(f"{float(p['x']):.6f} {float(p['y']):.6f} {float(p['z']):.6f}\n")
        except Exception as e:
            QMessageBox.critical(self, "Blender Error", f"Could not write temp file:\n{e}")
            return None

        temp_out = os.path.join(data_dir, "temp_blender_out.csv")
        lifname  = os.path.abspath(temp_pos)
        lofname  = os.path.abspath(temp_out)
        ifname   = lifname.replace(local_mount, remote_mount).replace('\\', '/')
        ofname   = lofname.replace(local_mount, remote_mount).replace('\\', '/')

        blender_exe    = self.cfg.get("BLENDER_EXE", "blender") or "blender"
        blender_script = self.cfg.get("BLENDER_SCRIPT", "")
        hostname = self.cfg.get("BLENDER_HOST", "")
        username = self.cfg.get("BLENDER_USER", "")
        key_file = self.cfg.get("BLENDER_KEY", "")
        cmd = [blender_exe, '--background', '--python', blender_script,
               '--', ifname, ofname, f"{spacing:.2f}"]

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(hostname, port=22, username=username, key_filename=key_file)
            _stdin, stdout, stderr = client.exec_command(" ".join(cmd))
            out = stdout.read().decode()
            err = stderr.read().decode()
            if out:
                print(out)
            if err:
                print("stderr:", err)
            # Try SFTP download in case file system is not shared
            if not os.path.exists(lofname):
                try:
                    sftp = client.open_sftp()
                    sftp.get(ofname, lofname)
                    sftp.close()
                except Exception:
                    pass
        except Exception as e:
            QMessageBox.critical(self, "SSH Error", str(e))
            return None
        finally:
            client.close()

        if not os.path.exists(lofname):
            QMessageBox.critical(self, "Blender Error",
                                 "Output file not found — Blender may have failed.")
            return None

        # Parse CSV: columns #Vertex_X, Vertex_Y, Vertex_Z, Normal_X, Normal_Y, Normal_Z
        parsed_rows: list[dict] = []
        try:
            with open(lofname, newline='', encoding='utf-8', errors='replace') as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith('#'):
                        continue
                    parts = [x.strip() for x in stripped.split(',')]
                    if len(parts) < 3:
                        continue
                    try:
                        parsed_rows.append({
                            "x": float(parts[0]),
                            "y": float(parts[1]),
                            "z": float(parts[2]),
                        })
                    except ValueError:
                        continue
        except Exception as e:
            QMessageBox.critical(self, "Blender Error", f"Could not parse output:\n{e}")
            return None

        return parsed_rows

    # ── Process cleanup ────────────────────────────────────────────────────

    def _cleanup_procs(self):
        for p in self._procs:
            try:
                p.terminate()
            except Exception:
                pass

    def closeEvent(self, event):
        self._img_thread.quit()
        self._img_thread.wait(2000)
        super().closeEvent(event)

    # ── Style ──────────────────────────────────────────────────────────────

    def _apply_style(self):
        self.setStyleSheet("""
        /* ── Base ── */
        QWidget {
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            font-size: 9pt;
            color: #20242a;
            background-color: #f3f4f6;
        }

        /* ── Motor bar frame ── */
        QFrame#motorBar {
            background: #ffffff;
            border: 1px solid #c8ccd2;
            border-radius: 6px;
        }
        QFrame#motorSep {
            color: #c8ccd2;
            max-height: 1px;
            margin: 0 0;
        }

        /* ── Axis badge ── */
        QLabel#axisLabel {
            background: #2f6fae;
            color: white;
            border-radius: 4px;
            font-weight: bold;
            font-size: 9pt;
            padding: 1px 2px;
        }

        /* ── Field tags (RBV, SP, Step) ── */
        QLabel#fieldTag {
            color: #6b7a8d;
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
        QLabel#movnIdle   { color: #c8ccd2; font-size: 11pt; background: transparent; }
        QLabel#movnActive { color: #e07b00; font-size: 11pt; background: transparent; }

        /* ── Offset / status labels ── */
        QLabel#offsetLabel {
            color: #6b7a8d;
            font-style: italic;
            font-size: 8.5pt;
            background: transparent;
        }
        QLabel#focusLabel {
            font-family: "Consolas", monospace;
            color: #1f4f7f;
            background: #edf4fd;
            border: 1px solid #b9c0ca;
            border-radius: 3px;
            padding: 1px 4px;
        }
        QLabel#camStateLabel {
            color: #6b7a8d;
            font-size: 8.5pt;
            background: transparent;
        }
        QLabel#statusLabel {
            font-family: "Consolas", monospace;
            font-size: 8.5pt;
            color: #20242a;
            background: transparent;
        }
        QLabel#descLabel { background: transparent; }

        /* ── Buttons (base) ── */
        QPushButton {
            background: #f8f9fb;
            border: 1px solid #b9c0ca;
            border-radius: 4px;
            padding: 3px 10px;
            min-height: 24px;
            color: #20242a;
        }
        QPushButton:hover   { background: #edf4fd; border-color: #2f6fae; }
        QPushButton:pressed { background: #2f6fae; color: white; border-color: #1f4f7f; }
        QPushButton:disabled { color: #a0a8b4; background: #f3f4f6; border-color: #dde1e7; }

        /* ── Tweak (◀▶) buttons ── */
        QPushButton#tweakBtn {
            background: #edf4fd;
            border: 1px solid #b9c0ca;
            border-radius: 3px;
            padding: 1px 2px;
            font-size: 9pt;
            min-height: 22px;
        }
        QPushButton#tweakBtn:hover   { background: #d4e6f8; border-color: #2f6fae; }
        QPushButton#tweakBtn:pressed { background: #2f6fae; color: white; }

        /* ── Action (green-ish) buttons ── */
        QPushButton#actionBtn {
            background: #f0fdf4;
            border: 1px solid #86efac;
            border-radius: 4px;
            color: #14532d;
            padding: 3px 8px;
        }
        QPushButton#actionBtn:hover   { background: #dcfce7; border-color: #4ade80; }
        QPushButton#actionBtn:pressed { background: #16a34a; color: white; }

        QPushButton#greenBtn {
            background: #f0fdf4; border: 1px solid #86efac;
            color: #166534; border-radius: 4px;
        }
        QPushButton#greenBtn:hover   { background: #dcfce7; }
        QPushButton#greenBtn:pressed { background: #16a34a; color: white; }

        QPushButton#redBtn {
            background: #fef2f2; border: 1px solid #fca5a5;
            color: #991b1b; border-radius: 4px;
        }
        QPushButton#redBtn:hover   { background: #fee2e2; }
        QPushButton#redBtn:pressed { background: #dc2626; color: white; }

        QPushButton#captureBtn {
            background: #fef3c7; border: 1px solid #fbbf24;
            color: #92400e; border-radius: 4px; font-weight: bold; padding: 3px 10px;
        }
        QPushButton#captureBtn:hover   { background: #fde68a; border-color: #f59e0b; }
        QPushButton#captureBtn:pressed { background: #f59e0b; color: white; }

        /* ── Line edits ── */
        QLineEdit {
            background: #ffffff;
            border: 1px solid #b9c0ca;
            border-radius: 3px;
            padding: 2px 5px;
            selection-background-color: #2f6fae;
            selection-color: white;
        }
        QLineEdit:focus { border-color: #2f6fae; }

        /* ── Tab widget ── */
        QTabWidget::pane {
            border: 1px solid #c8ccd2;
            border-radius: 0 4px 4px 4px;
            background: #ffffff;
        }
        QTabBar::tab {
            background: #e6e9ef;
            border: 1px solid #c8ccd2;
            border-bottom: none;
            border-radius: 4px 4px 0 0;
            padding: 5px 16px;
            margin-right: 2px;
            color: #4a5568;
        }
        QTabBar::tab:selected {
            background: #ffffff;
            color: #1f4f7f;
            font-weight: bold;
            border-bottom: 2px solid #2f6fae;
        }
        QTabBar::tab:hover:!selected { background: #d4e6f8; color: #2f6fae; }

        /* ── Group boxes ── */
        QGroupBox {
            border: 1px solid #c8ccd2;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 6px;
            font-weight: bold;
            color: #28313f;
            background: #ffffff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 6px;
            left: 10px;
            color: #28313f;
        }

        /* ── List widget ── */
        QListWidget {
            background: #ffffff;
            border: 1px solid #c8ccd2;
            border-radius: 4px;
            alternate-background-color: #f7f9fb;
            outline: none;
        }
        QListWidget::item { padding: 3px 6px; }
        QListWidget::item:selected { background: #2f6fae; color: white; }
        QListWidget::item:hover:!selected { background: #edf4fd; }

        /* ── Check boxes ── */
        QCheckBox { spacing: 5px; background: transparent; }
        QCheckBox::indicator {
            width: 14px; height: 14px;
            border: 1px solid #b9c0ca;
            border-radius: 3px;
            background: white;
        }
        QCheckBox::indicator:checked {
            background: #2f6fae;
            border-color: #1f4f7f;
        }

        /* ── Scroll area ── */
        QScrollArea { border: none; background: transparent; }
        QScrollBar:vertical {
            background: #f3f4f6; width: 10px; margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #b9c0ca; border-radius: 4px; min-height: 20px;
        }
        QScrollBar::handle:vertical:hover { background: #2f6fae; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

        /* ── Vertical separators ── */
        QFrame[frameShape="5"] { color: #c8ccd2; max-width: 1px; }
        QFrame#toolSep { color: #c8ccd2; max-width: 1px; }

        /* ── Table widget ── */
        QTableWidget {
            background: #ffffff;
            border: 1px solid #c8ccd2;
            gridline-color: #e6e9ef;
            alternate-background-color: #f7f9fb;
            outline: none;
        }
        QTableWidget QHeaderView::section {
            background: #eceff3;
            border: none;
            border-right: 1px solid #c8ccd2;
            border-bottom: 1px solid #c8ccd2;
            padding: 3px 6px;
            font-weight: bold;
            color: #28313f;
        }
        QTableWidget::item:selected { background: #2f6fae; color: white; }

        /* ── Combo box ── */
        QComboBox {
            background: #ffffff;
            border: 1px solid #b9c0ca;
            border-radius: 3px;
            padding: 2px 6px;
            min-height: 22px;
        }
        QComboBox:focus { border-color: #2f6fae; }
        QComboBox::drop-down { border: none; width: 18px; }
        QComboBox QAbstractItemView {
            background: #ffffff;
            border: 1px solid #c8ccd2;
            selection-background-color: #2f6fae;
            selection-color: white;
        }

        /* ── Spin boxes ── */
        QSpinBox, QDoubleSpinBox {
            background: #ffffff;
            border: 1px solid #b9c0ca;
            border-radius: 3px;
            padding: 2px 5px;
        }
        QSpinBox:focus, QDoubleSpinBox:focus { border-color: #2f6fae; }

        /* ── Menu bar (light theme matching FrameByFrame) ── */
        QMenuBar {
            background: #eceff3;
            color: #20242a;
            padding: 2px 4px;
            font-size: 9pt;
            spacing: 2px;
            border-bottom: 1px solid #c8ccd2;
        }
        QMenuBar::item {
            padding: 4px 14px;
            border-radius: 4px;
            background: transparent;
        }
        QMenuBar::item:selected { background: #2f6fae; color: white; }
        QMenuBar::item:pressed  { background: #1f4f7f; color: white; }

        QMenu {
            background: #ffffff;
            border: 1px solid #c8ccd2;
            border-radius: 6px;
            padding: 4px;
            font-size: 9pt;
        }
        QMenu::item {
            padding: 5px 32px 5px 14px;
            border-radius: 3px;
            color: #20242a;
        }
        QMenu::item:selected  { background: #edf4fd; color: #1f4f7f; }
        QMenu::item:disabled  { color: #a0a8b4; }
        QMenu::separator      { height: 1px; background: #c8ccd2; margin: 3px 10px; }
        QMenu::right-arrow    { image: none; width: 8px; }
    """)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SampleStation()
    w.resize(1500, 960)
    w.show()
    sys.exit(app.exec())
