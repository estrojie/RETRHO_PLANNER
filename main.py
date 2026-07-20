# main.py
from __future__ import annotations
import sys
import re
import warnings
from io       import BytesIO
from dataclasses import dataclass
from typing   import List, Optional, Tuple
from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.patches import Rectangle

from PySide6.QtCore    import Qt, QThread, Signal, QDate, QSize, QTimer, QSignalBlocker
from PySide6.QtGui     import QTextDocument, QFont, QImage, QPixmap, QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QGroupBox, QFormLayout, QLabel, QPushButton, QTabWidget, QFileDialog,
    QTableWidget, QTableWidgetItem, QMessageBox, QLineEdit, QSpinBox,
    QAbstractItemView, QComboBox, QDoubleSpinBox, QDateEdit, QDialog,
    QListWidget, QListWidgetItem, QCheckBox, QSizePolicy, QHeaderView,
    QStyle, QScrollArea, QGridLayout,
)

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astroquery.exceptions import NoResultsWarning

import planner_core as core

def std_icon(widget: QWidget, enum_name: str):
    sp = getattr(QStyle.StandardPixmap, enum_name, None) or \
         getattr(QStyle, enum_name, None) or \
         QStyle.StandardPixmap.SP_FileIcon
    return widget.style().standardIcon(sp)


def style_toolbar_button(btn: QPushButton):
    btn.setProperty("toolbarButton", True)
    btn.setCursor(Qt.PointingHandCursor)
    try:
        btn.style().unpolish(btn); btn.style().polish(btn)
    except Exception:
        pass


def copy_figure_to_clipboard(fig, parent=None, msg: str = "Plot copied to clipboard."):
    try:
        buf   = BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        image = QImage.fromData(buf.getvalue(), "PNG")
        if image.isNull():
            raise RuntimeError("Failed to create clipboard image.")
        QGuiApplication.clipboard().setPixmap(QPixmap.fromImage(image))
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(msg, 3000)
    except Exception as e:
        QMessageBox.warning(parent, "Clipboard Error", f"Could not copy plot.\n\n{e}")

def apply_app_style(app: QApplication):
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet("""
    QMainWindow {
        background-color: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #1f2228,stop:1 #181a1f);
    }
    QLabel { color: #d7d7d7; }
    QGroupBox {
        border:1px solid #3a3d45; border-radius:8px; margin-top:10px;
        padding:10px; font-weight:600; color:#d0d0d0;
        background-color:rgba(21,23,28,0.55);
    }
    QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 6px; color:#cfcfcf; }
    QLineEdit, QSpinBox, QDoubleSpinBox, QDateEdit, QComboBox {
        background-color:#14161b; border:1px solid #3a3d45; border-radius:6px;
        padding:5px; color:#e5e5e5; selection-background-color:#2b6cb0;
    }
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
    QDateEdit:focus, QComboBox:focus { border:1px solid #4a90e2; }
    QPushButton {
        background-color:#2a2d33; border:1px solid #3a3d45;
        padding:7px 10px; border-radius:6px; color:#f0f0f0;
    }
    QPushButton:hover { background-color:#353944; }
    QPushButton:pressed { background-color:#4b5060; }
    QPushButton[toolbarButton="true"] {
        background-color:#262b34; border:1px solid #41506a; padding:7px 12px;
    }
    QPushButton[toolbarButton="true"]:hover {
        background-color:#2e3f57; border:1px solid #5a8fd8;
    }
    QPushButton[toolbarButton="true"]:pressed {
        background-color:#1f5fa6; border:1px solid #79aef0;
    }
    QPushButton[toolbarButton="true"]:checked {
        background-color:#1b4f8a; border:1px solid #4a90e2; color:#ffffff;
    }
    QPushButton:disabled { color:#888; background-color:#23262c; border-color:#2c2f36; }
    QPushButton:checked  { background-color:#1b4f8a; border:1px solid #4a90e2; color:#ffffff; }
    QPushButton:checked:hover { background-color:#2262a6; }
    QTabWidget::pane {
        border:1px solid #3a3d45; border-radius:8px; top:-1px;
        background-color:rgba(21,23,28,0.35);
    }
    QTabBar::tab {
        background:#2a2d33; padding:7px 14px; border:1px solid #3a3d45;
        border-bottom:none; border-top-left-radius:7px; border-top-right-radius:7px;
        color:#cfcfcf; margin-right:2px;
    }
    QTabBar::tab:selected { background:#1b1d23; color:#ffffff; }
    QTableWidget {
        background-color:rgba(18,20,25,0.55); border:1px solid #3a3d45;
        border-radius:8px; gridline-color:#2f323a; color:#e5e5e5;
        selection-background-color:#1b4f8a; selection-color:#ffffff;
    }
    QHeaderView::section {
        background-color:#23262c; padding:6px; border:1px solid #3a3d45;
        color:#dcdcdc; font-weight:600;
    }
    QStatusBar {
        background-color:rgba(18,20,25,0.65); border-top:1px solid #3a3d45; color:#cfcfcf;
    }
    """)

@dataclass
class PlanRow:
    name:            str
    ra:              str
    dec:             str
    priority:        int = 3
    vmag:            str = "N/A"
    visible_windows: str = "—"
    notes:           str = ""

class PlanWorker(QThread):
    finished = Signal(list, object, list, list)
    failed   = Signal(str)

    def __init__(self, plan: List[PlanRow], min_alt: float, max_alt: float):
        super().__init__()
        self.plan    = plan
        self.min_alt = min_alt
        self.max_alt = max_alt

    def run(self):
        try:
            updated: List[PlanRow] = []
            coords, names = [], []

            for row in self.plan:
                try:
                    rt = core.resolve_target(row.name, row.ra, row.dec)
                    resolved_ok = True
                except Exception:
                    resolved_ok = False

                if not resolved_ok:
                    updated.append(PlanRow(
                        name             = row.name or "Unnamed Target",
                        ra               = row.ra  or "—",
                        dec              = row.dec or "—",
                        priority         = row.priority,
                        vmag             = "N/A",
                        visible_windows  = "—",
                        notes            = "Failed to Resolve (Check Name or Coordinates)",
                    ))
                    continue

                note = ""
                try:
                    if rt.coord.dec.deg > core.DEC_WARNING_LIMIT_DEG:
                        note = "Dec > +60° (Potentially Not Observable at RHO)"
                except Exception:
                    pass

                try:
                    windows   = core.compute_visibility_windows(rt.coord, self.min_alt, self.max_alt)
                    windows_s = core.format_visibility_windows(windows)
                except Exception:
                    windows_s = "—"

                preferred = row.vmag
                if str(preferred).strip().lower() in ("", "n/a", "na", "nan", "none", "—", "-"):
                    preferred = rt.vmag

                vmag_s = "N/A"
                try:
                    v = float(preferred)
                    if np.isfinite(v):
                        vmag_s = f"{v:.2f}"
                except Exception:
                    sv = str(preferred).strip()
                    if sv:
                        vmag_s = sv

                updated.append(PlanRow(
                    name            = rt.display_name,
                    ra              = rt.coord.ra.to_string(unit=core.u.hour, sep=":", precision=2),
                    dec             = rt.coord.dec.to_string(unit=core.u.deg,  sep=":", precision=2, alwayssign=True),
                    priority        = row.priority,
                    vmag            = vmag_s,
                    visible_windows = windows_s,
                    notes           = note,
                ))
                coords.append(rt.coord)
                names.append(rt.display_name)

            alt_fig = core.plot_altitudes(coords, names, self.min_alt, self.max_alt) if coords else None
            self.finished.emit(updated, alt_fig, coords, names)
        except Exception as e:
            self.failed.emit(str(e))


class FinderWorker(QThread):
    finished = Signal(int, object, object, object, object, object, object, object, str, str)
    failed   = Signal(int, str)

    def __init__(
        self,
        request_id: int,
        name: str, ra: str, dec: str,
        fov1_w: int, fov1_h: int,
        fov2_w: int, fov2_h: int,
        mode:     str,
        roll_deg: float = 0.0,
    ):
        super().__init__()
        self.request_id = request_id
        self.name       = name
        self.ra, self.dec = ra, dec
        self.fov1_w, self.fov1_h = fov1_w, fov1_h
        self.fov2_w, self.fov2_h = fov2_w, fov2_h
        self.mode     = mode
        self.roll_deg = float(roll_deg)

    def run(self):
        try:
            rt = core.resolve_target(self.name, self.ra, self.dec)

            data1, wcs1, lbl1 = core.fetch_finder_raw(
                rt.coord, self.fov1_w, self.mode, fov_h_arcmin=self.fov1_h)
            data2, wcs2, lbl2 = core.fetch_finder_raw(
                rt.coord, self.fov2_w, self.mode, fov_h_arcmin=self.fov2_h)

            if data1 is not None and wcs1 is not None:
                fig1 = core.render_finder_figure_from_data(
                    rt.coord, rt.display_name, data1, wcs1,
                    self.fov1_w, lbl1, self.roll_deg, fov_h_arcmin=self.fov1_h)
            else:
                fig1 = core._empty_finder_figure(rt.display_name, self.fov1_w, self.fov1_h)
                data1, wcs1, lbl1 = None, None, ""

            if data2 is not None and wcs2 is not None:
                fig2 = core.render_finder_figure_from_data(
                    rt.coord, rt.display_name, data2, wcs2,
                    self.fov2_w, lbl2, self.roll_deg, fov_h_arcmin=self.fov2_h)
            else:
                fig2 = core._empty_finder_figure(rt.display_name, self.fov2_w, self.fov2_h)
                data2, wcs2, lbl2 = None, None, ""

            self.finished.emit(
                self.request_id, fig1, fig2,
                data1, wcs1, data2, wcs2,
                rt.coord, lbl1, lbl2,
            )
        except Exception as e:
            self.failed.emit(self.request_id, str(e))


class StarIdWorker(QThread):
    finished = Signal(object)
    failed   = Signal(str)

    def __init__(self, coord: SkyCoord, radius_arcsec: float = 30.0):
        super().__init__()
        self.coord         = coord
        self.radius_arcsec = float(radius_arcsec)

    def run(self):
        try:
            result = core.identify_star_at_coord(self.coord, self.radius_arcsec)
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))

class AltitudeInspectorDialog(QDialog):
    def __init__(self, parent, coords, names, min_alt, max_alt):
        super().__init__(parent)
        self.setWindowTitle("Altitude/Airmass Inspector")
        self.resize(1100, 650)

        self.coords = list(coords)
        self.names  = list(names)
        self.min_alt = float(min_alt)
        self.max_alt = float(max_alt)
        self.max_selected_objects = 5
        self._updating_selection  = False

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        controls  = QWidget()
        cl        = QVBoxLayout(controls)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(10)

        self.chk_airmass = QCheckBox("Show Airmass (instead of Altitude)")
        self.chk_airmass.stateChanged.connect(self._replot)
        self.chk_utc = QCheckBox("Show UTC (instead of local time)")
        self.chk_utc.stateChanged.connect(self._replot)

        self.listw = QListWidget()
        self.listw.setSelectionMode(QAbstractItemView.MultiSelection)
        for nm in self.names:
            it = QListWidgetItem(nm)
            self.listw.addItem(it)
        self.listw.itemSelectionChanged.connect(self._on_selection_changed)

        lbl = QLabel("Objects to display (max 5):")

        btn_first5 = QPushButton("Select First 5")
        btn_first5.clicked.connect(self._select_first_five)

        btn_none = QPushButton("Unselect All")
        btn_none.clicked.connect(self._unselect_all)

        self.btn_copy_plot = QPushButton("Copy Plot")
        style_toolbar_button(self.btn_copy_plot)
        self.btn_copy_plot.setIcon(std_icon(self, "SP_DialogSaveButton"))
        self.btn_copy_plot.clicked.connect(
            lambda: copy_figure_to_clipboard(self.canvas.figure, self, "Altitude plot copied."))

        cl.addWidget(self.chk_airmass)
        cl.addWidget(self.chk_utc)
        cl.addWidget(lbl)
        cl.addWidget(btn_first5)
        cl.addWidget(btn_none)
        cl.addWidget(self.btn_copy_plot)
        cl.addWidget(self.listw, 1)
        controls.setFixedWidth(210)

        plot_w = QWidget()
        plot_l = QVBoxLayout(plot_w)
        plot_l.setContentsMargins(0, 0, 0, 0)
        self.canvas  = None
        self.toolbar = None
        root.addWidget(controls)
        root.addWidget(plot_w, 1)
        self._plot_widget = plot_w
        self._plot_layout = plot_l

        self._select_first_five()

    def _replace_plot(self, fig):
        if self.toolbar is not None:
            self._plot_layout.removeWidget(self.toolbar)
            self.toolbar.setParent(None)
        if self.canvas is not None:
            self._plot_layout.removeWidget(self.canvas)
            try: core.plt.close(self.canvas.figure)
            except Exception: pass
            self.canvas.setParent(None)
        self.canvas  = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self._plot_layout.addWidget(self.toolbar)
        self._plot_layout.addWidget(self.canvas, 1)

    def _replot(self):
        selected = [self.listw.item(i).text()
                    for i in range(self.listw.count())
                    if self.listw.item(i).isSelected()]
        if not selected:
            return
        y_mode   = "airmass" if self.chk_airmass.isChecked() else "altitude"
        tz_mode  = "utc"     if self.chk_utc.isChecked()    else "local"
        fig = core.plot_altitudes(
            self.coords, self.names, self.min_alt, self.max_alt,
            y_mode=y_mode, only_names=selected, display_tz=tz_mode,
        )
        self._replace_plot(fig)

    def _on_selection_changed(self):
        if self._updating_selection:
            return
        self._enforce_selection_limit()
        self._replot()

    def _select_first_five(self):
        self._updating_selection = True
        blocker = QSignalBlocker(self.listw)
        try:
            for i in range(self.listw.count()):
                self.listw.item(i).setSelected(i < self.max_selected_objects)
        finally:
            del blocker
            self._updating_selection = False
        self._replot()

    def _unselect_all(self):
        self._updating_selection = True
        blocker = QSignalBlocker(self.listw)
        try:
            for i in range(self.listw.count()):
                self.listw.item(i).setSelected(False)
        finally:
            del blocker
            self._updating_selection = False
        self._replot()

    def _enforce_selection_limit(self):
        if self._updating_selection:
            return
        selected = [self.listw.item(i) for i in range(self.listw.count())
                    if self.listw.item(i).isSelected()]
        if len(selected) <= self.max_selected_objects:
            return
        self._updating_selection = True
        try:
            for it in selected[self.max_selected_objects:]:
                it.setSelected(False)
        finally:
            self._updating_selection = False
        QMessageBox.information(
            self, "Selection limit",
            f"You can display at most {self.max_selected_objects} objects at a time.")

class FinderInspectorDialog(QDialog):
    @staticmethod
    def _fmt_mag(value, band: str | None = None) -> str | None:
        if value is None:
            return None
        s = str(value).strip()
        if s.lower() in ("", "none", "nan", "masked", "--", "—"):
            return None
        band_s = str(band or "mag").strip() or "mag"
        try:
            fv = float(s)
            return f"{band_s}={fv:.2f}" if np.isfinite(fv) else None
        except Exception:
            return f"{band_s}={s}"

    @staticmethod
    def _fmt_vmag(v) -> str | None:
        return FinderInspectorDialog._fmt_mag(v, "V")

    @staticmethod
    def _fmt_sep(sep) -> str | None:
        if sep is None:
            return None
        try:
            fs = float(sep)
            if not np.isfinite(fs):
                return None
        except Exception:
            return None
        base = f"{fs:.1f}\""
        if fs >= 60.0:
            base += f" = {fs/60.0:.2f}′"
        return f"[{base}]"

    def _lw(self) -> float:
        return float(self.lw_spin.value())

    def _remove_artists(self, artists):
        for a in (artists or []):
            try: a.remove()
            except Exception: pass

    def _rebuild_numbering(self):
        n = 0
        for i in range(self.list_ident.count()):
            it   = self.list_ident.item(i)
            meta = it.data(Qt.UserRole) or {}
            if not meta.get("is_marker", False):
                continue
            n += 1
            nt  = meta.get("num_text")
            if nt is not None:
                try: nt.set_text(str(n))
                except Exception: pass
            txt2 = re.sub(r"^\s*\d+\.\s*", "", it.text())
            it.setText(f"{n}. {txt2}")
            it.setToolTip(it.text())
            it.setSizeHint(self._sizehint_for_text(it.text()))
        self._label_counter = n
        if self.canvas:
            self.canvas.draw_idle()

    def _draw_numbered_marker(self, ax, x, y, number: int):
        created = []
        p = ax.plot([x], [y], marker="o", markersize=max(4, int(self._lw() * 3)), linestyle="")[0]
        self._ann_artists.append(p)
        created.append(p)
        num_txt = ax.text(x, y, f"{number}", fontsize=10, fontweight="bold",
                          ha="left", va="bottom",
                          bbox=dict(boxstyle="round,pad=0.18", alpha=0.65))
        self._marker_labels.append(num_txt)
        created.append(num_txt)
        return created, num_txt

    def _coord_from_click(self, ax, x: float, y: float) -> SkyCoord | None:
        wcs = getattr(ax, "_rho_wcs", None) or getattr(ax, "wcs", None)
        if wcs is None:
            return None
        try:
            c = wcs.pixel_to_world(float(x), float(y))
            if isinstance(c, SkyCoord) and np.isfinite(c.ra.deg) and np.isfinite(c.dec.deg):
                return c.icrs
        except Exception:
            pass
        return None

    def __init__(
        self,
        parent,
        initial_fig,
        which_fov:    int,
        raw_data:     np.ndarray | None = None,
        raw_wcs:      WCS | None        = None,
        raw_coord:    SkyCoord | None   = None,
        raw_survey:   str               = "",
        raw_fov:      int               = 90,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Finder Inspector (FOV{which_fov})")
        self.resize(1250, 780)

        self.parent_window = parent
        self.which_fov     = which_fov

        self._raw_data   = raw_data
        self._raw_wcs    = raw_wcs
        self._raw_coord  = raw_coord
        self._raw_survey = raw_survey
        self._raw_fov    = raw_fov

        self.mode         = None
        self._press_xy    = None
        self._rect_patch  = None
        self._circle_patch = None

        self._ann_artists  = []
        self._ann_texts    = []
        self._measure_markers = []
        self._measure_p1   = None
        self._measure_artist = None
        self._label_counter = 0
        self._pending_ident_item = None
        self._pending_ident_num  = None
        self._pending_ident_xy   = None
        self._marker_labels      = []
        self._free_line   = None
        self._free_xs     = []
        self._free_ys     = []
        self._id_workers  = set()

        self._roll_timer = QTimer(self)
        self._roll_timer.setSingleShot(True)
        self._roll_timer.setInterval(400)
        self._roll_timer.timeout.connect(self._apply_roll_from_cache)

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        left    = QWidget()
        self.left_l = QVBoxLayout(left)
        self.left_l.setContentsMargins(0, 0, 0, 0)
        self.left_l.setSpacing(10)

        top  = QWidget()
        tl   = QHBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(10)

        self.fov_spin = QSpinBox()
        self.fov_spin.setRange(1, 360)
        self.fov_spin.setValue(int(parent.in_fov1.value()) if which_fov == 1
                               else int(parent.in_fov2.value()))

        btn_apply_fov = QPushButton("Update FOV")
        style_toolbar_button(btn_apply_fov)
        btn_apply_fov.setIcon(std_icon(self, "SP_BrowserReload"))
        btn_apply_fov.clicked.connect(self._request_new_finder)

        self.lw_spin = QSpinBox()
        self.lw_spin.setRange(1, 12)
        self.lw_spin.setValue(2)

        self.roll_spin = QDoubleSpinBox()
        self.roll_spin.setRange(-360.0, 360.0)
        self.roll_spin.setDecimals(1)
        self.roll_spin.setSingleStep(1.0)
        self.roll_spin.setSuffix("°")
        self.roll_spin.setValue(float(parent.in_roll.value()) if hasattr(parent, "in_roll") else 0.0)
        self.roll_spin.valueChanged.connect(lambda _: self._roll_timer.start())

        self.btn_copy_plot = QPushButton("Copy Plot")
        style_toolbar_button(self.btn_copy_plot)
        self.btn_copy_plot.setIcon(std_icon(self, "SP_DialogSaveButton"))
        self.btn_copy_plot.clicked.connect(
            lambda: copy_figure_to_clipboard(self.canvas.figure, self,
                                             "Finder chart copied to clipboard."))

        tl.addWidget(QLabel("FOV (arcmin):")); tl.addWidget(self.fov_spin)
        tl.addWidget(btn_apply_fov)
        tl.addSpacing(20)
        tl.addWidget(QLabel("Roll:"));            tl.addWidget(self.roll_spin)
        tl.addSpacing(20)
        tl.addWidget(QLabel("Line thickness:")); tl.addWidget(self.lw_spin)
        tl.addStretch(1)
        tl.addWidget(self.btn_copy_plot)
        self.left_l.addWidget(top)

        tools_box = QGroupBox("Tools")
        tools_l   = QHBoxLayout(tools_box)
        tools_l.setContentsMargins(10, 8, 10, 8)
        tools_l.setSpacing(8)

        def _mk(label, mode_name):
            b = QPushButton(label)
            b.setCheckable(True)
            b.toggled.connect(lambda on: self._set_mode(mode_name if on else None))
            return b

        self.btn_rect     = _mk("Draw Rectangle",    "rect")
        self.btn_circle   = _mk("Draw Circle",        "circle")
        self.btn_free     = _mk("Free Draw",          "free")
        self.btn_guide    = _mk("Mark Guide Star",    "guide")
        self.btn_measure  = _mk("Measure Separation", "measure")
        self.btn_identify = _mk("Identify Star",      "identify")

        self.btn_clear_ann = QPushButton("Clear Annotations")
        self.btn_clear_ann.setIcon(std_icon(self, "SP_TrashIcon"))
        self.btn_clear_ann.clicked.connect(self._clear_annotations)

        for w in (self.btn_rect, self.btn_circle, self.btn_free,
                  self.btn_guide, self.btn_measure, self.btn_identify):
            tools_l.addWidget(w)
        tools_l.addStretch(1)
        tools_l.addWidget(self.btn_clear_ann)
        self.left_l.addWidget(tools_box)

        plotw = QWidget()
        self.plot_layout = QVBoxLayout(plotw)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(6)

        self.canvas   = None
        self.toolbar  = None
        self._cid_press = self._cid_move = self._cid_rel = self._cid_leave = None

        self.coord_readout = QLabel("Move the cursor over the finder chart to read RA and Dec")
        self.coord_readout.setAlignment(Qt.AlignCenter)
        self.coord_readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.coord_readout.setMinimumHeight(30)
        coord_font = QFont("Consolas", 10)
        self.coord_readout.setFont(coord_font)
        self.coord_readout.setStyleSheet(
            "QLabel { background:#11151b; color:#dbeafe; border:1px solid #354052; "
            "border-radius:5px; padding:5px 10px; }"
        )

        self._replace_plot(initial_fig)
        self.left_l.addWidget(plotw, 1)

        right   = QGroupBox("Identified / Labeled Objects")
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(10, 10, 10, 10)
        right_l.setSpacing(8)

        self.list_ident = QListWidget()
        self.list_ident.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_ident.setWordWrap(True)
        self.list_ident.setUniformItemSizes(False)
        try:
            self.list_ident.setTextElideMode(Qt.ElideNone)
            f = self.list_ident.font()
            f.setFamily("Consolas")
            self.list_ident.setFont(f)
        except Exception:
            pass
        right_l.addWidget(self.list_ident, 1)

        btns    = QWidget()
        btns_l  = QHBoxLayout(btns)
        btns_l.setContentsMargins(0, 0, 0, 0)
        btns_l.setSpacing(10)

        btn_rm  = QPushButton("Remove Selected")
        btn_rm.setIcon(std_icon(self, "SP_TrashIcon"))
        btn_rm.clicked.connect(self._remove_selected_labels)

        btn_clr = QPushButton("Clear List")
        btn_clr.setIcon(std_icon(self, "SP_DialogResetButton"))
        btn_clr.clicked.connect(self._clear_list)

        btns_l.addWidget(btn_rm); btns_l.addWidget(btn_clr)
        right_l.addWidget(btns)

        root.addWidget(left,  1)
        root.addWidget(right, 0)
        root.setStretch(0, 4)
        root.setStretch(1, 1)

        self._update_title_wcs_hint()

    def _sizehint_for_text(self, txt: str) -> QSize:
        doc = QTextDocument()
        doc.setDefaultFont(self.list_ident.font())
        doc.setPlainText(str(txt))
        vw = max(180, self.list_ident.viewport().width() - 18)
        doc.setTextWidth(vw)
        return QSize(vw, max(22, int(doc.size().height()) + 10))

    def _add_list_item(self, txt: str, meta: dict | None = None) -> QListWidgetItem:
        it = QListWidgetItem(str(txt))
        it.setData(Qt.UserRole, meta or {})
        it.setToolTip(str(txt))
        it.setSizeHint(self._sizehint_for_text(txt))
        self.list_ident.addItem(it)
        return it

    def _replace_plot(self, fig):
        if self.toolbar is not None:
            self.plot_layout.removeWidget(self.toolbar)
            self.toolbar.setParent(None)
        if self.canvas is not None:
            self.plot_layout.removeWidget(self.canvas)
            for cid in (self._cid_press, self._cid_move, self._cid_rel, self._cid_leave):
                if cid is not None:
                    try: self.canvas.mpl_disconnect(cid)
                    except Exception: pass
            try: core.plt.close(self.canvas.figure)
            except Exception: pass
            self.canvas.setParent(None)

        try: self.plot_layout.removeWidget(self.coord_readout)
        except Exception: pass

        self.canvas  = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.coord_readout)
        self.plot_layout.addWidget(self.canvas, 1)
        self.coord_readout.setText("Move the cursor over the finder chart to read RA and Dec")
        self.coord_readout.setToolTip("")

        self._cid_press = self.canvas.mpl_connect("button_press_event",    self._on_press)
        self._cid_move  = self.canvas.mpl_connect("motion_notify_event",   self._on_move)
        self._cid_rel   = self.canvas.mpl_connect("button_release_event",  self._on_release)
        self._cid_leave = self.canvas.mpl_connect("figure_leave_event",    self._on_figure_leave)

    def set_figure(self, fig, raw_data=None, raw_wcs=None, raw_coord=None,
                   raw_survey="", raw_fov=90):
        self._clear_annotations()
        if raw_data is not None:
            self._raw_data   = raw_data
            self._raw_wcs    = raw_wcs
            self._raw_coord  = raw_coord
            self._raw_survey = raw_survey
            self._raw_fov    = raw_fov
        self._replace_plot(fig)
        self._label_counter = 0
        self._update_title_wcs_hint()

    def _ax(self):
        if not self.canvas or not self.canvas.figure:
            return None
        axes = list(self.canvas.figure.axes)
        if not axes:
            return None
        for a in axes:
            w = getattr(a, "_rho_wcs", None) or getattr(a, "wcs", None)
            if w is None:
                continue
            try:
                if w.has_celestial:
                    return a
            except Exception:
                continue
        try:
            return max(axes, key=lambda a: a.get_position().width * a.get_position().height)
        except Exception:
            return axes[0]

    def _update_title_wcs_hint(self):
        ax   = self._ax()
        wcs  = getattr(ax, "_rho_wcs", None) if ax is not None else None
        roll = float(getattr(ax, "_rho_display_roll", 0.0)) if ax is not None else 0.0

        base = f"Finder Inspector (FOV{self.which_fov})"
        if wcs is not None:
            hint = f"  [WCS OK, roll={roll:.1f}°]" if abs(roll) > 1e-9 else "  [WCS OK]"
        else:
            hint = "  [No WCS — star ID disabled]"
        self.setWindowTitle(base + hint)

    def _set_mode(self, mode: str | None):
        self.mode      = mode
        self._press_xy = None
        self._rect_patch = self._circle_patch = None
        if mode != "measure":
            self._measure_p1 = None
        self._free_line = None
        self._free_xs   = []
        self._free_ys   = []

        mapping = {
            "rect":     self.btn_rect,
            "circle":   self.btn_circle,
            "free":     self.btn_free,
            "guide":    self.btn_guide,
            "measure":  self.btn_measure,
            "identify": self.btn_identify,
        }
        for m, btn in mapping.items():
            should = (mode == m)
            if btn.isChecked() != should:
                btn.blockSignals(True)
                btn.setChecked(should)
                btn.blockSignals(False)

    def _set_coord_readout_default(self):
        if hasattr(self, "coord_readout"):
            self.coord_readout.setText(
                "Move the cursor over the finder chart to read RA and Dec"
            )
            self.coord_readout.setToolTip("")

    def _update_coord_readout(self, event):
        if not hasattr(self, "coord_readout"):
            return
        ax = getattr(event, "inaxes", None)
        if (ax is None or self.canvas is None or
                ax.figure is not self.canvas.figure or
                event.xdata is None or event.ydata is None):
            self._set_coord_readout_default()
            return

        x, y = float(event.xdata), float(event.ydata)
        wcs = getattr(ax, "_rho_wcs", None) or getattr(ax, "wcs", None)
        text = core.format_finder_cursor(wcs, x, y, include_pixel=True)
        self.coord_readout.setText(text or f"Pixel x={x:.2f}, y={y:.2f}")

        c = self._coord_from_click(ax, x, y)
        if c is not None:
            self.coord_readout.setToolTip(
                f"ICRS decimal degrees:  RA={c.ra.deg:.8f}°,  Dec={c.dec.deg:+.8f}°"
            )
        else:
            self.coord_readout.setToolTip("")

    def _on_figure_leave(self, _event):
        self._set_coord_readout_default()

    def _on_press(self, event):
        ax = event.inaxes
        if ax is None or self.canvas is None or ax.figure is not self.canvas.figure:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)

        if self.mode == "rect":
            self._press_xy   = (x, y)
            self._rect_patch = Rectangle((x, y), 0, 0, fill=False, linewidth=self._lw())
            ax.add_patch(self._rect_patch)
            self.canvas.draw_idle()

        elif self.mode == "circle":
            from matplotlib.patches import Circle
            self._press_xy     = (x, y)
            self._circle_patch = Circle((x, y), radius=0.0, fill=False, linewidth=self._lw())
            ax.add_patch(self._circle_patch)
            self.canvas.draw_idle()

        elif self.mode == "free":
            self._free_xs   = [x]; self._free_ys = [y]
            self._free_line = ax.plot(self._free_xs, self._free_ys, "-", linewidth=self._lw())[0]
            self.canvas.draw_idle()

        elif self.mode == "guide":
            self._place_marker(ax, x, y, label="Guide")
            self._set_mode(None)

        elif self.mode == "measure":
            if self._measure_p1 is None:
                self._clear_measure_artist()
                self._clear_measure_points()
                self._measure_p1 = (x, y)
                p = ax.plot([x], [y], marker="o",
                            markersize=max(4, int(self._lw() * 3)), linestyle="")[0]
                t = ax.text(x, y, " P1", fontsize=10)
                self._measure_markers.extend([p, t])
                self.canvas.draw_idle()
            else:
                p = ax.plot([x], [y], marker="o",
                            markersize=max(4, int(self._lw() * 3)), linestyle="")[0]
                t = ax.text(x, y, " P2", fontsize=10)
                self._measure_markers.extend([p, t])
                self._draw_measurement(ax, self._measure_p1, (x, y))
                self._measure_p1 = None
                self.canvas.draw_idle()

        elif self.mode == "identify":
            self._identify_at_click(ax, x, y)

    def _on_move(self, event):
        self._update_coord_readout(event)
        ax = event.inaxes
        if ax is None or self.canvas is None or ax.figure is not self.canvas.figure:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)

        if self.mode == "rect" and self._press_xy and self._rect_patch:
            x0, y0 = self._press_xy
            self._rect_patch.set_width(x - x0)
            self._rect_patch.set_height(y - y0)
            self.canvas.draw_idle()

        elif self.mode == "circle" and self._press_xy and self._circle_patch:
            x0, y0 = self._press_xy
            self._circle_patch.set_radius(float(np.hypot(x - x0, y - y0)))
            self.canvas.draw_idle()

        elif self.mode == "free" and self._free_line is not None:
            self._free_xs.append(x); self._free_ys.append(y)
            self._free_line.set_data(self._free_xs, self._free_ys)
            self.canvas.draw_idle()

    def _on_release(self, event):
        if self.mode == "rect" and self._rect_patch is not None:
            self._ann_artists.append(self._rect_patch)
            self._rect_patch = None; self._press_xy = None
            self._set_mode(None)

        elif self.mode == "circle" and self._circle_patch is not None:
            self._ann_artists.append(self._circle_patch)
            self._circle_patch = None; self._press_xy = None
            self._set_mode(None)

        elif self.mode == "free" and self._free_line is not None:
            self._ann_artists.append(self._free_line)
            self._free_line = None; self._free_xs = []; self._free_ys = []
            self._set_mode(None)

    def _place_marker(self, ax, x, y, label=None, add_to_list=True, meta=None):
        if ax is None:
            return

        created_artists = []

        marker_num = None
        if add_to_list:
            self._label_counter += 1
            marker_num = self._label_counter

        p = ax.plot([x], [y], marker="o",
                    markersize=max(4, int(self._lw() * 3)), linestyle="")[0]
        self._ann_artists.append(p)
        created_artists.append(p)

        label_artist = None
        if label:
            label_artist = ax.text(x, y, f" {label}", fontsize=10)
            self._ann_texts.append(label_artist)
            created_artists.append(label_artist)

        num_artist = None
        if marker_num is not None:
            num_artist = ax.text(x, y, f"{marker_num}", fontsize=10,
                                 fontweight="bold", ha="left", va="bottom",
                                 bbox=dict(boxstyle="round,pad=0.18", alpha=0.65))
            self._marker_labels.append(num_artist)
            created_artists.append(num_artist)

        self.canvas.draw_idle()

        if add_to_list:
            base_txt  = str(label) if label else "Marker"
            list_txt  = f"{marker_num}. {base_txt}" if marker_num is not None else base_txt
            it        = self._add_list_item(list_txt, meta={})
            it_meta   = {
                "is_marker": True,
                "artists":   created_artists,
                "num_text":  num_artist,
                "raw_meta":  meta or {},
            }
            it.setData(Qt.UserRole, it_meta)
            it.setToolTip(list_txt)
            it.setSizeHint(self._sizehint_for_text(list_txt))

    def _clear_measure_points(self):
        for a in list(self._measure_markers):
            try: a.remove()
            except Exception: pass
        self._measure_markers.clear()

    def _clear_measure_artist(self):
        if self._measure_artist is None:
            return
        artists = self._measure_artist if isinstance(self._measure_artist, tuple) \
                  else (self._measure_artist,)
        for a in artists:
            try: a.remove()
            except Exception: pass
        self._measure_artist = None

    def _draw_measurement(self, ax, p1, p2):
        if ax is None:
            return
        self._clear_measure_artist()
        x1, y1 = p1
        x2, y2 = p2
        dpix   = float(np.hypot(x2 - x1, y2 - y1))

        sky_txt = ""
        c1 = self._coord_from_click(ax, x1, y1)
        c2 = self._coord_from_click(ax, x2, y2)
        if c1 is not None and c2 is not None:
            try:
                sep    = c1.separation(c2)
                sep_as = sep.to(u.arcsec).value
                sep_am = sep.to(u.arcmin).value
                pa     = c1.position_angle(c2).to(u.deg).value
                sky_txt = f"{sep_as:.1f}\" ({sep_am:.2f}′), PA={pa:.1f}°"
            except Exception:
                pass

        line = ax.plot([x1, x2], [y1, y2], "-", linewidth=self._lw())[0]
        txt  = ax.text((x1 + x2) / 2.0, (y1 + y2) / 2.0,
                       f"{dpix:.1f}px\n{sky_txt}", fontsize=10,
                       ha="left", va="bottom",
                       bbox=dict(boxstyle="round,pad=0.25", alpha=0.6))
        self._measure_artist = (line, txt)
        self.canvas.draw_idle()

    def _clear_annotations(self):
        ax = self._ax()
        if ax is None:
            return
        self._clear_measure_artist()
        self._clear_measure_points()
        for a in list(self._ann_artists):
            try: a.remove()
            except Exception: pass
        for t in list(self._ann_texts) + list(self._marker_labels):
            try: t.remove()
            except Exception: pass
        self._ann_artists.clear()
        self._ann_texts.clear()
        self._marker_labels.clear()
        self._label_counter  = 0
        self._measure_p1     = None
        self._pending_ident_item = None
        self._pending_ident_num  = None
        self._pending_ident_xy   = None
        if self.canvas:
            self.canvas.draw_idle()

    def _remove_selected_labels(self):
        for it in list(self.list_ident.selectedItems()):
            meta = it.data(Qt.UserRole) or {}
            if meta.get("is_marker", False):
                self._remove_artists(meta.get("artists", []))
            self.list_ident.takeItem(self.list_ident.row(it))
        self._rebuild_numbering()

    def _clear_list(self):
        for i in range(self.list_ident.count()):
            it   = self.list_ident.item(i)
            meta = it.data(Qt.UserRole) or {}
            if meta.get("is_marker", False):
                self._remove_artists(meta.get("artists", []))
        self.list_ident.clear()
        self._label_counter = 0
        self._marker_labels.clear()
        self._ann_artists.clear()
        self._ann_texts.clear()
        if self.canvas:
            self.canvas.draw_idle()

    def _identify_at_click(self, ax, x, y):
        coord = self._coord_from_click(ax, x, y)
        if coord is None:
            QMessageBox.information(
                self, "Identify",
                "Could not convert click to sky coordinates.\n"
                "Is there a valid WCS attached to this chart?",
            )
            return

        self._label_counter += 1
        num = self._label_counter
        pending_text = f"{num}. Finding the brightest nearby catalogue source…"
        item = QListWidgetItem(pending_text)
        item.setToolTip(pending_text)
        item.setSizeHint(self._sizehint_for_text(pending_text))
        item.setData(
            Qt.UserRole,
            {"is_marker": True, "artists": [], "num_text": None, "raw_meta": {}},
        )
        self.list_ident.addItem(item)

        ax0, click_x, click_y = ax, float(x), float(y)

        try:
            self.parent_window.statusBar().showMessage(
                "Searching Gaia, Tycho-2, and SIMBAD for the brightest nearby source…"
            )
        except Exception:
            pass

        worker = StarIdWorker(coord, radius_arcsec=30.0)
        self._id_workers.add(worker)

        def _cleanup():
            self._id_workers.discard(worker)
            worker.deleteLater()

        def _item_still_exists() -> bool:
            try:
                return self.list_ident.row(item) >= 0
            except Exception:
                return False

        def _best_marker_xy(meta_payload: dict) -> tuple[float, float]:
            best_coord = meta_payload.get("coord") if isinstance(meta_payload, dict) else None
            wcs = getattr(ax0, "_rho_wcs", None) or getattr(ax0, "wcs", None)
            if best_coord is not None and wcs is not None:
                try:
                    mx, my = wcs.world_to_pixel(best_coord)
                    mx, my = float(mx), float(my)
                    if np.isfinite(mx) and np.isfinite(my):
                        return mx, my
                except Exception:
                    pass
            return click_x, click_y

        def _finish(text_block: str, meta_payload: dict):
            if not _item_still_exists():
                return
            mx, my = _best_marker_xy(meta_payload)
            artists, num_artist = self._draw_numbered_marker(ax0, mx, my, num)
            meta = item.data(Qt.UserRole) or {}
            meta.update({
                "artists": artists,
                "num_text": num_artist,
                "raw_meta": meta_payload,
            })
            item.setData(Qt.UserRole, meta)
            full_text = f"{num}. {text_block}"
            item.setText(full_text)
            item.setToolTip(full_text)
            item.setSizeHint(self._sizehint_for_text(full_text))
            if self.canvas:
                self.canvas.draw_idle()

        def _on_result(res):
            try:
                if not res.get("ok", False):
                    ra_s = coord.ra.to_string(
                        unit=u.hour, sep=":", precision=3, pad=True,
                    )
                    dec_s = coord.dec.to_string(
                        unit=u.deg, sep=":", precision=2,
                        alwayssign=True, pad=True,
                    )
                    _finish(
                        f"Unknown — no catalogue match within 30″\n"
                        f"RA {ra_s}    Dec {dec_s}",
                        {"ok": False, "coord": coord},
                    )
                    try:
                        self.parent_window.statusBar().showMessage("Identify: no match.")
                    except Exception:
                        pass
                    return

                main_id = str(res.get("main_id", "Unknown")).strip() or "Unknown"
                mag_txt = self._fmt_mag(
                    res.get("mag", res.get("vmag")),
                    res.get("mag_band", "V"),
                )
                sep_txt = self._fmt_sep(res.get("sep_arcsec"))
                catalog = str(res.get("catalog", "")).strip()
                cbest = res.get("coord")

                extras = [value for value in (mag_txt, sep_txt) if value]
                line1 = main_id
                if extras:
                    line1 += "    " + "    ".join(extras)
                if catalog and catalog.lower() not in main_id.lower():
                    line1 += f"    ({catalog})"

                lines = [line1]
                if cbest is not None:
                    ra_s = cbest.ra.to_string(
                        unit=u.hour, sep=":", precision=3, pad=True,
                    )
                    dec_s = cbest.dec.to_string(
                        unit=u.deg, sep=":", precision=2,
                        alwayssign=True, pad=True,
                    )
                    lines.append(f"RA {ra_s}    Dec {dec_s}")

                n_candidates = int(res.get("candidate_count", 1) or 1)
                selection_radius = res.get("selection_radius_arcsec")
                if n_candidates > 1:
                    try:
                        radius_text = f" within {float(selection_radius):.0f}″"
                    except Exception:
                        radius_text = ""
                    lines.append(
                        f"Brightest match{radius_text}; "
                        f"{n_candidates} distinct nearby catalogue sources considered"
                    )

                _finish("\n".join(lines), res)
                try:
                    self.parent_window.statusBar().showMessage(
                        f"Identify complete: {main_id}"
                    )
                except Exception:
                    pass
            except Exception as exc:
                _on_failed(str(exc))

        def _on_failed(message: str):
            try:
                ra_s = coord.ra.to_string(unit=u.hour, sep=":", precision=3, pad=True)
                dec_s = coord.dec.to_string(
                    unit=u.deg, sep=":", precision=2, alwayssign=True, pad=True,
                )
                _finish(
                    f"Catalogue query failed\nRA {ra_s}    Dec {dec_s}",
                    {"ok": False, "coord": coord, "error": str(message)},
                )
            except Exception:
                _finish(
                    "Catalogue query failed",
                    {"ok": False, "coord": coord, "error": str(message)},
                )
            QMessageBox.warning(self, "Query failed", str(message))
            try:
                self.parent_window.statusBar().showMessage("Identify query failed.")
            except Exception:
                pass

        worker.finished.connect(_on_result)
        worker.finished.connect(lambda _: _cleanup())
        worker.failed.connect(_on_failed)
        worker.failed.connect(lambda _: _cleanup())
        worker.start()

    def _apply_roll_from_cache(self):
        if self._raw_data is None or self._raw_wcs is None or self._raw_coord is None:
            self._request_new_finder()
            return

        new_roll = float(self.roll_spin.value())
        row      = getattr(self.parent_window, "_selected_row", None)
        name     = row.name if row else ""

        try:
            fig = core.render_finder_figure_from_data(
                self._raw_coord, name,
                self._raw_data, self._raw_wcs,
                self._raw_fov, self._raw_survey,
                roll_deg=new_roll,
            )
            self._clear_annotations()
            self._replace_plot(fig)
            self._update_title_wcs_hint()
            try: self.parent_window.in_roll.blockSignals(True)
            except Exception: pass
            try: self.parent_window.in_roll.setValue(new_roll)
            except Exception: pass
            try: self.parent_window.in_roll.blockSignals(False)
            except Exception: pass
        except Exception as e:
            QMessageBox.warning(self, "Roll update failed", str(e))

    def _request_new_finder(self):
        if self.parent_window._selected_row is None:
            QMessageBox.information(self, "No target", "Select a target first.")
            return
        new_fov  = int(self.fov_spin.value())
        new_roll = float(self.roll_spin.value())

        try: self.parent_window.in_roll.setValue(new_roll)
        except Exception: pass

        if self.which_fov == 1:
            try: self.parent_window.in_fov1.setValue(new_fov)
            except Exception: pass
        else:
            try: self.parent_window.in_fov2.setValue(new_fov)
            except Exception: pass

        self.parent_window._open_finder_dialog_request = self
        self.parent_window.on_row_selected()

class ExposureCalculatorDialog(QDialog):
    RHO_DEFAULTS = {
        "aperture_m": 0.356,
        "obstruction_pct": 0.0,
        "throughput_pct": 40.0,
        "physical_pixel_scale_arcsec": 0.54,
        "read_noise_e": 9.3,
        "gain_e_per_adu": 0.37,
        "dark_current_e_s_physical_pix": 0.15,
        "full_well_e": 25500.0,
        "well_fraction_pct": 80.0,
        "minimum_exposure_s": 0.10,
    }

    GOAL_PRESETS = {
        "finder": ("Finder / framing", 20.0),
        "general": ("General imaging", 50.0),
        "photometry": ("Standard photometry", 100.0),
        "precision": ("High-precision photometry", 200.0),
    }

    CONDITION_PRESETS = {
        "good": ("Good", 7.0, 1.15),
        "typical": ("Typical", 10.3, 1.30),
        "poor": ("Poor", 14.0, 1.70),
    }

    TECHNICAL_COLUMNS = (1, 2, 3, 4)

    def __init__(self, parent=None, target_name: str = "", target_vmag=None):
        super().__init__(parent)
        self.parent_window = parent
        self._results = []
        self.setWindowTitle("RHO Exposure Time Calculator")
        self.setMinimumSize(720, 540)
        self._size_for_available_screen(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        intro = QLabel(
            "Choose a catalogue magnitude in any supported Johnson/Cousins, "
            "Gaia, Sloan, or narrowband input band. Simple mode uses the RHO "
            "camera profile and observing presets; Advanced mode exposes the "
            "underlying assumptions."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        target_box = QGroupBox("Target and Camera Setup")
        target_grid = QGridLayout(target_box)
        target_grid.setContentsMargins(10, 9, 10, 9)
        target_grid.setHorizontalSpacing(8)
        target_grid.setVerticalSpacing(7)

        self.etc_target_name = QLineEdit(target_name or "")
        self.etc_ref_mag = self._dspin(-10.0, 40.0, 12.0, 3, " mag")
        self.etc_ref_band = QComboBox()
        self._populate_reference_bands()
        self._set_reference_band("Johnson V")
        self.etc_ref_band.setToolTip(
            "Johnson/Cousins and Gaia DR3 catalogue magnitudes are interpreted "
            "in their Vega systems. Sloan and listed narrowband magnitudes are "
            "interpreted as AB magnitudes."
        )

        self.etc_binning = QComboBox()
        for b in (1, 2, 3):
            self.etc_binning.addItem(f"{b} x {b}", b)
        self.etc_binning.setCurrentIndex(2)

        self.etc_use_peak_counts = QCheckBox("Use")
        self.etc_use_peak_counts.setChecked(True)
        self.etc_use_peak_counts.setToolTip(
            "Use a preferred peak-pixel level for each subexposure. The value "
            "is detector ADU above bias; the S/N goal still determines how "
            "many frames are required."
        )
        self.etc_peak_counts = self._dspin(100.0, 65535.0, 40000.0, 0, " ADU")
        self.etc_peak_counts.setSingleStep(1000.0)
        self.etc_peak_counts.setToolTip(
            "Preferred peak-pixel counts per subexposure, above bias. "
            "40,000 ADU is a conservative RHO starting point."
        )
        self.etc_use_peak_counts.toggled.connect(self._sync_peak_counts_control)

        peak_counts_widget = QWidget()
        peak_counts_layout = QHBoxLayout(peak_counts_widget)
        peak_counts_layout.setContentsMargins(0, 0, 0, 0)
        peak_counts_layout.setSpacing(8)
        peak_counts_layout.addWidget(self.etc_use_peak_counts)
        peak_counts_layout.addWidget(self.etc_peak_counts)
        peak_counts_layout.addStretch(1)

        self.etc_target_selector = QComboBox()
        self.etc_target_selector.setMinimumContentsLength(28)
        self.etc_target_selector.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.etc_target_selector.setToolTip(
            "Choose any target currently listed in the observing planner. "
            "The target name and available catalogue V magnitude are copied "
            "into the calculator automatically."
        )
        self.etc_target_selector.currentIndexChanged.connect(
            self._on_planner_target_changed
        )

        target_grid.addWidget(QLabel("Planner target:"), 0, 0)
        target_grid.addWidget(self.etc_target_selector, 0, 1, 1, 5)
        target_grid.addWidget(QLabel("Target name:"), 1, 0)
        target_grid.addWidget(self.etc_target_name, 1, 1, 1, 5)
        target_grid.addWidget(QLabel("Magnitude:"), 2, 0)
        target_grid.addWidget(self.etc_ref_mag, 2, 1)
        target_grid.addWidget(QLabel("Input band:"), 2, 2)
        target_grid.addWidget(self.etc_ref_band, 2, 3)
        target_grid.addWidget(QLabel("Binning:"), 2, 4)
        target_grid.addWidget(self.etc_binning, 2, 5)
        target_grid.addWidget(QLabel("Peak-count target:"), 3, 0)
        target_grid.addWidget(peak_counts_widget, 3, 1, 1, 5)
        target_grid.setColumnStretch(1, 1)
        target_grid.setColumnStretch(3, 2)
        target_grid.setColumnStretch(5, 1)
        root.addWidget(target_box)

        self.etc_splitter = QSplitter(Qt.Vertical)
        self.etc_splitter.setChildrenCollapsible(False)
        root.addWidget(self.etc_splitter, 1)

        self.etc_tabs = QTabWidget()
        self.etc_tabs.addTab(self._build_simple_page(), "Simple")
        self.etc_tabs.addTab(self._build_advanced_page(), "Advanced")
        self.etc_splitter.addWidget(self.etc_tabs)

        result_box = QGroupBox("Estimated Exposure Plan")
        result_l = QVBoxLayout(result_box)
        result_l.setContentsMargins(8, 8, 8, 8)
        result_l.setSpacing(6)

        result_tools = QHBoxLayout()
        self.etc_summary = QLabel("")
        self.etc_summary.setWordWrap(True)
        self.etc_show_technical = QCheckBox("Show technical columns")
        self.etc_show_technical.toggled.connect(self._set_technical_columns)
        result_tools.addWidget(self.etc_summary, 1)
        result_tools.addWidget(self.etc_show_technical)
        result_l.addLayout(result_tools)

        self.etc_table = QTableWidget(0, 10)
        self.etc_table.setHorizontalHeaderLabels([
            "Filter", "Bandpass", "Est. mag (AB)", "Source rate", "Sky rate",
            "Total time", "Suggested sequence", "Peak counts",
            "Saturation limit", "Notes",
        ])
        self.etc_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.etc_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.etc_table.setAlternatingRowColors(True)
        self.etc_table.setWordWrap(True)
        self.etc_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        hdr = self.etc_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for col in (1, 2, 3, 4, 5, 6, 7, 8):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(9, QHeaderView.Stretch)
        result_l.addWidget(self.etc_table, 1)
        self.etc_splitter.addWidget(result_box)
        self.etc_splitter.setStretchFactor(0, 0)
        self.etc_splitter.setStretchFactor(1, 1)
        self.etc_splitter.setSizes([275, 430])

        buttons = QHBoxLayout()
        self.etc_btn_calculate = QPushButton("Calculate Exposure Plan")
        self.etc_btn_calculate.setIcon(std_icon(self, "SP_MediaPlay"))
        self.etc_btn_calculate.clicked.connect(self.calculate)

        self.etc_btn_copy = QPushButton("Copy Results")
        self.etc_btn_copy.setIcon(std_icon(self, "SP_DialogSaveButton"))
        self.etc_btn_copy.clicked.connect(self.copy_results)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)

        buttons.addWidget(self.etc_btn_calculate)
        buttons.addWidget(self.etc_btn_copy)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        root.addLayout(buttons)

        self.refresh_planner_targets(
            preferred_row=getattr(self.parent_window, "_selected_row", None)
        )
        self.set_target(target_name, target_vmag)
        self._sync_profile_controls()
        self._sync_spectrum_controls()
        self._sync_peak_counts_control()
        self._set_technical_columns(False)
        self.calculate()

    def _size_for_available_screen(self, parent):
        try:
            screen = parent.screen() if parent is not None else QGuiApplication.primaryScreen()
            geom = screen.availableGeometry() if screen is not None else None
            if geom is not None:
                width = min(1180, max(720, int(geom.width() * 0.90)))
                height = min(840, max(540, int(geom.height() * 0.88)))
                self.resize(width, height)
                return
        except Exception:
            pass
        self.resize(1080, 760)

    def _populate_reference_bands(self):
        first_group = True
        for _family, band_specs in core.reference_magnitude_band_groups():
            if not first_group:
                self.etc_ref_band.insertSeparator(self.etc_ref_band.count())
            first_group = False
            for spec in band_specs:
                self.etc_ref_band.addItem(spec.display_name, spec.key)

    @staticmethod
    def _scrollable(widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(widget)
        return scroll

    def _build_simple_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        box = QGroupBox("Quick Observing Choices")
        form = QFormLayout(box)
        form.setVerticalSpacing(8)

        self.etc_source_type = QComboBox()
        self.etc_source_type.addItem("Star / continuum object", "star")
        self.etc_source_type.addItem("Emission nebula", "nebula")

        self.etc_goal = QComboBox()
        for key, (label, _snr) in self.GOAL_PRESETS.items():
            self.etc_goal.addItem(label, key)
        self.etc_goal.setCurrentIndex(1)

        self.etc_conditions = QComboBox()
        for key, (label, _fwhm, _airmass) in self.CONDITION_PRESETS.items():
            self.etc_conditions.addItem(label, key)
        self.etc_conditions.setCurrentIndex(1)

        form.addRow("Source type:", self.etc_source_type)
        form.addRow("Imaging goal:", self.etc_goal)
        form.addRow("Conditions:", self.etc_conditions)

        note = QLabel(
            "Simple mode automatically supplies the RHO telescope/camera values, "
            "an S/N goal, airmass, measured image width, and safe broadband / "
            "narrowband subexposure limits. The peak-count target sets the "
            "preferred level in each frame, while S/N determines how many frames "
            "are required. The selected input band remains fully configurable."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#aeb7c4;")
        form.addRow(note)
        layout.addWidget(box)
        layout.addStretch(1)
        return self._scrollable(page)

    def _build_advanced_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        inst_box = QGroupBox("Instrument Profile")
        inst_form = QFormLayout(inst_box)
        inst_form.setVerticalSpacing(7)

        self.etc_profile = QComboBox()
        self.etc_profile.addItem("RHO 14-inch + STF-8300M", "rho")
        self.etc_profile.addItem("Custom telescope / camera", "custom")
        self.etc_profile.currentIndexChanged.connect(self._sync_profile_controls)

        d = self.RHO_DEFAULTS
        self.etc_aperture = self._dspin(0.05, 20.0, d["aperture_m"], 3, " m")
        self.etc_obstruction = self._dspin(0.0, 90.0, d["obstruction_pct"], 1, " %")
        self.etc_throughput = self._dspin(0.1, 100.0, d["throughput_pct"], 1, " %")
        self.etc_physical_pixel_scale = self._dspin(
            0.01, 10.0, d["physical_pixel_scale_arcsec"], 3, " arcsec/physical pix")
        self.etc_read_noise = self._dspin(0.0, 100.0, d["read_noise_e"], 2, " e-/read")
        self.etc_gain = self._dspin(0.001, 100.0, d["gain_e_per_adu"], 3, " e-/ADU")
        self.etc_dark = self._dspin(
            0.0, 100.0, d["dark_current_e_s_physical_pix"], 4, " e-/s/physical pix")
        self.etc_full_well = self._dspin(1000.0, 1.0e7, d["full_well_e"], 0, " e-")
        self.etc_well_fraction = self._dspin(
            10.0, 100.0, d["well_fraction_pct"], 1, " %")

        inst_form.addRow("Profile:", self.etc_profile)
        inst_form.addRow("Clear aperture:", self.etc_aperture)
        inst_form.addRow("Central obstruction (diameter):", self.etc_obstruction)
        inst_form.addRow("Base system efficiency:", self.etc_throughput)
        inst_form.addRow("1x1 plate scale:", self.etc_physical_pixel_scale)
        inst_form.addRow("Read noise:", self.etc_read_noise)
        inst_form.addRow("Camera gain:", self.etc_gain)
        inst_form.addRow("Dark current:", self.etc_dark)
        inst_form.addRow("Full well:", self.etc_full_well)
        inst_form.addRow("Use well up to:", self.etc_well_fraction)
        layout.addWidget(inst_box)

        obs_box = QGroupBox("Exact Observing Assumptions")
        obs_form = QFormLayout(obs_box)
        obs_form.setVerticalSpacing(7)

        self.etc_snr = self._dspin(1.0, 10000.0, 100.0, 1)
        self.etc_airmass = self._dspin(1.0, 5.0, 1.30, 2)
        self.etc_seeing = self._dspin(0.2, 60.0, 10.3, 2, " arcsec")
        self.etc_ap_radius = self._dspin(0.5, 5.0, 1.5, 2, " x FWHM")
        self.etc_max_broadband_sub = self._dspin(0.01, 86400.0, 300.0, 2, " s")
        self.etc_max_narrowband_sub = self._dspin(0.01, 86400.0, 900.0, 2, " s")
        self.etc_min_exp = self._dspin(
            0.001, 60.0, d["minimum_exposure_s"], 3, " s")

        obs_form.addRow("Desired S/N:", self.etc_snr)
        obs_form.addRow("Airmass:", self.etc_airmass)
        obs_form.addRow("Measured stellar FWHM:", self.etc_seeing)
        obs_form.addRow("Photometry aperture radius:", self.etc_ap_radius)
        obs_form.addRow("Maximum broadband subexposure:", self.etc_max_broadband_sub)
        obs_form.addRow("Maximum narrowband subexposure:", self.etc_max_narrowband_sub)
        obs_form.addRow("Minimum practical exposure:", self.etc_min_exp)
        layout.addWidget(obs_box)

        source_box = QGroupBox("Spectral and Emission-Line Assumptions")
        source_form = QFormLayout(source_box)
        source_form.setVerticalSpacing(7)

        self.etc_spectrum = QComboBox()
        self.etc_spectrum.addItem("Stellar blackbody color", "blackbody")
        self.etc_spectrum.addItem("Flat spectrum (constant f_nu)", "flat_fnu")
        self.etc_spectrum.currentIndexChanged.connect(self._sync_spectrum_controls)
        self.etc_temperature = self._dspin(1000.0, 50000.0, 5800.0, 0, " K")

        source_form.addRow("Continuum model:", self.etc_spectrum)
        source_form.addRow("Effective temperature:", self.etc_temperature)

        line_note = QLabel(
            "Optional integrated line flux in the measurement aperture, in "
            "10^-14 erg s^-1 cm^-2. Leave zero for a continuum-only estimate."
        )
        line_note.setWordWrap(True)
        source_form.addRow(line_note)

        self.etc_line_flux = {}
        for filt in ("H-alpha", "H-beta", "OIII", "SII"):
            spin = self._dspin(0.0, 1.0e9, 0.0, 4, " x10^-14")
            self.etc_line_flux[filt] = spin
            source_form.addRow(f"{filt} line flux:", spin)
        layout.addWidget(source_box)
        layout.addStretch(1)
        return self._scrollable(page)

    @staticmethod
    def _dspin(lo, hi, value, decimals=2, suffix="") -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(float(lo), float(hi))
        spin.setDecimals(int(decimals))
        spin.setValue(float(value))
        if suffix:
            spin.setSuffix(str(suffix))
        spin.setKeyboardTracking(False)
        return spin

    @staticmethod
    def _finite_float(value):
        try:
            x = float(value)
            return x if np.isfinite(x) else None
        except Exception:
            return None

    @staticmethod
    def _fmt_seconds(value: float) -> str:
        try:
            x = float(value)
        except Exception:
            return "—"
        if not np.isfinite(x):
            return "—"
        if x < 0.01:
            return f"{x:.3e} s"
        if x < 10.0:
            return f"{x:.3f} s"
        if x < 120.0:
            return f"{x:.1f} s"
        if x < 7200.0:
            return f"{x/60.0:.1f} min"
        return f"{x/3600.0:.2f} hr"

    @staticmethod
    def _fmt_rate(value: float) -> str:
        try:
            x = float(value)
        except Exception:
            return "—"
        if not np.isfinite(x):
            return "—"
        if abs(x) >= 1.0e4 or (0.0 < abs(x) < 0.01):
            return f"{x:.3e} e-/s"
        return f"{x:.2f} e-/s"

    @staticmethod
    def _fmt_counts(value: float) -> str:
        try:
            x = float(value)
        except Exception:
            return "—"
        if not np.isfinite(x):
            return "—"
        return f"{x:,.0f} ADU"

    def _set_reference_band(self, band_name: str):
        try:
            canonical = core.get_reference_magnitude_band(band_name).key
        except Exception:
            canonical = "Johnson V"
        idx = self.etc_ref_band.findData(canonical)
        if idx >= 0:
            self.etc_ref_band.setCurrentIndex(idx)

    def _sync_profile_controls(self):
        if not hasattr(self, "etc_profile"):
            return
        is_custom = self.etc_profile.currentData() == "custom"
        fields = (
            self.etc_aperture, self.etc_obstruction, self.etc_throughput,
            self.etc_physical_pixel_scale, self.etc_read_noise, self.etc_gain,
            self.etc_dark, self.etc_full_well, self.etc_well_fraction,
        )
        if not is_custom:
            d = self.RHO_DEFAULTS
            values = (
                d["aperture_m"], d["obstruction_pct"], d["throughput_pct"],
                d["physical_pixel_scale_arcsec"], d["read_noise_e"],
                d["gain_e_per_adu"], d["dark_current_e_s_physical_pix"],
                d["full_well_e"], d["well_fraction_pct"],
            )
            for field, value in zip(fields, values):
                field.setValue(float(value))
        for field in fields:
            field.setEnabled(is_custom)

    def _sync_spectrum_controls(self):
        if hasattr(self, "etc_temperature"):
            self.etc_temperature.setEnabled(
                self.etc_spectrum.currentData() == "blackbody")

    def _sync_peak_counts_control(self):
        if hasattr(self, "etc_peak_counts") and hasattr(self, "etc_use_peak_counts"):
            self.etc_peak_counts.setEnabled(self.etc_use_peak_counts.isChecked())

    def _set_technical_columns(self, shown: bool):
        for col in self.TECHNICAL_COLUMNS:
            self.etc_table.setColumnHidden(col, not bool(shown))

    @staticmethod
    def _plan_row_key(row) -> tuple[str, str, str]:
        if row is None:
            return ("", "", "")
        return (
            str(getattr(row, "name", "") or "").strip(),
            str(getattr(row, "ra", "") or "").strip(),
            str(getattr(row, "dec", "") or "").strip(),
        )

    def refresh_planner_targets(self, preferred_row=None):
        if not hasattr(self, "etc_target_selector"):
            return

        plan = list(getattr(self.parent_window, "plan", []) or [])
        current_key = self.etc_target_selector.currentData()
        preferred_key = self._plan_row_key(preferred_row) if preferred_row is not None else None
        desired_key = preferred_key or current_key

        blocker = QSignalBlocker(self.etc_target_selector)
        try:
            self.etc_target_selector.clear()
            if not plan:
                self.etc_target_selector.addItem("No targets currently in planner", None)
                self.etc_target_selector.setEnabled(False)
                return

            self.etc_target_selector.setEnabled(True)
            self.etc_target_selector.addItem("Choose a planner target…", None)

            selected_combo_index = 0
            for plan_index, row in enumerate(plan):
                name = str(getattr(row, "name", "") or "").strip()
                if not name:
                    name = "Unnamed Target"

                mag = self._finite_float(getattr(row, "vmag", None))
                mag_text = f"V={mag:.2f}" if mag is not None else "V unavailable"
                display = f"{plan_index + 1}. {name}  ({mag_text})"
                key = self._plan_row_key(row)
                self.etc_target_selector.addItem(display, key)
                self.etc_target_selector.setItemData(
                    self.etc_target_selector.count() - 1,
                    f"RA {getattr(row, 'ra', '—')}  |  Dec {getattr(row, 'dec', '—')}",
                    Qt.ToolTipRole,
                )
                if desired_key and key == desired_key:
                    selected_combo_index = self.etc_target_selector.count() - 1

            self.etc_target_selector.setCurrentIndex(selected_combo_index)
        finally:
            del blocker

    def _row_for_selector_key(self, key):
        if not key:
            return None
        for row in list(getattr(self.parent_window, "plan", []) or []):
            if self._plan_row_key(row) == tuple(key):
                return row
        return None

    def _select_planner_target(self, row=None, name: str = ""):
        if not hasattr(self, "etc_target_selector") or not self.etc_target_selector.isEnabled():
            return

        desired = self._plan_row_key(row) if row is not None else None
        if desired is None and name:
            for candidate in list(getattr(self.parent_window, "plan", []) or []):
                if str(getattr(candidate, "name", "") or "").strip() == str(name).strip():
                    desired = self._plan_row_key(candidate)
                    break
        if desired is None:
            return

        for i in range(self.etc_target_selector.count()):
            if self.etc_target_selector.itemData(i) == desired:
                blocker = QSignalBlocker(self.etc_target_selector)
                try:
                    self.etc_target_selector.setCurrentIndex(i)
                finally:
                    del blocker
                return

    def _on_planner_target_changed(self, _index: int):
        key = self.etc_target_selector.currentData()
        row = self._row_for_selector_key(key)
        if row is None:
            return

        self.set_target(row.name, row.vmag, planner_row=row)
        if self._finite_float(row.vmag) is None:
            msg = (
                f"{row.name} has no usable planner V magnitude. Enter a "
                "catalogue magnitude and choose its input band manually."
            )
            self.etc_target_selector.setToolTip(msg)
            if self.parent_window is not None:
                self.parent_window.statusBar().showMessage(msg, 5000)
        else:
            self.etc_target_selector.setToolTip(
                "Choose any target currently listed in the observing planner."
            )

    def set_target(self, name: str = "", vmag=None, planner_row=None):
        if name:
            self.etc_target_name.setText(str(name))
        mag = self._finite_float(vmag)
        if mag is not None:
            self.etc_ref_mag.setValue(mag)
            self._set_reference_band("Johnson V")
        self._select_planner_target(row=planner_row, name=name)

    def _simple_values(self):
        goal_key = str(self.etc_goal.currentData() or "general")
        cond_key = str(self.etc_conditions.currentData() or "typical")
        _goal_label, snr = self.GOAL_PRESETS[goal_key]
        _cond_label, fwhm, airmass = self.CONDITION_PRESETS[cond_key]
        source_type = str(self.etc_source_type.currentData() or "star")
        return source_type, float(snr), float(fwhm), float(airmass)

    def _build_calculation_inputs(self):
        binning = int(self.etc_binning.currentData() or 1)
        simple_mode = self.etc_tabs.currentIndex() == 0

        if simple_mode:
            source_type, snr, seeing, airmass = self._simple_values()
            d = self.RHO_DEFAULTS
            aperture_m = d["aperture_m"]
            obstruction = d["obstruction_pct"] / 100.0
            throughput = d["throughput_pct"] / 100.0
            physical_scale = d["physical_pixel_scale_arcsec"]
            read_noise = d["read_noise_e"]
            gain = d["gain_e_per_adu"]
            dark_physical = d["dark_current_e_s_physical_pix"]
            full_well = d["full_well_e"]
            well_fraction = d["well_fraction_pct"] / 100.0
            min_exp = d["minimum_exposure_s"]
            ap_radius = 1.5
            max_broadband = 300.0
            max_narrowband = 900.0
            spectrum_model = "blackbody" if source_type == "star" else "flat_fnu"
            temperature = 5800.0
            line_fluxes = {name: 0.0 for name in ("H-alpha", "H-beta", "OIII", "SII")}
            mode_desc = (
                f"Simple mode · {self.etc_goal.currentText()} · "
                f"{self.etc_conditions.currentText()} conditions · {binning}x{binning}"
            )
        else:
            aperture_m = float(self.etc_aperture.value())
            obstruction = float(self.etc_obstruction.value()) / 100.0
            throughput = float(self.etc_throughput.value()) / 100.0
            physical_scale = float(self.etc_physical_pixel_scale.value())
            read_noise = float(self.etc_read_noise.value())
            gain = float(self.etc_gain.value())
            dark_physical = float(self.etc_dark.value())
            full_well = float(self.etc_full_well.value())
            well_fraction = float(self.etc_well_fraction.value()) / 100.0
            min_exp = float(self.etc_min_exp.value())
            snr = float(self.etc_snr.value())
            airmass = float(self.etc_airmass.value())
            seeing = float(self.etc_seeing.value())
            ap_radius = float(self.etc_ap_radius.value())
            max_broadband = float(self.etc_max_broadband_sub.value())
            max_narrowband = float(self.etc_max_narrowband_sub.value())
            spectrum_model = str(self.etc_spectrum.currentData())
            temperature = float(self.etc_temperature.value())
            line_fluxes = {
                name: float(spin.value()) * 1.0e-14
                for name, spin in self.etc_line_flux.items()
            }
            mode_desc = f"Advanced mode · {binning}x{binning} binning"

        pixel_scale = physical_scale * binning
        dark_binned = dark_physical * (binning ** 2)

        cfg = core.ExposureCalculatorConfig(
            aperture_m=aperture_m,
            central_obstruction_fraction=obstruction,
            base_system_throughput=throughput,
            pixel_scale_arcsec=pixel_scale,
            read_noise_e=read_noise,
            gain_e_per_adu=gain,
            dark_current_e_s_pix=dark_binned,
            full_well_e=full_well,
            saturation_fraction=well_fraction,
            seeing_fwhm_arcsec=seeing,
            aperture_radius_fwhm=ap_radius,
            max_subexposure_s=max_broadband,
            max_narrowband_subexposure_s=max_narrowband,
            minimum_practical_exposure_s=min_exp,
            desired_peak_counts_adu=(
                float(self.etc_peak_counts.value())
                if self.etc_use_peak_counts.isChecked() else None
            ),
        )
        target = core.ExposureTarget(
            reference_mag_ab=float(self.etc_ref_mag.value()),
            reference_band=str(self.etc_ref_band.currentData() or "Johnson V"),
            spectrum_model=spectrum_model,
            effective_temperature_k=temperature,
            target_snr=snr,
            airmass=airmass,
            line_fluxes_erg_s_cm2=line_fluxes,
        )
        return cfg, target, mode_desc

    def calculate(self):
        try:
            cfg, target, mode_desc = self._build_calculation_inputs()
            self._results = core.calculate_exposure_times(cfg, target)
            band = core.get_reference_magnitude_band(target.reference_band)
            count_text = (
                f" · peak target: {cfg.desired_peak_counts_adu:,.0f} ADU"
                if cfg.desired_peak_counts_adu is not None
                else " · no peak-count target"
            )
            self.etc_summary.setText(
                f"{mode_desc} · input: {self.etc_ref_mag.value():.3f} mag in "
                f"{band.display_name} · S/N target: {target.target_snr:g}{count_text}"
            )
            self._populate_results()
        except Exception as exc:
            QMessageBox.critical(self, "Exposure calculation failed", str(exc))

    def _populate_results(self):
        self.etc_table.setRowCount(len(self._results))
        for r, result in enumerate(self._results):
            n = int(result["suggested_n"])
            sub = float(result["suggested_subexposure_s"])
            sequence = "—" if n <= 0 or not np.isfinite(sub) else f"{n} x {self._fmt_seconds(sub)}"
            values = [
                result["filter"],
                f'{result["central_nm"]:.1f} / {result["width_nm"]:.1f} nm',
                f'{result["estimated_ab_mag"]:.2f}',
                self._fmt_rate(result["source_rate_e_s"]),
                self._fmt_rate(result["sky_rate_e_s"]),
                self._fmt_seconds(result["total_time_s"]),
                sequence,
                self._fmt_counts(result.get("predicted_peak_counts_adu", float("nan"))),
                self._fmt_seconds(result["saturation_time_s"]),
                result["notes"] or "",
            ]
            for c, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if c == 7:
                    target_counts = result.get("desired_peak_counts_adu")
                    count_time = result.get("count_target_time_s")
                    try:
                        finite_count_time = np.isfinite(float(count_time))
                    except Exception:
                        finite_count_time = False
                    if target_counts is not None and finite_count_time:
                        achieved_snr = result.get("achieved_snr")
                        snr_text = ""
                        try:
                            if np.isfinite(float(achieved_snr)):
                                snr_text = f"; expected stacked S/N: {float(achieved_snr):.1f}"
                        except Exception:
                            pass
                        item.setToolTip(
                            f"Target: {float(target_counts):,.0f} ADU; "
                            f"nominal time to target: {self._fmt_seconds(float(count_time))}"
                            f"{snr_text}"
                        )
                elif c == 9 and value:
                    item.setToolTip(str(value))
                self.etc_table.setItem(r, c, item)
        self.etc_table.resizeRowsToContents()

    def copy_results(self):
        if not self._results:
            QMessageBox.information(self, "No results", "Calculate exposure times first.")
            return
        headers = [
            self.etc_table.horizontalHeaderItem(c).text()
            for c in range(self.etc_table.columnCount())
        ]
        lines = ["\t".join(headers)]
        for r in range(self.etc_table.rowCount()):
            lines.append("\t".join(
                self.etc_table.item(r, c).text() if self.etc_table.item(r, c) else ""
                for c in range(self.etc_table.columnCount())
            ))
        QGuiApplication.clipboard().setText("\n".join(lines))
        if self.parent_window is not None:
            self.parent_window.statusBar().showMessage("Exposure results copied.", 3000)

class MainWindow(QMainWindow):
    def __init__(self):
        self._alt_dialog       = None
        self._finder_dialog_fov1 = None
        self._finder_dialog_fov2 = None
        self._exposure_dialog    = None
        super().__init__()

        self.setWindowTitle("Observing Planner (Desktop)")
        self.resize(1650, 920)
        self.setMinimumSize(1500, 840)
        self.statusBar().showMessage("Ready")

        self.plan:            List[PlanRow] = []
        self._finder_workers: set           = set()
        self._plan_workers:   set           = set()
        self._finder_request_id             = 0

        self._last_coords: list = []
        self._last_names:  list = []
        self._selected_row      = None
        self._open_finder_dialog_request = None

        self._raw_finder: dict = {}

        self._roll_debounce = QTimer(self)
        self._roll_debounce.setSingleShot(True)
        self._roll_debounce.setInterval(400)
        self._roll_debounce.timeout.connect(self._apply_roll_from_cache)

        root   = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter)

        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())

        QTimer.singleShot(0, lambda: splitter.setSizes([330, 840, 520]))
        QTimer.singleShot(0, lambda: self._right_split.setSizes([420, 620]))

        self._bind_altitude_click()
        self._bind_finder_clicks()
        self.apply_date_location(initial=True)

    def _build_left_panel(self) -> QWidget:
        left   = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(6, 6, 6, 6)
        left_l.setSpacing(10)

        # Planning setup
        plan_box  = QGroupBox("Planning Setup")
        plan_form = QFormLayout(plan_box)
        plan_form.setContentsMargins(10, 8, 10, 8)
        plan_form.setVerticalSpacing(8)

        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDate(QDate.currentDate())

        site = core.get_site_config()
        self._default_site    = site
        self._default_min_alt = int(core.DEFAULT_MIN_ALT_DEG)
        self._default_max_alt = int(core.DEFAULT_MAX_ALT_DEG)
        self._default_fov1    = int(core.DEFAULT_FOV1_ARCMIN)
        self._default_fov2    = int(core.DEFAULT_FOV2_ARCMIN)
        self._default_roll    = 0.0
        self._default_survey  = "DSS"

        self.lat_spin    = QDoubleSpinBox(); self.lat_spin.setRange(-90.0, 90.0);     self.lat_spin.setDecimals(6);   self.lat_spin.setValue(site.lat)
        self.lon_spin    = QDoubleSpinBox(); self.lon_spin.setRange(-180.0, 180.0);   self.lon_spin.setDecimals(6);   self.lon_spin.setValue(site.lon)
        self.height_spin = QDoubleSpinBox(); self.height_spin.setRange(-500.0,9000.0);self.height_spin.setDecimals(1);self.height_spin.setValue(site.height_m)

        self.tz_combo = QComboBox()
        self.tz_combo.setEditable(True)
        self.tz_combo.addItems(["US/Eastern", "US/Central", "US/Mountain", "US/Pacific", "UTC"])
        self.tz_combo.setCurrentText(site.timezone)

        self.btn_apply = QPushButton("Apply date/location")
        self.btn_apply.setIcon(std_icon(self, "SP_DialogApplyButton"))
        self.btn_apply.clicked.connect(self.apply_date_location)

        self.btn_reset = QPushButton("Reset defaults")
        self.btn_reset.setIcon(std_icon(self, "SP_DialogResetButton"))
        self.btn_reset.clicked.connect(self.reset_defaults)

        plan_form.addRow("Date:",            self.date_edit)
        plan_form.addRow("Latitude (deg):",  self.lat_spin)
        plan_form.addRow("Longitude (deg):", self.lon_spin)
        plan_form.addRow("Elevation (m):",   self.height_spin)
        plan_form.addRow("Timezone (IANA):", self.tz_combo)
        btn_row = QWidget(); br_l = QHBoxLayout(btn_row); br_l.setContentsMargins(0,0,0,0); br_l.setSpacing(8)
        br_l.addWidget(self.btn_apply); br_l.addWidget(self.btn_reset)
        plan_form.addRow(btn_row)

        sky_box  = QGroupBox("Sky Conditions")
        sky_form = QFormLayout(sky_box)
        sky_form.setContentsMargins(10, 8, 10, 8)
        sky_form.setVerticalSpacing(8)

        self.lbl_sunset    = QLabel("—")
        self.lbl_moon_alt  = QLabel("—")
        self.lbl_moon_illum = QLabel("—")
        self.lbl_cloud_now  = QLabel("—")
        self.lbl_cloud_next = QLabel("—")

        sky_form.addRow("Sunset (plan date):",   self.lbl_sunset)
        sky_form.addRow("Moon alt (plan date):",  self.lbl_moon_alt)
        sky_form.addRow("Moon illum (plan date):", self.lbl_moon_illum)
        sky_form.addRow("Cloud cover (now):",     self.lbl_cloud_now)
        sky_form.addRow("Cloud cover (+1 hr):",   self.lbl_cloud_next)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.setIcon(std_icon(self, "SP_BrowserReload"))
        btn_refresh.clicked.connect(self.refresh_sky)

        settings_box  = QGroupBox("Planning Settings")
        settings_form = QFormLayout(settings_box)
        settings_form.setContentsMargins(10, 8, 10, 8)
        settings_form.setVerticalSpacing(8)

        self.in_min_alt = QSpinBox();   self.in_min_alt.setRange(0, 90);   self.in_min_alt.setValue(int(core.DEFAULT_MIN_ALT_DEG))
        self.in_max_alt = QSpinBox();   self.in_max_alt.setRange(0, 90);   self.in_max_alt.setValue(int(core.DEFAULT_MAX_ALT_DEG))
        self.in_fov1    = QSpinBox();   self.in_fov1.setRange(1, 360);     self.in_fov1.setValue(core.DEFAULT_FOV1_ARCMIN)
        self.in_fov2    = QSpinBox();   self.in_fov2.setRange(1, 360);     self.in_fov2.setValue(core.DEFAULT_FOV2_ARCMIN)

        self.in_roll = QDoubleSpinBox()
        self.in_roll.setRange(-360.0, 360.0)
        self.in_roll.setDecimals(1)
        self.in_roll.setSingleStep(1.0)
        self.in_roll.setSuffix("°")
        self.in_roll.setValue(0.0)
        self.in_roll.valueChanged.connect(lambda _: self._roll_debounce.start())

        self.in_survey = QComboBox()
        self.in_survey.addItems(["DSS", "DSS2 Red", "DSS2 Blue", "Pan-STARRS"])
        self.in_survey.setCurrentText("DSS")
        self.in_survey.setToolTip("Survey used for finder charts. Pan-STARRS is deeper but not full-sky.")
        self.in_survey.currentIndexChanged.connect(self.refresh_finders_for_selected)

        settings_form.addRow("Min alt (°):",           self.in_min_alt)
        settings_form.addRow("Max alt (°):",           self.in_max_alt)
        settings_form.addRow("Finder FOV1 (arcmin):",  self.in_fov1)
        settings_form.addRow("Finder FOV2 (arcmin):",  self.in_fov2)
        settings_form.addRow("Finder roll:",           self.in_roll)
        settings_form.addRow("Finder source:",         self.in_survey)

        left_l.addWidget(plan_box)
        left_l.addWidget(sky_box)
        left_l.addWidget(btn_refresh)
        left_l.addWidget(settings_box)

        self.btn_exposure = QPushButton("Exposure Time Calculator (Beta)")
        self.btn_exposure.setToolTip(
            "Estimate exposure times and saturation-safe subexposures for all RHO filters."
        )
        self.btn_exposure.clicked.connect(self.open_exposure_calculator)
        left_l.addWidget(self.btn_exposure)
        left_l.addStretch(1)
        return left

    def _build_center_panel(self) -> QWidget:
        center   = QWidget()
        center_l = QVBoxLayout(center)
        center_l.setContentsMargins(6, 6, 6, 6)
        center_l.setSpacing(10)

        tabs = QTabWidget()
        tabs.addTab(self._build_manual_tab(),  "Manual Entry")
        tabs.addTab(self._build_upload_tab(), "Upload Target List")

        self.tbl = QTableWidget(0, 7)
        self.tbl.setHorizontalHeaderLabels(["Name", "RA", "Dec", "Priority", "Vmag", "Visible Windows", "Notes"])
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tbl.itemSelectionChanged.connect(self.on_row_selected)
        self.tbl.setAlternatingRowColors(True)
        self.tbl.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tbl.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.tbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tbl.setWordWrap(False)
        try: self.tbl.setTextElideMode(Qt.ElideRight)
        except Exception: pass

        hh = self.tbl.horizontalHeader()
        hh.setStretchLastSection(False); hh.setMinimumSectionSize(60)
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2, 3, 4):
            hh.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        for col in (5, 6):
            hh.setSectionResizeMode(col, QHeaderView.Stretch)

        btn_bar   = QWidget()
        btn_bar_l = QHBoxLayout(btn_bar)
        btn_bar_l.setContentsMargins(0, 0, 0, 0)
        btn_bar_l.setSpacing(10)

        self.btn_plan = QPushButton("Plan Observations")
        self.btn_plan.setIcon(std_icon(self, "SP_MediaPlay"))
        self.btn_plan.clicked.connect(self.plan_observations)

        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.setIcon(std_icon(self, "SP_TrashIcon"))
        self.btn_remove.clicked.connect(self.remove_selected)

        self.btn_clear = QPushButton("Clear Plan")
        self.btn_clear.setIcon(std_icon(self, "SP_DialogResetButton"))
        self.btn_clear.clicked.connect(self.clear_plan)

        btn_bar_l.addWidget(self.btn_plan)
        btn_bar_l.addWidget(self.btn_remove)
        btn_bar_l.addWidget(self.btn_clear)
        btn_bar_l.addStretch(1)

        center_l.addWidget(tabs)
        center_l.addWidget(QLabel("Planned Targets"))
        center_l.addWidget(self.tbl, 1)
        center_l.addWidget(btn_bar)
        return center

    def _build_right_panel(self) -> QWidget:
        right   = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(6, 6, 6, 6)
        right_l.setSpacing(10)

        self._right_split = QSplitter(Qt.Vertical)
        self._right_split.setChildrenCollapsible(False)

        card_css = """
        QWidget {
            border:1px solid #3a3d45; border-radius:10px;
            background-color:rgba(21,23,28,0.45);
        }"""

        alt_container = QWidget()
        alt_container.setStyleSheet(card_css)
        alt_l = QVBoxLayout(alt_container)
        alt_l.setContentsMargins(10, 10, 10, 10); alt_l.setSpacing(8)
        alt_l.addWidget(QLabel("Altitude Plot (selected date & location)"))

        alt_btn_row   = QWidget()
        alt_btn_row_l = QHBoxLayout(alt_btn_row)
        alt_btn_row_l.setContentsMargins(0, 0, 0, 0); alt_btn_row_l.setSpacing(8)

        self.btn_open_alt = QPushButton("Open Altitude Inspector")
        style_toolbar_button(self.btn_open_alt)
        self.btn_open_alt.clicked.connect(self.open_altitude_inspector)

        self.btn_copy_alt = QPushButton("Copy Altitude Plot")
        style_toolbar_button(self.btn_copy_alt)
        self.btn_copy_alt.setIcon(std_icon(self, "SP_DialogSaveButton"))
        self.btn_copy_alt.clicked.connect(self.copy_altitude_plot)

        alt_btn_row_l.addWidget(self.btn_open_alt)
        alt_btn_row_l.addWidget(self.btn_copy_alt)
        alt_btn_row_l.addStretch(1)
        alt_l.addWidget(alt_btn_row)

        self.alt_canvas = FigureCanvas(core.plt.figure(figsize=(7.8, 4.9)))
        alt_l.addWidget(self.alt_canvas, 1)
        self._right_split.addWidget(alt_container)

        finder_container = QWidget()
        finder_container.setStyleSheet(card_css)
        finder_l = QVBoxLayout(finder_container)
        finder_l.setContentsMargins(10, 10, 10, 10); finder_l.setSpacing(8)
        finder_l.addWidget(QLabel("Finder Charts (selected target)"))

        finder_btn_row   = QWidget()
        finder_btn_row_l = QHBoxLayout(finder_btn_row)
        finder_btn_row_l.setContentsMargins(0, 0, 0, 0); finder_btn_row_l.setSpacing(8)

        self.btn_open_fov1 = QPushButton("Open FOV1 Inspector")
        style_toolbar_button(self.btn_open_fov1)
        self.btn_open_fov1.clicked.connect(lambda: self.open_finder_inspector(1))

        self.btn_copy_fov1 = QPushButton("Copy FOV1")
        style_toolbar_button(self.btn_copy_fov1)
        self.btn_copy_fov1.setIcon(std_icon(self, "SP_DialogSaveButton"))
        self.btn_copy_fov1.clicked.connect(lambda: self.copy_finder_plot(1))

        self.btn_open_fov2 = QPushButton("Open FOV2 Inspector")
        style_toolbar_button(self.btn_open_fov2)
        self.btn_open_fov2.clicked.connect(lambda: self.open_finder_inspector(2))

        self.btn_copy_fov2 = QPushButton("Copy FOV2")
        style_toolbar_button(self.btn_copy_fov2)
        self.btn_copy_fov2.setIcon(std_icon(self, "SP_DialogSaveButton"))
        self.btn_copy_fov2.clicked.connect(lambda: self.copy_finder_plot(2))

        finder_btn_row_l.addWidget(self.btn_open_fov1); finder_btn_row_l.addWidget(self.btn_copy_fov1)
        finder_btn_row_l.addSpacing(12)
        finder_btn_row_l.addWidget(self.btn_open_fov2); finder_btn_row_l.addWidget(self.btn_copy_fov2)
        finder_btn_row_l.addStretch(1)
        finder_l.addWidget(finder_btn_row)

        self.finder_tabs     = QTabWidget()
        self.finder_canvas_1 = FigureCanvas(core.plt.figure(figsize=(7.8, 6.3)))
        self.finder_canvas_2 = FigureCanvas(core.plt.figure(figsize=(7.8, 6.3)))
        self.finder_tabs.addTab(self.finder_canvas_1, "FOV1")
        self.finder_tabs.addTab(self.finder_canvas_2, "FOV2")
        finder_l.addWidget(self.finder_tabs, 1)

        self._right_split.addWidget(finder_container)
        right_l.addWidget(self._right_split, 1)
        return right

    def _build_manual_tab(self) -> QWidget:
        w = QWidget(); l = QVBoxLayout(w); l.setSpacing(10)
        box  = QGroupBox("Add Target")
        form = QFormLayout(box); form.setVerticalSpacing(8)

        self.in_name = QLineEdit()
        self.in_ra   = QLineEdit()
        self.in_dec  = QLineEdit()
        self.in_pr   = QSpinBox(); self.in_pr.setRange(1, 5); self.in_pr.setValue(3)

        form.addRow("Name:",                 self.in_name)
        form.addRow("RA (hh:mm:ss OR deg):", self.in_ra)
        form.addRow("Dec (dd:mm:ss OR deg):", self.in_dec)
        form.addRow("Priority:",             self.in_pr)

        btn_add = QPushButton("Add to Plan")
        btn_add.setIcon(std_icon(self, "SP_DialogApplyButton"))
        btn_add.clicked.connect(self.add_manual)

        l.addWidget(box); l.addWidget(btn_add); l.addStretch(1)
        return w

    def _build_upload_tab(self) -> QWidget:
        w = QWidget(); l = QVBoxLayout(w); l.setSpacing(10)
        btn = QPushButton("Choose CSV/XLSX…")
        btn.setIcon(std_icon(self, "SP_DialogOpenButton"))
        btn.clicked.connect(self.upload_file)
        self.lbl_upload = QLabel("No file loaded.")
        self.lbl_upload.setWordWrap(True)
        l.addWidget(btn); l.addWidget(self.lbl_upload); l.addStretch(1)
        return w

    def open_exposure_calculator(self):
        row = self._selected_row
        name = row.name if row is not None else ""
        mag = row.vmag if row is not None else None

        if self._exposure_dialog is not None and self._exposure_dialog.isVisible():
            self._exposure_dialog.refresh_planner_targets(preferred_row=row)
            self._exposure_dialog.set_target(name, mag, planner_row=row)
            self._exposure_dialog.raise_()
            self._exposure_dialog.activateWindow()
            return

        dlg = ExposureCalculatorDialog(self, target_name=name, target_vmag=mag)
        self._exposure_dialog = dlg
        dlg.finished.connect(lambda _: setattr(self, "_exposure_dialog", None))
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _refresh_open_exposure_targets(self, preferred_row=None):
        """Keep an open calculator synchronized with planner-list changes."""
        dlg = self._exposure_dialog
        if dlg is not None and dlg.isVisible():
            dlg.refresh_planner_targets(preferred_row=preferred_row)

    def refresh_sky(self):
        self.statusBar().showMessage("Refreshing sky conditions…")
        cond = core.sky_conditions()

        sunset = cond.get("sunset_local")
        self.lbl_sunset.setText(sunset.strftime("%Y-%m-%d %H:%M") if sunset else "unavailable")

        moon_alt = cond.get("moon_alt_deg")
        self.lbl_moon_alt.setText(f"{moon_alt:.1f}°" if moon_alt is not None else "unavailable")

        illum = cond.get("moon_illum_frac")
        self.lbl_moon_illum.setText(f"{illum*100:.1f}%" if illum is not None else "unavailable")

        cloud_now = cond.get("cloud_now_pct")
        self.lbl_cloud_now.setText(f"{cloud_now:.0f}%" if cloud_now is not None else "unavailable")

        cloud_next = cond.get("cloud_next_pct")
        self.lbl_cloud_next.setText(f"{cloud_next:.0f}%" if cloud_next is not None else "unavailable")

        self.statusBar().showMessage("Ready")

    def apply_date_location(self, initial: bool = False):
        try:
            self.statusBar().showMessage("Applying site/date…")
            qd = self.date_edit.date()
            core.set_planning_date(date(qd.year(), qd.month(), qd.day()))

            tz = self.tz_combo.currentText().strip()
            if not tz:
                raise ValueError("Timezone cannot be blank (example: US/Eastern).")
            ZoneInfo(tz)   # validate early
            core.set_site(float(self.lat_spin.value()), float(self.lon_spin.value()),
                          float(self.height_spin.value()), tz)
            self.refresh_sky()

            if not initial and self.plan:
                self.plan_observations()
            self.statusBar().showMessage("Ready")
        except Exception as e:
            self.statusBar().showMessage("Apply failed.")
            QMessageBox.critical(self, "Apply failed", str(e))

    def reset_defaults(self):
        self.date_edit.setDate(QDate.currentDate())
        self.lat_spin.setValue(self._default_site.lat)
        self.lon_spin.setValue(self._default_site.lon)
        self.height_spin.setValue(self._default_site.height_m)
        self.tz_combo.setCurrentText(self._default_site.timezone)
        self.in_min_alt.setValue(self._default_min_alt)
        self.in_max_alt.setValue(self._default_max_alt)
        self.in_fov1.setValue(self._default_fov1)
        self.in_fov2.setValue(self._default_fov2)
        self.in_roll.setValue(self._default_roll)
        self.in_survey.setCurrentText(self._default_survey)
        self.apply_date_location(initial=False)

    def add_manual(self):
        try:
            name = self.in_name.text().strip()
            ra   = self.in_ra.text().strip()
            dec  = self.in_dec.text().strip()
            pr   = int(self.in_pr.value())
            if not (name or (ra and dec)):
                raise ValueError("Enter a name (SIMBAD) or provide RA/Dec.")
            row = PlanRow(name=name, ra=ra, dec=dec, priority=pr)
            self.plan.append(row)
            self._append_table_row(row)
            self._refresh_open_exposure_targets(preferred_row=row)
            self.statusBar().showMessage(f"Added target: {name or 'Unnamed'}")
        except Exception as e:
            self.statusBar().showMessage("Invalid target.")
            QMessageBox.critical(self, "Invalid target", str(e))

    def upload_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open target list", "",
                                              "Data Files (*.csv *.xlsx)")
        if not path:
            return
        try:
            self.statusBar().showMessage("Loading target list…")
            df = core.load_targets_from_file(path)
            self.lbl_upload.setText(f"Loaded {len(df)} rows from:\n{path}")
            for _, r in df.iterrows():
                vmag_val = r.get("vmag", np.nan)
                row = PlanRow(
                    name     = str(r["name"]),
                    ra       = str(r["ra"]),
                    dec      = str(r["dec"]),
                    priority = int(r["priority"]),
                    vmag     = "N/A" if pd.isna(vmag_val) else str(vmag_val),
                )
                self.plan.append(row)
                self._append_table_row(row)
            self._refresh_open_exposure_targets()
            self.statusBar().showMessage(f"Loaded {len(df)} targets.")
        except Exception as e:
            self.statusBar().showMessage("Upload failed.")
            QMessageBox.critical(self, "Upload failed", str(e))

    def clear_plan(self):
        self.plan = []
        self.tbl.setRowCount(0)
        self._last_coords = []
        self._last_names  = []
        self._selected_row = None
        self._raw_finder  = {}
        self._refresh_open_exposure_targets()
        self._set_altitude_fig(core.plt.figure(figsize=(7.8, 4.9)))
        self._set_finder_figs(core.plt.figure(figsize=(7.8, 6.3)),
                              core.plt.figure(figsize=(7.8, 6.3)))
        self.statusBar().showMessage("Plan cleared.")

    def remove_selected(self):
        idxs = self.tbl.selectionModel().selectedRows()
        if not idxs:
            QMessageBox.information(self, "No selection", "Select rows to remove.")
            return
        rows  = sorted({idx.row() for idx in idxs}, reverse=True)
        names = [self.plan[r].name for r in rows if 0 <= r < len(self.plan)]

        msg = (f"Remove '{names[0]}' from the plan?"
               if len(rows) == 1
               else f"Remove these {len(rows)} targets?\n\n" + "\n".join(names[:10]))
        if QMessageBox.question(self, "Remove target(s)", msg) != QMessageBox.Yes:
            return

        for r in rows:
            if 0 <= r < len(self.plan):
                self.plan.pop(r)
                self.tbl.removeRow(r)

        self._selected_row = None
        self._refresh_open_exposure_targets()
        self.statusBar().showMessage(f"Removed {len(rows)} target(s).")

        if not self.plan:
            self._last_coords = []; self._last_names = []
            self._raw_finder  = {}
            self._set_altitude_fig(core.plt.figure(figsize=(7.8, 4.9)))
            self._set_finder_figs(core.plt.figure(figsize=(7.8, 6.3)),
                                  core.plt.figure(figsize=(7.8, 6.3)))
            return

        new_r = min(rows[-1], self.tbl.rowCount() - 1)
        if new_r >= 0:
            self.tbl.selectRow(new_r)
            self.on_row_selected()

    def plan_observations(self):
        if not self.plan:
            QMessageBox.information(self, "No targets", "Add at least one target.")
            return
        min_alt = float(self.in_min_alt.value())
        max_alt = float(self.in_max_alt.value())
        if max_alt <= min_alt:
            QMessageBox.warning(self, "Settings", "Max altitude must be > min altitude.")
            return

        self.btn_plan.setEnabled(False)
        self.statusBar().showMessage("Planning observations…")

        worker = PlanWorker(self.plan, min_alt, max_alt)
        self._plan_workers.add(worker)
        worker.finished.connect(self.on_plan_finished)
        worker.failed.connect(self.on_plan_failed)

        def _cleanup():
            self._plan_workers.discard(worker)
            worker.deleteLater()
            self.btn_plan.setEnabled(True)

        worker.finished.connect(lambda *_: _cleanup())
        worker.failed.connect(lambda *_: _cleanup())
        worker.start()

    def on_plan_failed(self, msg: str):
        self.statusBar().showMessage("Planning failed.")
        QMessageBox.critical(self, "Planning failed", msg)

    def on_plan_finished(self, updated_plan, altitude_fig, coords, names):
        self.plan         = updated_plan
        self._last_coords = coords
        self._last_names  = names
        self._raw_finder  = {}   

        self.tbl.setRowCount(0)
        for row in self.plan:
            self._append_table_row(row)

        if altitude_fig is not None:
            self._set_altitude_fig(altitude_fig)

        if self.tbl.rowCount() > 0:
            self.tbl.selectRow(0)

        preferred = self.plan[0] if self.plan else None
        self._refresh_open_exposure_targets(preferred_row=preferred)
        self.statusBar().showMessage("Planning complete.")

    def refresh_finders_for_selected(self):
        if self.tbl.selectionModel().selectedRows():
            self.on_row_selected()

    def on_row_selected(self):
        idxs = self.tbl.selectionModel().selectedRows()
        if not idxs:
            return
        r = idxs[0].row()
        if r < 0 or r >= len(self.plan):
            return

        row               = self.plan[r]
        self._selected_row = row

        fov1 = int(self.in_fov1.value())
        fov2 = int(self.in_fov2.value())
        mode = self.in_survey.currentText()
        roll = float(self.in_roll.value())

        self.statusBar().showMessage(f"Generating finder charts for {row.name}…")
        self._finder_request_id += 1
        req_id = self._finder_request_id

        worker = FinderWorker(
            req_id, row.name, row.ra, row.dec,
            fov1, fov1, fov2, fov2, mode, roll_deg=roll,
        )
        self._finder_workers.add(worker)
        worker.finished.connect(self.on_finder_finished)
        worker.failed.connect(self.on_finder_failed)

        def _cleanup():
            self._finder_workers.discard(worker)
            worker.deleteLater()

        worker.finished.connect(lambda *_: _cleanup())
        worker.failed.connect(lambda *_: _cleanup())
        worker.start()

    def on_finder_failed(self, request_id: int, msg: str):
        if request_id != self._finder_request_id:
            return
        self.statusBar().showMessage("Finder chart failed.")
        QMessageBox.warning(self, "Finder chart failed", msg)

    def on_finder_finished(
        self,
        request_id, fig1, fig2,
        data1, wcs1, data2, wcs2,
        coord, lbl1, lbl2,
    ):
        if request_id != self._finder_request_id:
            return

        self._raw_finder[1] = (data1, wcs1, coord, lbl1, int(self.in_fov1.value()))
        self._raw_finder[2] = (data2, wcs2, coord, lbl2, int(self.in_fov2.value()))

        self._set_finder_figs(fig1, fig2)

        if self._open_finder_dialog_request is not None:
            dlg = self._open_finder_dialog_request
            if isinstance(dlg, FinderInspectorDialog):
                which  = dlg.which_fov
                raw    = self._raw_finder.get(which, (None, None, None, "", 90))
                fig    = fig1 if which == 1 else fig2
                dlg.set_figure(fig, *raw)
            self._open_finder_dialog_request = None

        self.statusBar().showMessage("Finder charts updated.")

    def _apply_roll_from_cache(self):
        roll = float(self.in_roll.value())
        row  = self._selected_row
        name = row.name if row else ""

        fig1 = fig2 = None
        for which in (1, 2):
            raw = self._raw_finder.get(which)
            if raw is None:
                continue
            data, wcs, coord, lbl, fov = raw
            if data is None or wcs is None or coord is None:
                continue
            try:
                fig = core.render_finder_figure_from_data(
                    coord, name, data, wcs, fov, lbl, roll_deg=roll)
                if which == 1:
                    fig1 = fig
                else:
                    fig2 = fig
            except Exception:
                pass
        if fig1 is not None and fig2 is not None:
            self._set_finder_figs(fig1, fig2)

    def open_altitude_inspector(self):
        if not self._last_coords:
            QMessageBox.information(self, "No altitude plot", "Plan at least one target first.")
            return
        if self._alt_dialog is not None and self._alt_dialog.isVisible():
            self._alt_dialog.raise_(); self._alt_dialog.activateWindow()
            return
        dlg = AltitudeInspectorDialog(self, self._last_coords, self._last_names,
                                      self.in_min_alt.value(), self.in_max_alt.value())
        self._alt_dialog = dlg
        dlg.finished.connect(lambda _: setattr(self, "_alt_dialog", None))
        dlg.show(); dlg.raise_(); dlg.activateWindow()

    def open_finder_inspector(self, which_fov: int):
        existing = self._finder_dialog_fov1 if which_fov == 1 else self._finder_dialog_fov2
        if existing is not None and existing.isVisible():
            existing.raise_(); existing.activateWindow()
            return

        raw    = self._raw_finder.get(which_fov, (None, None, None, "", 90))
        data, wcs, coord, lbl, fov = raw
        roll   = float(self.in_roll.value())
        row    = self._selected_row
        name   = row.name if row else ""

        if data is not None and wcs is not None and coord is not None:
            fig = core.render_finder_figure_from_data(coord, name, data, wcs, fov, lbl, roll_deg=roll)
        else:
            fig = core.plt.figure(figsize=(7.8, 6.3))

        dlg = FinderInspectorDialog(
            self, fig, which_fov,
            raw_data=data, raw_wcs=wcs, raw_coord=coord,
            raw_survey=lbl, raw_fov=fov,
        )
        if which_fov == 1:
            self._finder_dialog_fov1 = dlg
            dlg.finished.connect(lambda _: setattr(self, "_finder_dialog_fov1", None))
        else:
            self._finder_dialog_fov2 = dlg
            dlg.finished.connect(lambda _: setattr(self, "_finder_dialog_fov2", None))

        dlg.show(); dlg.raise_(); dlg.activateWindow()

    def copy_altitude_plot(self):
        if self.alt_canvas is None or self.alt_canvas.figure is None:
            QMessageBox.information(self, "No altitude plot", "No plot to copy yet.")
            return
        copy_figure_to_clipboard(self.alt_canvas.figure, self, "Altitude plot copied.")

    def copy_finder_plot(self, which_fov: int):
        canvas = self.finder_canvas_1 if which_fov == 1 else self.finder_canvas_2
        if canvas is None or canvas.figure is None:
            QMessageBox.information(self, "No finder chart", "No finder chart to copy yet.")
            return
        copy_figure_to_clipboard(canvas.figure, self, f"Finder FOV{which_fov} copied.")

    def _bind_altitude_click(self):
        self.alt_canvas.mpl_connect("button_press_event",
                                    lambda e: self.open_altitude_inspector())

    def _bind_finder_clicks(self):
        self.finder_canvas_1.mpl_connect("button_press_event",
                                          lambda e: self.open_finder_inspector(1))
        self.finder_canvas_2.mpl_connect("button_press_event",
                                          lambda e: self.open_finder_inspector(2))

    def closeEvent(self, event):
        for w in list(self._finder_workers) + list(self._plan_workers):
            if w.isRunning():
                w.requestInterruption()
                w.wait(2000)
        event.accept()

    def _set_altitude_fig(self, fig):
        parent = self.alt_canvas.parentWidget()
        layout = parent.layout()
        layout.removeWidget(self.alt_canvas)
        try: core.plt.close(self.alt_canvas.figure)
        except Exception: pass
        self.alt_canvas.setParent(None)
        self.alt_canvas = FigureCanvas(fig)
        layout.addWidget(self.alt_canvas, 1)
        self._bind_altitude_click()

    def _set_finder_figs(self, fig1, fig2):
        try: core.plt.close(self.finder_canvas_1.figure)
        except Exception: pass
        try: core.plt.close(self.finder_canvas_2.figure)
        except Exception: pass

        self.finder_tabs.clear()
        self.finder_canvas_1.setParent(None)
        self.finder_canvas_2.setParent(None)
        self.finder_canvas_1 = FigureCanvas(fig1)
        self.finder_canvas_2 = FigureCanvas(fig2)
        self.finder_tabs.addTab(self.finder_canvas_1, "FOV1")
        self.finder_tabs.addTab(self.finder_canvas_2, "FOV2")
        self._bind_finder_clicks()

    def _append_table_row(self, row: PlanRow):
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)
        vals = [row.name, row.ra, row.dec, str(row.priority),
                row.vmag, row.visible_windows,
                row.notes.strip() if row.notes else ""]
        for c, v in enumerate(vals):
            it = QTableWidgetItem(v)
            it.setFlags(it.flags() ^ Qt.ItemIsEditable)
            if c in (5, 6) and str(v).strip():
                it.setToolTip(str(v))
            self.tbl.setItem(r, c, it)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_app_style(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
