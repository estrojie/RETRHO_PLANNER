# main.py
from __future__ import annotations
import sys
from io import BytesIO
from dataclasses import dataclass
from typing import List
from datetime import date
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
# Matplotlib setup
import matplotlib
matplotlib.use("QtAgg") 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.patches import Rectangle
# PySide6
from PySide6.QtCore import Qt, QThread, Signal, QDate, QSize, QTimer, QSignalBlocker
from PySide6.QtGui import QTextDocument, QFont, QImage, QPixmap, QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QGroupBox, QFormLayout, QLabel, QPushButton, QTabWidget, QFileDialog,
    QTableWidget, QTableWidgetItem, QMessageBox, QLineEdit, QSpinBox,
    QAbstractItemView, QComboBox, QDoubleSpinBox, QDateEdit, QDialog,
    QListWidget, QListWidgetItem, QCheckBox, QSizePolicy, QHeaderView,
    QStyle,
)
# Astropy / Astroquery
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs.utils import pixel_to_skycoord
import warnings
from astroquery.exceptions import NoResultsWarning
# Local modules
import planner_core_v6 as core

# Qt6/PySide6 standard icon helper
def std_icon(widget: QWidget, enum_name: str):
    """
    Return a standard icon in a PySide6/Qt6 compatible way.
    enum_name example: "SP_DialogApplyButton"
    """
    sp = getattr(QStyle.StandardPixmap, enum_name, None)
    if sp is None:
        sp = getattr(QStyle, enum_name, None)
    if sp is None:
        sp = QStyle.StandardPixmap.SP_FileIcon
    return widget.style().standardIcon(sp)


def style_toolbar_button(btn: QPushButton):
    """Apply the highlighted toolbar-button styling already defined in the app QSS."""
    btn.setProperty("toolbarButton", True)
    btn.setCursor(Qt.PointingHandCursor)
    try:
        btn.style().unpolish(btn)
        btn.style().polish(btn)
    except Exception:
        pass


def copy_figure_to_clipboard(fig, parent=None, success_message: str = "Plot copied to clipboard."):
    try:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        image = QImage.fromData(buf.getvalue(), "PNG")
        if image.isNull():
            raise RuntimeError("Failed to create clipboard image from figure.")
        QGuiApplication.clipboard().setPixmap(QPixmap.fromImage(image))
        if parent is not None and hasattr(parent, "statusBar"):
            parent.statusBar().showMessage(success_message, 3000)
    except Exception as e:
        QMessageBox.warning(parent, "Clipboard Error", f"Could not copy plot to clipboard.\n\n{e}")


# App styling (Theme / QSS)
def apply_app_style(app: QApplication):
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet("""
    QMainWindow {
        background-color: qlineargradient(
            x1:0, y1:0, x2:0, y2:1,
            stop:0 #1f2228,
            stop:1 #181a1f
        );
    }
    QLabel { color: #d7d7d7; }
    QGroupBox {
        border: 1px solid #3a3d45;
        border-radius: 8px;
        margin-top: 10px;
        padding: 10px;
        font-weight: 600;
        color: #d0d0d0;
        background-color: rgba(21,23,28,0.55);
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px 0 6px;
        color: #cfcfcf;
    }
    QLineEdit, QSpinBox, QDoubleSpinBox, QDateEdit, QComboBox {
        background-color: #14161b;
        border: 1px solid #3a3d45;
        border-radius: 6px;
        padding: 5px;
        color: #e5e5e5;
        selection-background-color: #2b6cb0;
    }
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QDateEdit:focus, QComboBox:focus {
        border: 1px solid #4a90e2;
    }
    QPushButton {
        background-color: #2a2d33;
        border: 1px solid #3a3d45;
        padding: 7px 10px;
        border-radius: 6px;
        color: #f0f0f0;
    }
    QPushButton:hover { background-color: #353944; }
    QPushButton:pressed { background-color: #4b5060; }
    QPushButton[toolbarButton="true"] {
        background-color: #262b34;
        border: 1px solid #41506a;
        padding: 7px 12px;
    }
    QPushButton[toolbarButton="true"]:hover {
        background-color: #2e3f57;
        border: 1px solid #5a8fd8;
    }
    QPushButton[toolbarButton="true"]:pressed {
        background-color: #1f5fa6;
        border: 1px solid #79aef0;
    }
    QPushButton[toolbarButton="true"]:checked {
        background-color: #1b4f8a;
        border: 1px solid #4a90e2;
        color: #ffffff;
    }
    QPushButton:disabled { color: #888; background-color: #23262c; border-color: #2c2f36; }
    QPushButton:checked {
    background-color: #1b4f8a;
    border: 1px solid #4a90e2;
    color: #ffffff;
    }
    QPushButton:checked:hover {
        background-color: #2262a6;
    }
    QTabWidget::pane {
        border: 1px solid #3a3d45;
        border-radius: 8px;
        top: -1px;
        background-color: rgba(21,23,28,0.35);
    }
    QTabBar::tab {
        background: #2a2d33;
        padding: 7px 14px;
        border: 1px solid #3a3d45;
        border-bottom: none;
        border-top-left-radius: 7px;
        border-top-right-radius: 7px;
        color: #cfcfcf;
        margin-right: 2px;
    }
    QTabBar::tab:selected {
        background: #1b1d23;
        color: #ffffff;
    }
    QTableWidget {
        background-color: rgba(18,20,25,0.55);
        border: 1px solid #3a3d45;
        border-radius: 8px;
        gridline-color: #2f323a;
        color: #e5e5e5;
        selection-background-color: #1b4f8a;
        selection-color: #ffffff;
    }
    QHeaderView::section {
        background-color: #23262c;
        padding: 6px;
        border: 1px solid #3a3d45;
        color: #dcdcdc;
        font-weight: 600;
    }
    QStatusBar {
        background-color: rgba(18,20,25,0.65);
        border-top: 1px solid #3a3d45;
        color: #cfcfcf;
    }
    """)

# Data model
@dataclass
class PlanRow:
    name: str
    ra: str
    dec: str
    priority: int = 3
    vmag: str = "N/A"
    visible_windows: str = "—"
    notes: str = ""

# Workers
class PlanWorker(QThread):
    finished = Signal(list, object, list, list) 
    failed = Signal(str)

    def __init__(self, plan: List[PlanRow], min_alt: float, max_alt: float):
        super().__init__()
        self.plan = plan
        self.min_alt = min_alt
        self.max_alt = max_alt

    def run(self):
        try:
            updated: List[PlanRow] = []
            coords = []
            names = []

            for row in self.plan:
                try:
                    rt = core.resolve_target(row.name, row.ra, row.dec)
                    resolved_ok = True
                except Exception:
                    resolved_ok = False

                if not resolved_ok:
                    updated_row = PlanRow(
                        name=(row.name or "Unnamed Target"),
                        ra=row.ra or "—",
                        dec=row.dec or "—",
                        priority=row.priority,
                        vmag="N/A",
                        visible_windows="—",
                        notes="Failed to Resolve (Check Name or Coordinates)",
                    )
                    updated.append(updated_row)
                    continue

                # Notes rules
                note = ""
                try:
                    if rt.coord.dec.deg > core.DEC_WARNING_LIMIT_DEG:
                        note = "Dec > +60° (Potentially Not Observable at RHO)"
                except Exception:
                    pass

                # Visibility window 
                try:
                    windows = core.compute_visibility_windows(rt.coord, self.min_alt, self.max_alt)
                    windows_s = core.format_visibility_windows(windows)
                except Exception:
                    windows = []
                    windows_s = "—"

                # Vmag preference
                preferred_vmag = row.vmag
                if str(preferred_vmag).strip().lower() in ("", "n/a", "na", "nan", "none", "—", "-"):
                    preferred_vmag = rt.vmag

                vmag_s = "N/A"
                try:
                    v = float(preferred_vmag)
                    if np.isfinite(v):
                        vmag_s = f"{v:.2f}"
                except Exception:
                    sv = str(preferred_vmag).strip()
                    vmag_s = sv if sv else "N/A"

                updated_row = PlanRow(
                    name=rt.display_name,
                    ra=rt.coord.ra.to_string(unit=core.u.hour, sep=":", precision=2),
                    dec=rt.coord.dec.to_string(unit=core.u.deg, sep=":", precision=2, alwayssign=True),
                    priority=row.priority,
                    vmag=vmag_s,
                    visible_windows=windows_s,
                    notes=note,
                )
                updated.append(updated_row)

                coords.append(rt.coord)
                names.append(rt.display_name)

            alt_fig = core.plot_altitudes(coords, names, self.min_alt, self.max_alt) if coords else None
            self.finished.emit(updated, alt_fig, coords, names)
        except Exception as e:
            self.failed.emit(str(e))


class FinderWorker(QThread):
    finished = Signal(int, object, object)
    failed = Signal(int, str)

    def __init__(self, request_id: int, name: str, ra: str, dec: str, fov1: int, fov2: int, mode: str, roll_deg: float = 0.0):
        super().__init__()
        self.request_id = request_id
        self.name = name
        self.ra = ra
        self.dec = dec
        self.fov1 = fov1
        self.fov2 = fov2
        self.mode = mode
        self.roll_deg = float(roll_deg)

    def run(self):
        try:
            rt = core.resolve_target(self.name, self.ra, self.dec)
            fig1 = core.finder_figure(rt.coord, rt.display_name, fov_arcmin=self.fov1, mode=self.mode, roll_deg=self.roll_deg)
            fig2 = core.finder_figure(rt.coord, rt.display_name, fov_arcmin=self.fov2, mode=self.mode, roll_deg=self.roll_deg)
            self.finished.emit(self.request_id, fig1, fig2)
        except Exception as e:
            self.failed.emit(self.request_id, str(e))


from astroquery.simbad import Simbad


class StarIdWorker(QThread):
    finished = Signal(object)  
    failed = Signal(str)

    def __init__(self, coord_candidates: list[SkyCoord], radius_arcsec: float = 120.0):
        super().__init__()
        self.cands = list(coord_candidates)
        self.radius_arcsec = float(radius_arcsec)

        self.simbad = Simbad()
        self.simbad.add_votable_fields("main_id", "ra", "dec", "flux(V)")

    def _query_region(self, coord: SkyCoord):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NoResultsWarning)
            rad_min = self.radius_arcsec / 60.0
            return self.simbad.query_region(coord.icrs, radius=f"{rad_min}m")

    def _query_tap_nearest(self, coord: SkyCoord):
        ra_deg = float(coord.icrs.ra.deg)
        dec_deg = float(coord.icrs.dec.deg)
        rad_deg = self.radius_arcsec / 3600.0  # arcsec -> deg

        adql = f"""
        SELECT TOP 50
            main_id, ra, dec, flux_V,
            DISTANCE(
                POINT('ICRS', ra, dec),
                POINT('ICRS', {ra_deg}, {dec_deg})
            ) AS dist_deg
        FROM basic
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra_deg}, {dec_deg}, {rad_deg})
        )
        ORDER BY dist_deg ASC
        """

        try:
            t = self.simbad.query_tap(adql)
            return t
        except Exception:
            return None

    @staticmethod
    def _row_coord_from_table(row) -> SkyCoord | None:
        for ra_k, dec_k, as_deg in [
            ("ra", "dec", True),
            ("RA", "DEC", False),
            ("RAJ2000", "DEJ2000", False),
            ("RA_ICRS", "DEC_ICRS", False),
        ]:
            try:
                ra_v = row[ra_k]
                dec_v = row[dec_k]
            except Exception:
                continue

            try:
                if as_deg:
                    return SkyCoord(float(ra_v) * u.deg, float(dec_v) * u.deg, frame="icrs")
                else:
                    return SkyCoord(str(ra_v), str(dec_v), unit=(u.hourangle, u.deg), frame="icrs")
            except Exception:
                continue

        return None

    def run(self):
        try:
            for cand in self.cands:
                r = self._query_region(cand)
                if r is None or len(r) == 0:
                    r = self._query_tap_nearest(cand)

                if r is None or len(r) == 0:
                    continue

                best_row = None
                best_sep = None
                best_coord = None

                for row in r:
                    c = self._row_coord_from_table(row)
                    if c is None:
                        continue
                    try:
                        sep = cand.separation(c).arcsec
                    except Exception:
                        continue
                    if best_sep is None or sep < best_sep:
                        best_sep = sep
                        best_row = row
                        best_coord = c

                if best_row is None:
                    continue

                main_id = "Unknown"
                for k in ("MAIN_ID", "main_id"):
                    try:
                        main_id = str(best_row[k]).strip()
                        break
                    except Exception:
                        pass

                vmag = None
                for k in ("FLUX_V", "flux_V", "flux(V)", "V"):
                    try:
                        vmag = best_row[k]
                        break
                    except Exception:
                        pass

                self.finished.emit({
                    "ok": True,
                    "main_id": main_id,
                    "vmag": None if vmag is None else str(vmag),
                    "sep_arcsec": float(best_sep) if best_sep is not None else None,
                    "coord": best_coord,
                    "clicked": cand,
                })
                return

            self.finished.emit({"ok": False, "msg": "No SIMBAD object found within radius."})

        except Exception as e:
            self.failed.emit(str(e))

class AltitudeInspectorDialog(QDialog):
    def __init__(self, parent, coords, names, min_alt, max_alt):
        super().__init__(parent)
        self.setWindowTitle("Altitude/Airmass Inspector")
        self.resize(1100, 650)

        self.coords = list(coords)
        self.names = list(names)
        self.min_alt = float(min_alt)
        self.max_alt = float(max_alt)

        self.max_selected_objects = 5
        self._updating_selection = False

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        controls = QWidget()
        cl = QVBoxLayout(controls)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(10)

        self.chk_airmass = QCheckBox("Show Airmass (instead of Altitude)")
        self.chk_airmass.stateChanged.connect(self._replot)

        self.chk_utc = QCheckBox("Show UTC (instead of local time)")
        self.chk_utc.stateChanged.connect(self._replot)

        self.listw = QListWidget()
        self.listw.setSelectionMode(QAbstractItemView.MultiSelection)

        for i, nm in enumerate(self.names):
            it = QListWidgetItem(nm)
            it.setSelected(i < self.max_selected_objects)
            self.listw.addItem(it)

        self.listw.itemSelectionChanged.connect(self._on_selection_changed)

        cl.addWidget(self.chk_airmass)
        cl.addWidget(self.chk_utc)
        cl.addWidget(QLabel("Objects to display (max 5):"))

        sel_btns = QWidget()
        sel_btns_l = QHBoxLayout(sel_btns)
        sel_btns_l.setContentsMargins(0, 0, 0, 0)
        sel_btns_l.setSpacing(8)

        self.btn_select_first5 = QPushButton("Select First 5")
        self.btn_select_first5.clicked.connect(self._select_first_five)

        self.btn_unselect_all = QPushButton("Unselect All")
        self.btn_unselect_all.clicked.connect(self._unselect_all)

        self.btn_copy_plot = QPushButton("Copy Plot")
        self.btn_copy_plot.setIcon(std_icon(self, "SP_DialogSaveButton"))
        self.btn_copy_plot.clicked.connect(lambda: copy_figure_to_clipboard(self.canvas.figure, self, "Altitude plot copied to clipboard."))

        sel_btns_l.addWidget(self.btn_select_first5)
        sel_btns_l.addWidget(self.btn_unselect_all)
        sel_btns_l.addWidget(self.btn_copy_plot)

        cl.addWidget(sel_btns)
        cl.addWidget(self.listw, 1)

        plotw = QWidget()
        self.plot_layout = QVBoxLayout(plotw)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(6)

        fig0 = core.plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvas(fig0)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.canvas, 1)

        root.addWidget(controls, 0)
        root.addWidget(plotw, 1)
        root.setStretch(0, 0)
        root.setStretch(1, 1)

        self._replot()

    def _selected_names(self):
        return [
            self.listw.item(i).text()
            for i in range(self.listw.count())
            if self.listw.item(i).isSelected()
        ]

    def _replace_plot(self, fig):
        self.plot_layout.removeWidget(self.toolbar)
        self.plot_layout.removeWidget(self.canvas)
        self.toolbar.setParent(None)
        self.canvas.setParent(None)

        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.canvas, 1)

    def _replot(self):
        y_mode = "airmass" if self.chk_airmass.isChecked() else "altitude"
        selected = self._selected_names()
        tz_mode = "utc" if self.chk_utc.isChecked() else "local"

        fig = core.plot_altitudes(
            self.coords,
            self.names,
            self.min_alt,
            self.max_alt,
            y_mode=y_mode,
            only_names=selected,
            display_tz=tz_mode,
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

        selected_items = [
            self.listw.item(i)
            for i in range(self.listw.count())
            if self.listw.item(i).isSelected()
        ]

        if len(selected_items) <= self.max_selected_objects:
            return

        self._updating_selection = True
        try:
            for it in selected_items[self.max_selected_objects:]:
                it.setSelected(False)
        finally:
            self._updating_selection = False

        QMessageBox.information(
            self,
            "Selection limit",
            f"You can display at most {self.max_selected_objects} objects at a time."
        )

# Finder inspector
class FinderInspectorDialog(QDialog):
    @staticmethod
    def _fmt_vmag(v) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        if s.lower() in ("", "none", "nan", "masked", "--", "—"):
            return None
        try:
            fv = float(s)
            if not np.isfinite(fv):
                return None
            return f"V={fv:.2f}"
        except Exception:
            return f"V={s}"

    @staticmethod
    def _fmt_sep_arcsec(sep) -> str | None:
        if sep is None:
            return None
        try:
            fs = float(sep)
            if not np.isfinite(fs):
                return None
        except Exception:
            return None

        if fs < 1.0:
            base = f"{fs:.2f}\""
        elif fs < 10.0:
            base = f"{fs:.1f}\""
        else:
            base = f"{fs:.1f}\""

        if fs >= 60.0:
            base += f" = {fs/60.0:.2f}′"

        return f"[{base}]"
    def _remove_artists(self, artists):
        """Remove matplotlib artists safely."""
        if not artists:
            return
        for a in artists:
            try:
                a.remove()
            except Exception:
                pass

    def _rebuild_numbering(self):
        n = 0
        for i in range(self.list_ident.count()):
            it = self.list_ident.item(i)
            meta = it.data(Qt.UserRole) or {}
            if not meta.get("is_marker", False):
                continue

            n += 1
            num_text = meta.get("num_text", None)
            if num_text is not None:
                try:
                    num_text.set_text(str(n))
                except Exception:
                    pass
            txt = it.text()

            import re
            txt2 = re.sub(r"^\s*\d+\.\s*", "", txt)
            it.setText(f"{n}. {txt2}")
            it.setToolTip(it.text())
            it.setSizeHint(self._sizehint_for_text(it.text()))

        self._label_counter = n
        self.canvas.draw_idle()
    def _draw_numbered_marker(self, ax, x, y, number: int, label=None):
        """Draw a marker + number bubble on the plot without touching _label_counter."""
        created = []

        p = ax.plot(
            [x], [y],
            marker="o",
            markersize=max(4, int(self._lw() * 3)),
            linestyle=""
        )[0]
        self._ann_artists.append(p)
        created.append(p)

        label_artist = None
        if label:
            label_artist = ax.text(x, y, f" {label}", fontsize=10)
            self._ann_texts.append(label_artist)
            created.append(label_artist)

        num_artist = ax.text(
            x, y,
            f"{number}",
            fontsize=10,
            fontweight="bold",
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.18", alpha=0.65)
        )
        self._marker_labels.append(num_artist)
        created.append(num_artist)

        return created, num_artist    
    def __init__(self, parent, initial_fig, which_fov: int):
        super().__init__(parent)
        self.setWindowTitle(f"Finder Inspector (FOV{which_fov})")
        self.resize(1250, 780)

        self.parent_window = parent
        self.which_fov = which_fov
        self._current_roll_deg = float(parent.in_roll.value()) if hasattr(parent, "in_roll") else 0.0

        self.mode = None
        self._press_xy = None

        self._rect_patch = None
        self._circle_patch = None

        self._ann_artists = []
        self._ann_texts = []
        self._measure_markers = []
        self._measure_p1 = None
        self._measure_artist = None
        self._label_counter = 0
        self._pending_ident_item = None
        self._pending_ident_num = None
        self._pending_ident_xy = None  
        self._marker_labels = []   
        self._free_line = None
        self._free_xs = []
        self._free_ys = []

        self._id_workers = set()

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        left = QWidget()
        self.left_l = QVBoxLayout(left)
        self.left_l.setContentsMargins(0, 0, 0, 0)
        self.left_l.setSpacing(10)

        top = QWidget()
        tl = QHBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(10)

        self.fov_spin = QSpinBox()
        self.fov_spin.setRange(1, 360)
        self.fov_spin.setValue(int(parent.in_fov1.value()) if which_fov == 1 else int(parent.in_fov2.value()))

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
        self.roll_spin.setValue(self._current_roll_deg)

        self.btn_copy_plot = QPushButton("Copy Plot")
        style_toolbar_button(self.btn_copy_plot)
        self.btn_copy_plot.setIcon(std_icon(self, "SP_DialogSaveButton"))
        self.btn_copy_plot.clicked.connect(lambda: copy_figure_to_clipboard(self.canvas.figure, self, "Finder chart copied to clipboard."))

        tl.addWidget(QLabel("FOV (arcmin):"))
        tl.addWidget(self.fov_spin)
        tl.addWidget(btn_apply_fov)
        tl.addSpacing(20)
        tl.addWidget(QLabel("Roll:"))
        tl.addWidget(self.roll_spin)
        tl.addSpacing(20)
        tl.addWidget(QLabel("Line thickness:"))
        tl.addWidget(self.lw_spin)
        tl.addStretch(1)
        tl.addWidget(self.btn_copy_plot)

        self.left_l.addWidget(top)

        tools_box = QGroupBox("Tools")
        tools_l = QHBoxLayout(tools_box)
        tools_l.setContentsMargins(10, 8, 10, 8)
        tools_l.setSpacing(8)

        def _mk_tool_btn(text, mode_name):
            b = QPushButton(text)
            b.setCheckable(True)
            b.toggled.connect(lambda on: self._set_mode(mode_name if on else None))
            return b

        self.btn_draw_rect = _mk_tool_btn("Draw Rectangle", "rect")
        self.btn_draw_circle = _mk_tool_btn("Draw Circle", "circle")
        self.btn_free_draw = _mk_tool_btn("Free Draw", "free")
        self.btn_guide = _mk_tool_btn("Mark Guide Star", "guide")
        self.btn_measure = _mk_tool_btn("Measure Separation", "measure")
        self.btn_identify = _mk_tool_btn("Identify Star", "identify")

        self.btn_clear = QPushButton("Clear Annotations")
        self.btn_clear.setIcon(std_icon(self, "SP_TrashIcon"))
        self.btn_clear.clicked.connect(self._clear_annotations)

        tools_l.addWidget(self.btn_draw_rect)
        tools_l.addWidget(self.btn_draw_circle)
        tools_l.addWidget(self.btn_free_draw)
        tools_l.addWidget(self.btn_guide)
        tools_l.addWidget(self.btn_measure)
        tools_l.addWidget(self.btn_identify)
        tools_l.addStretch(1)
        tools_l.addWidget(self.btn_clear)

        self.left_l.addWidget(tools_box)

        plotw = QWidget()
        self.plot_layout = QVBoxLayout(plotw)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(6)

        self.canvas = None
        self.toolbar = None
        self._cid_press = None
        self._cid_move = None
        self._cid_rel = None

        self._replace_plot(initial_fig)
        self.left_l.addWidget(plotw, 1)

        right = QGroupBox("Identified / Labeled Objects")
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(10, 10, 10, 10)
        right_l.setSpacing(8)

        self.list_ident = QListWidget()
        self.list_ident.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_ident.setWordWrap(True)
        self.list_ident.setUniformItemSizes(False)
        try:
            self.list_ident.setTextElideMode(Qt.ElideNone)
        except Exception:
            pass

        try:
            f = self.list_ident.font()
            f.setFamily("Consolas")
            self.list_ident.setFont(f)
        except Exception:
            pass

        right_l.addWidget(self.list_ident, 1)

        btns = QWidget()
        btns_l = QHBoxLayout(btns)
        btns_l.setContentsMargins(0, 0, 0, 0)
        btns_l.setSpacing(10)

        self.btn_remove_selected = QPushButton("Remove Selected")
        self.btn_remove_selected.setIcon(std_icon(self, "SP_TrashIcon"))
        self.btn_remove_selected.clicked.connect(self._remove_selected_labels)

        self.btn_clear_list = QPushButton("Clear List")
        self.btn_clear_list.setIcon(std_icon(self, "SP_DialogResetButton"))
        self.btn_clear_list.clicked.connect(self._clear_list)

        btns_l.addWidget(self.btn_remove_selected)
        btns_l.addWidget(self.btn_clear_list)
        right_l.addWidget(btns)

        root.addWidget(left, 1)
        root.addWidget(right, 0)
        root.setStretch(0, 4)
        root.setStretch(1, 1)

        self._update_wcs_hint_in_title()

    def _sizehint_for_text(self, txt: str) -> QSize:
        txt = str(txt)
        doc = QTextDocument()
        doc.setDefaultFont(self.list_ident.font())
        doc.setPlainText(txt)
        vw = max(180, self.list_ident.viewport().width() - 18)
        doc.setTextWidth(vw)
        h = int(doc.size().height()) + 10
        return QSize(vw, max(22, h))

    def _add_list_item(self, txt: str, meta: dict | None = None) -> QListWidgetItem:
        txt = str(txt)
        it = QListWidgetItem(txt)
        it.setData(Qt.UserRole, meta or {})
        it.setToolTip(txt)
        it.setSizeHint(self._sizehint_for_text(txt))
        self.list_ident.addItem(it)
        return it

    def _set_last_list_item_text(self, txt: str):
        it = self.list_ident.item(self.list_ident.count() - 1)
        if it is None:
            return
        txt = str(txt)
        it.setText(txt)
        it.setToolTip(txt)
        it.setSizeHint(self._sizehint_for_text(txt))

    def _clear_measure_points(self):
        for a in list(self._measure_markers):
            try:
                a.remove()
            except Exception:
                pass
        self._measure_markers.clear()

    def _replace_plot(self, fig):
        if self.toolbar is not None:
            self.plot_layout.removeWidget(self.toolbar)
            self.toolbar.setParent(None)
        if self.canvas is not None:
            self.plot_layout.removeWidget(self.canvas)
            try:
                if self._cid_press is not None:
                    self.canvas.mpl_disconnect(self._cid_press)
                if self._cid_move is not None:
                    self.canvas.mpl_disconnect(self._cid_move)
                if self._cid_rel is not None:
                    self.canvas.mpl_disconnect(self._cid_rel)
            except Exception:
                pass
            try:
                core.plt.close(self.canvas.figure)
            except Exception:
                pass
            self.canvas.setParent(None)

        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.canvas, 1)

        self._cid_press = self.canvas.mpl_connect("button_press_event", self._on_press)
        self._cid_move = self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self._cid_rel = self.canvas.mpl_connect("button_release_event", self._on_release)

    def set_figure(self, fig):
        self._clear_annotations()
        self._replace_plot(fig)
        self._current_roll_deg = float(self.roll_spin.value()) if hasattr(self, "roll_spin") else self._current_roll_deg
        self._update_wcs_hint_in_title()
        self._label_counter = 0

    def _ax(self):
        if not self.canvas or not self.canvas.figure:
            return None

        axes = list(self.canvas.figure.axes)
        if not axes:
            return None

        best = None
        for a in axes:
            w = getattr(a, "wcs", None) or getattr(a, "_rho_wcs", None)
            if w is None:
                continue
            try:
                if getattr(w, "has_celestial", False) and w.has_celestial:
                    best = a
                    break
            except Exception:
                continue

        if best is not None:
            return best

        try:
            return max(axes, key=lambda a: a.get_position().width * a.get_position().height)
        except Exception:
            return axes[0]

    def _coord_candidates_from_click(self, ax, x, y) -> list[SkyCoord]:
        cands: list[SkyCoord] = []

        def _accept(c: SkyCoord):
            try:
                if np.isfinite(c.ra.deg) and np.isfinite(c.dec.deg) and (-90 <= c.dec.deg <= 90):
                    cands.append(c.icrs)
            except Exception:
                pass

        w = getattr(ax, "wcs", None) or getattr(ax, "_rho_wcs", None)
        has_celestial = False
        if w is not None:
            try:
                has_celestial = bool(getattr(w, "has_celestial", False) and w.has_celestial)
            except Exception:
                has_celestial = False

        x0 = float(x)
        y0 = float(y)
        roll_deg = float(getattr(ax, "_rho_roll_deg", 0.0) or 0.0)
        shape = getattr(ax, "_rho_data_shape", None)
        if shape is not None and len(shape) >= 2 and abs(roll_deg) > 1e-9:
            try:
                h, wpx = float(shape[0]), float(shape[1])
                x0, y0 = core.viewer_to_original_pixel(x0, y0, wpx, h, roll_deg)
            except Exception:
                pass

        if has_celestial:
            for origin in (0, 1):
                try:
                    c = pixel_to_skycoord(float(x0), float(y0), w, origin=origin)
                    if c is not None:
                        _accept(c)
                except Exception:
                    pass

            try:
                c2 = w.pixel_to_world(float(x0), float(y0))
                if isinstance(c2, SkyCoord):
                    _accept(c2)
            except Exception:
                pass

        try:
            _accept(SkyCoord(float(x0) * u.deg, float(y0) * u.deg, frame="icrs"))
        except Exception:
            pass

        try:
            _accept(SkyCoord(float(x0) * u.hourangle, float(y0) * u.deg, frame="icrs"))
        except Exception:
            pass

        uniq: list[SkyCoord] = []
        for c in cands:
            if not uniq:
                uniq.append(c)
            else:
                try:
                    if all(c.separation(u2).arcsec > 0.01 for u2 in uniq):
                        uniq.append(c)
                except Exception:
                    uniq.append(c)

        return uniq

    def _lw(self) -> float:
        return float(self.lw_spin.value())

    def _update_wcs_hint_in_title(self):
        ax = self._ax()
        if ax is None:
            return
        w = getattr(ax, "wcs", None) or getattr(ax, "_rho_wcs", None)
        rolled = float(getattr(ax, "_rho_roll_deg", 0.0) or 0.0)
        if w is not None:
            extra = f" (WCS OK, roll={rolled:.1f}°)" if abs(rolled) > 1e-9 else " (WCS OK)"
        else:
            extra = f" (No WCS — roll={rolled:.1f}°)" if abs(rolled) > 1e-9 else " (No WCS — ID/sky sep disabled)"
        t = ax.get_title()
        if extra not in t:
            ax.set_title(t + extra)
            self.canvas.draw_idle()

    def _set_mode(self, mode: str | None):
        self.mode = mode

        self._press_xy = None
        self._rect_patch = None
        self._circle_patch = None

        if mode != "measure":
            self._measure_p1 = None

        self._free_line = None
        self._free_xs = []
        self._free_ys = []

        mapping = {
            "rect": self.btn_draw_rect,
            "circle": self.btn_draw_circle,
            "free": self.btn_free_draw,
            "guide": self.btn_guide,
            "measure": self.btn_measure,
            "identify": self.btn_identify,
        }
        for m, btn in mapping.items():
            should = (mode == m)
            if btn.isChecked() != should:
                btn.blockSignals(True)
                btn.setChecked(should)
                btn.blockSignals(False)

    def _on_press(self, event):
        ax = event.inaxes
        if ax is None:
            return
        if (self.canvas is None) or (ax.figure is not self.canvas.figure):
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)

        if self.mode == "rect":
            self._press_xy = (x, y)
            self._rect_patch = Rectangle((x, y), 0, 0, fill=False, linewidth=self._lw())
            ax.add_patch(self._rect_patch)
            self.canvas.draw_idle()
            return

        if self.mode == "circle":
            self._press_xy = (x, y)
            from matplotlib.patches import Circle
            self._circle_patch = Circle((x, y), radius=0.0, fill=False, linewidth=self._lw())
            ax.add_patch(self._circle_patch)
            self.canvas.draw_idle()
            return

        if self.mode == "free":
            self._free_xs = [x]
            self._free_ys = [y]
            self._free_line = ax.plot(self._free_xs, self._free_ys, "-", linewidth=self._lw())[0]
            self.canvas.draw_idle()
            return

        if self.mode == "guide":
            self._place_marker(ax, x, y, label="Guide")
            self._set_mode(None)
            return

        if self.mode == "measure":
            if self._measure_p1 is None:
                self._clear_measure_artist()
                self._clear_measure_points()

                self._measure_p1 = (x, y)
                p = ax.plot([x], [y], marker="o", markersize=max(4, int(self._lw() * 3)), linestyle="")[0]
                t = ax.text(x, y, " P1", fontsize=10)
                self._measure_markers.extend([p, t])

                self.canvas.draw_idle()
                return

            p2 = (x, y)
            p = ax.plot([x], [y], marker="o", markersize=max(4, int(self._lw() * 3)), linestyle="")[0]
            t = ax.text(x, y, " P2", fontsize=10)
            self._measure_markers.extend([p, t])

            self._draw_measurement(ax, self._measure_p1, p2)
            self._measure_p1 = None
            self.canvas.draw_idle()
            return

        if self._measure_p1 is None:
            self._clear_measure_artist()

        if self.mode == "identify":
            self._identify_at_click(ax, x, y)
            return

    def _on_move(self, event):
        ax = event.inaxes
        if ax is None:
            return
        if (self.canvas is None) or (ax.figure is not self.canvas.figure):
            return
        if event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)

        if self.mode == "rect" and self._press_xy and self._rect_patch:
            x0, y0 = self._press_xy
            self._rect_patch.set_width(x - x0)
            self._rect_patch.set_height(y - y0)
            self.canvas.draw_idle()
            return

        if self.mode == "circle" and self._press_xy and self._circle_patch:
            x0, y0 = self._press_xy
            r = float(np.hypot(x - x0, y - y0))
            self._circle_patch.set_radius(r)
            self.canvas.draw_idle()
            return

        if self.mode == "free" and self._free_line is not None:
            self._free_xs.append(x)
            self._free_ys.append(y)
            self._free_line.set_data(self._free_xs, self._free_ys)
            self.canvas.draw_idle()
            return

    def _on_release(self, event):
        if self.mode == "rect" and self._rect_patch is not None:
            self._ann_artists.append(self._rect_patch)
            self._rect_patch = None
            self._press_xy = None
            self._set_mode(None)
            return

        if self.mode == "circle" and self._circle_patch is not None:
            self._ann_artists.append(self._circle_patch)
            self._circle_patch = None
            self._press_xy = None
            self._set_mode(None)
            return

        if self.mode == "free" and self._free_line is not None:
            self._ann_artists.append(self._free_line)
            self._free_line = None
            self._free_xs = []
            self._free_ys = []
            self._set_mode(None)
            return

    def _place_marker(self, ax, x, y, label=None, add_to_list=True, meta=None):
        if ax is None:
            return

        created_artists = []

        marker_num = None
        if add_to_list:
            self._label_counter += 1
            marker_num = self._label_counter

        p = ax.plot(
            [x], [y],
            marker="o",
            markersize=max(4, int(self._lw() * 3)),
            linestyle=""
        )[0]
        self._ann_artists.append(p)
        created_artists.append(p)

        label_artist = None
        if label:
            label_artist = ax.text(x, y, f" {label}", fontsize=10)
            self._ann_texts.append(label_artist)
            created_artists.append(label_artist)

        num_artist = None
        if marker_num is not None:
            num_artist = ax.text(
                x, y,
                f"{marker_num}",
                fontsize=10,
                fontweight="bold",
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.18", alpha=0.65)
            )
            self._marker_labels.append(num_artist)
            created_artists.append(num_artist)

        self.canvas.draw_idle()

        if add_to_list:
            base_txt = str(label) if label else "Marker"
            list_txt = f"{marker_num}. {base_txt}" if marker_num is not None else base_txt

            it = self._add_list_item(list_txt, meta={})
            it_meta = {
                "is_marker": True,
                "artists": created_artists,     
                "num_text": num_artist,         
                "raw_meta": meta or {},         
            }
            it.setData(Qt.UserRole, it_meta)
            it.setToolTip(list_txt)
            it.setSizeHint(self._sizehint_for_text(list_txt))
        marker_num = None
        if add_to_list:
            self._label_counter += 1
            marker_num = self._label_counter
        p = ax.plot(
            [x], [y],
            marker="o",
            markersize=max(4, int(self._lw() * 3)),
            linestyle=""
        )[0]
        self._ann_artists.append(p)

        if label:
            t = ax.text(x, y, f" {label}", fontsize=10)
            self._ann_texts.append(t)

        if marker_num is not None:
            tn = ax.text(
                x, y,
                f"{marker_num}",
                fontsize=10,
                fontweight="bold",
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.18", alpha=0.65)
            )
            self._marker_labels.append(tn)

        self.canvas.draw_idle()

        if add_to_list:
            base_txt = str(label) if label else "Marker"
            if marker_num is not None:
                base_txt = f"{marker_num}. {base_txt}"
            self._add_list_item(base_txt, meta=meta or {})

    def _clear_measure_artist(self):
        if self._measure_artist is None:
            return
        try:
            for a in (self._measure_artist if isinstance(self._measure_artist, tuple) else (self._measure_artist,)):
                try:
                    a.remove()
                except Exception:
                    pass
        except Exception:
            pass
        self._measure_artist = None

    def _draw_measurement(self, ax, p1, p2):
        if ax is None:
            return

        self._clear_measure_artist()
        x1, y1 = p1
        x2, y2 = p2
        dpix = float(np.hypot(x2 - x1, y2 - y1))

        cands1 = self._coord_candidates_from_click(ax, x1, y1)
        cands2 = self._coord_candidates_from_click(ax, x2, y2)

        best_sep = None
        best_pair = None

        if cands1 and cands2:
            for c1 in cands1:
                for c2 in cands2:
                    try:
                        sep = c1.separation(c2)
                        if best_sep is None or sep < best_sep:
                            best_sep = sep
                            best_pair = (c1, c2)
                    except Exception:
                        pass

        sky_txt = ""
        if best_sep is not None and best_pair is not None:
            c1, c2 = best_pair
            sep_as = best_sep.to(u.arcsec).value
            sep_am = best_sep.to(u.arcmin).value
            try:
                pa = c1.position_angle(c2).to(u.deg).value
                pa_txt = f", PA={pa:.1f}°"
            except Exception:
                pa_txt = ""
            sky_txt = f"{sep_as:.1f}\" ({sep_am:.2f}′){pa_txt}"

        line = ax.plot([x1, x2], [y1, y2], "-", linewidth=self._lw())[0]
        txt = ax.text(
            (x1 + x2) / 2.0, (y1 + y2) / 2.0,
            f"{dpix:.1f}px\n{sky_txt}",
            fontsize=10,
            ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.6)
        )

        self._measure_artist = (line, txt)
        self.canvas.draw_idle()

    def _clear_annotations(self):
        ax = self._ax()
        if ax is None:
            return
        self._clear_measure_artist()
        self._clear_measure_points()

        for a in list(self._ann_artists):
            try:
                a.remove()
            except Exception:
                pass
        for t in list(self._ann_texts):
            try:
                t.remove()
            except Exception:
                pass
        for t in list(self._marker_labels):
            try:
                t.remove()
            except Exception:
                pass
        self._label_counter = 0
        self._pending_ident_item = None
        self._pending_ident_num = None
        self._pending_ident_xy = None
        self._marker_labels.clear()                    
        self._ann_artists.clear()
        self._ann_texts.clear()
        self._measure_p1 = None
        self.canvas.draw_idle()

    def _remove_selected_labels(self):
        items = list(self.list_ident.selectedItems())
        if not items:
            return

        for it in items:
            meta = it.data(Qt.UserRole) or {}
            if meta.get("is_marker", False):
                self._remove_artists(meta.get("artists", []))
            self.list_ident.takeItem(self.list_ident.row(it))

        self._rebuild_numbering()

    def _clear_list(self):
        for i in range(self.list_ident.count()):
            it = self.list_ident.item(i)
            meta = it.data(Qt.UserRole) or {}
            if meta.get("is_marker", False):
                self._remove_artists(meta.get("artists", []))

        self.list_ident.clear()

        self._label_counter = 0
        self._marker_labels.clear()
        self._ann_artists.clear()
        self._ann_texts.clear()

        self.canvas.draw_idle()

    def _identify_at_click(self, ax, x, y):
        cands = self._coord_candidates_from_click(ax, x, y)
        if not cands:
            QMessageBox.information(self, "Identify", "Could not interpret click coordinates (no WCS?).")
            return
        self._label_counter += 1
        num = self._label_counter

        pending_text = f"{num}. (querying SIMBAD...)"
        it = QListWidgetItem(pending_text)
        it.setToolTip(pending_text)
        it.setSizeHint(self._sizehint_for_text(pending_text))

        it_meta = {
            "is_marker": True,
            "artists": [],
            "num_text": None,
            "raw_meta": {},
        }
        it.setData(Qt.UserRole, it_meta)
        self.list_ident.addItem(it)

        self._pending_ident_item = it
        self._pending_ident_num = num
        self._pending_ident_xy = (ax, float(x), float(y))

        # Status bar feedback
        try:
            self.parent_window.statusBar().showMessage("Querying SIMBAD…")
        except Exception:
            pass

        w = StarIdWorker(cands, radius_arcsec=600.0)
        self._id_workers.add(w)

        def _cleanup():
            self._id_workers.discard(w)
            w.deleteLater()

        def _finish_with_text(text_block: str, meta_payload: dict):
            # If the dialog was cleared while query was running, just bail safely
            if self._pending_ident_item is None or self._pending_ident_xy is None:
                return

            ax0, x0, y0 = self._pending_ident_xy
            num0 = int(self._pending_ident_num)

            # Draw exactly ONE marker for this identify action
            artists, num_artist = self._draw_numbered_marker(ax0, x0, y0, num0, label=None)

            # Attach artists to the existing list item
            it0 = self._pending_ident_item
            m = it0.data(Qt.UserRole) or {}
            m["artists"] = artists
            m["num_text"] = num_artist
            m["raw_meta"] = meta_payload or {}
            it0.setData(Qt.UserRole, m)

            # Update list text
            full_text = f"{num0}. {text_block}"
            it0.setText(full_text)
            it0.setToolTip(full_text)
            it0.setSizeHint(self._sizehint_for_text(full_text))

            self.canvas.draw_idle()

            # clear pending state
            self._pending_ident_item = None
            self._pending_ident_num = None
            self._pending_ident_xy = None

        def _ok(res):
            if not res.get("ok", False):
                # Build "no match" display
                try:
                    c0 = cands[0]
                    ra_s = c0.ra.to_string(unit=u.hour, sep=":", precision=2)
                    dec_s = c0.dec.to_string(unit=u.deg, sep=":", precision=1, alwayssign=True)
                    text_block = f"Unknown (no match)\nRA={ra_s}  Dec={dec_s}"
                    meta_payload = {"ok": False, "ra": ra_s, "dec": dec_s}
                except Exception:
                    text_block = "Unknown (no match)"
                    meta_payload = {"ok": False}

                _finish_with_text(text_block, meta_payload)

                try:
                    self.parent_window.statusBar().showMessage("SIMBAD identify: no match.")
                except Exception:
                    pass
                return

            main_id = res.get("main_id", "Unknown")
            vmag_txt = self._fmt_vmag(res.get("vmag", None))
            sep_txt = self._fmt_sep_arcsec(res.get("sep_arcsec", None))
            cbest = res.get("coord", None)

            line1 = str(main_id)
            extras = []
            if vmag_txt:
                extras.append(vmag_txt)
            if sep_txt:
                extras.append(sep_txt)
            if extras:
                line1 += "  " + "  ".join(extras)

            lines = [line1]
            if cbest is not None:
                ra_s = cbest.ra.to_string(unit=u.hour, sep=":", precision=2)
                dec_s = cbest.dec.to_string(unit=u.deg, sep=":", precision=1, alwayssign=True)
                lines.append(f"RA={ra_s}  Dec={dec_s}")

            text_block = "\n".join(lines)
            _finish_with_text(text_block, res)

            try:
                self.parent_window.statusBar().showMessage("SIMBAD identify complete.")
            except Exception:
                pass

        w.finished.connect(_ok)
        w.finished.connect(_cleanup)
        w.failed.connect(lambda msg: QMessageBox.warning(self, "SIMBAD query failed", msg))
        w.failed.connect(lambda msg: (self.parent_window.statusBar().showMessage("SIMBAD query failed.") if self.parent_window else None))
        w.failed.connect(_cleanup)
        w.start()

    def _request_new_finder(self):
        if self.parent_window._selected_row is None:
            QMessageBox.information(self, "No target", "Select a target first.")
            return

        new_fov = int(self.fov_spin.value())
        new_roll = float(self.roll_spin.value())
        self.parent_window.in_roll.setValue(new_roll)
        if self.which_fov == 1:
            self.parent_window.in_fov1.setValue(new_fov)
        else:
            self.parent_window.in_fov2.setValue(new_fov)

        self.parent_window._open_finder_dialog_request = self
        self.parent_window.on_row_selected()

# Main window
class MainWindow(QMainWindow):
    def reset_defaults(self):
        # Restore default date/location/timezone
        self.date_edit.setDate(self._default_plan_date)

        self.lat_spin.setValue(float(self._default_site.lat))
        self.lon_spin.setValue(float(self._default_site.lon))
        self.height_spin.setValue(float(self._default_site.height_m))
        self.tz_combo.setCurrentText(str(self._default_site.timezone))

        # Restore Planning Settings defaults
        self.in_min_alt.setValue(self._default_min_alt)
        self.in_max_alt.setValue(self._default_max_alt)
        self.in_fov1.setValue(self._default_fov1)
        self.in_fov2.setValue(self._default_fov2)
        self.in_roll.setValue(self._default_roll)
        self.in_survey.setCurrentText(self._default_survey)

        # Apply immediately so sky conditions/plots update
        self.apply_date_location(initial=False)

    def __init__(self):
        self._alt_dialog = None
        self._finder_dialog_fov1 = None
        self._finder_dialog_fov2 = None

        super().__init__()
        self.setWindowTitle("Observing Planner (Desktop)")
        self.resize(1650, 920)
        self.setMinimumSize(1500, 840)

        # Status bar
        self.statusBar().showMessage("Ready")

        self.plan: List[PlanRow] = []
        self._finder_workers = set()
        self._plan_workers = set()
        self._finder_request_id = 0

        self._last_coords = []
        self._last_names = []
        self._selected_row = None
        self._open_finder_dialog_request = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter)

        # Left panel
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(6, 6, 6, 6)
        left_l.setSpacing(10)

        plan_box = QGroupBox("Planning Setup")
        plan_form = QFormLayout(plan_box)
        plan_form.setContentsMargins(10, 8, 10, 8)
        plan_form.setVerticalSpacing(8)

        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        today = QDate.currentDate()
        self.date_edit.setDate(today)

        site = core.get_site_config()
        # Defaults snapshot 
        self._default_site = site
        self._default_plan_date = QDate.currentDate()
        self._default_min_alt = int(core.DEFAULT_MIN_ALT_DEG)
        self._default_max_alt = int(core.DEFAULT_MAX_ALT_DEG)
        self._default_fov1 = int(core.DEFAULT_FOV1_ARCMIN)
        self._default_fov2 = int(core.DEFAULT_FOV2_ARCMIN)
        self._default_roll = 0.0
        self._default_survey = "DSS"

        self.lat_spin = QDoubleSpinBox()
        self.lat_spin.setRange(-90.0, 90.0)
        self.lat_spin.setDecimals(6)
        self.lat_spin.setValue(site.lat)

        self.lon_spin = QDoubleSpinBox()
        self.lon_spin.setRange(-180.0, 180.0)
        self.lon_spin.setDecimals(6)
        self.lon_spin.setValue(site.lon)

        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(-500.0, 9000.0)
        self.height_spin.setDecimals(1)
        self.height_spin.setValue(site.height_m)

        self.tz_combo = QComboBox()
        self.tz_combo.setEditable(True)

        # Common timezones
        tz_list = [
            "US/Eastern", "US/Central", "US/Mountain", "US/Pacific",
            "UTC",
        ]
        self.tz_combo.addItems(tz_list)

        # Set initial timezone from site config
        self.tz_combo.setCurrentText(site.timezone)

        self.btn_apply = QPushButton("Apply date/location")
        self.btn_apply.setIcon(std_icon(self, "SP_DialogApplyButton"))
        self.btn_apply.clicked.connect(self.apply_date_location)

        self.btn_reset_defaults = QPushButton("Reset defaults")
        self.btn_reset_defaults.setIcon(std_icon(self, "SP_DialogResetButton"))
        self.btn_reset_defaults.clicked.connect(self.reset_defaults)

        plan_form.addRow("Date:", self.date_edit)
        plan_form.addRow("Latitude (deg):", self.lat_spin)
        plan_form.addRow("Longitude (deg):", self.lon_spin)
        plan_form.addRow("Elevation (m):", self.height_spin)
        plan_form.addRow("Timezone (IANA):", self.tz_combo)
        btn_row = QWidget()
        btn_row_l = QHBoxLayout(btn_row)
        btn_row_l.setContentsMargins(0, 0, 0, 0)
        btn_row_l.setSpacing(8)
        btn_row_l.addWidget(self.btn_apply)
        btn_row_l.addWidget(self.btn_reset_defaults)
        plan_form.addRow(btn_row)

        sky_box = QGroupBox("Sky Conditions")
        sky_form = QFormLayout(sky_box)
        sky_form.setContentsMargins(10, 8, 10, 8)
        sky_form.setVerticalSpacing(8)

        self.lbl_sunset = QLabel("—")
        self.lbl_moon_alt = QLabel("—")
        self.lbl_moon_illum = QLabel("—")
        self.lbl_cloud_now = QLabel("—")
        self.lbl_cloud_next = QLabel("—")

        sky_form.addRow("Sunset (plan date):", self.lbl_sunset)
        sky_form.addRow("Moon alt (plan date):", self.lbl_moon_alt)
        sky_form.addRow("Moon illum (plan date):", self.lbl_moon_illum)
        sky_form.addRow("Cloud cover (now):", self.lbl_cloud_now)
        sky_form.addRow("Cloud cover (+1 hr):", self.lbl_cloud_next)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.setIcon(std_icon(self, "SP_BrowserReload"))
        btn_refresh.clicked.connect(self.refresh_sky)

        settings_box = QGroupBox("Planning Settings")
        settings_form = QFormLayout(settings_box)
        settings_form.setContentsMargins(10, 8, 10, 8)
        settings_form.setVerticalSpacing(8)

        self.in_min_alt = QSpinBox()
        self.in_min_alt.setRange(0, 90)
        self.in_min_alt.setValue(int(core.DEFAULT_MIN_ALT_DEG))

        self.in_max_alt = QSpinBox()
        self.in_max_alt.setRange(0, 90)
        self.in_max_alt.setValue(int(core.DEFAULT_MAX_ALT_DEG))

        self.in_fov1 = QSpinBox()
        self.in_fov1.setRange(1, 360)
        self.in_fov1.setValue(core.DEFAULT_FOV1_ARCMIN)

        self.in_fov2 = QSpinBox()
        self.in_fov2.setRange(1, 360)
        self.in_fov2.setValue(core.DEFAULT_FOV2_ARCMIN)

        self.in_roll = QDoubleSpinBox()
        self.in_roll.setRange(-360.0, 360.0)
        self.in_roll.setDecimals(1)
        self.in_roll.setSingleStep(1.0)
        self.in_roll.setSuffix("°")
        self.in_roll.setValue(0.0)
        self.in_roll.valueChanged.connect(self.refresh_finders_for_selected)

        self.in_survey = QComboBox()
        self.in_survey.addItems(["DSS", "DSS2 Red", "DSS2 Blue", "Pan-STARRS"])
        self.in_survey.setCurrentText("DSS")
        self.in_survey.setToolTip("Choose the survey used for the finder chart. Pan-STARRS is deeper but is not full-sky.")
        self.in_survey.currentIndexChanged.connect(self.refresh_finders_for_selected)

        settings_form.addRow("Min alt (°):", self.in_min_alt)
        settings_form.addRow("Max alt (°):", self.in_max_alt)
        settings_form.addRow("Finder FOV1 (arcmin):", self.in_fov1)
        settings_form.addRow("Finder FOV2 (arcmin):", self.in_fov2)
        settings_form.addRow("Finder roll:", self.in_roll)
        settings_form.addRow("Finder source:", self.in_survey)

        left_l.addWidget(plan_box)
        left_l.addWidget(sky_box)
        left_l.addWidget(btn_refresh)
        left_l.addWidget(settings_box)
        left_l.addStretch(1)

        # Center panel
        center = QWidget()
        center_l = QVBoxLayout(center)
        center_l.setContentsMargins(6, 6, 6, 6)
        center_l.setSpacing(10)

        tabs = QTabWidget()
        tabs.addTab(self._build_manual_tab(), "Manual Entry")
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
        try:
            self.tbl.setTextElideMode(Qt.ElideRight)
        except Exception:
            pass

        hh = self.tbl.horizontalHeader()
        hh.setStretchLastSection(False)
        hh.setMinimumSectionSize(60)
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(5, QHeaderView.Stretch)
        hh.setSectionResizeMode(6, QHeaderView.Stretch)

        btn_bar = QWidget()
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

        # Right panel
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(6, 6, 6, 6)
        right_l.setSpacing(10)

        right_split = QSplitter(Qt.Vertical)
        right_split.setChildrenCollapsible(False)

        # Card-ish container style for plot panels
        card_css = """
        QWidget {
            border: 1px solid #3a3d45;
            border-radius: 10px;
            background-color: rgba(21,23,28,0.45);
        }
        """

        alt_container = QWidget()
        alt_container.setStyleSheet(card_css)
        alt_l = QVBoxLayout(alt_container)
        alt_l.setContentsMargins(10, 10, 10, 10)
        alt_l.setSpacing(8)
        alt_l.addWidget(QLabel("Altitude Plot (selected date & location)"))

        alt_btn_row = QWidget()
        alt_btn_row_l = QHBoxLayout(alt_btn_row)
        alt_btn_row_l.setContentsMargins(0, 0, 0, 0)
        alt_btn_row_l.setSpacing(8)

        self.btn_open_altitude_inspector = QPushButton("Open Altitude Inspector")
        style_toolbar_button(self.btn_open_altitude_inspector)
        self.btn_open_altitude_inspector.setIcon(std_icon(self, "SP_FileDialogDetailedView"))
        self.btn_open_altitude_inspector.clicked.connect(self.open_altitude_inspector)

        self.btn_copy_altitude = QPushButton("Copy Altitude Plot")
        style_toolbar_button(self.btn_copy_altitude)
        self.btn_copy_altitude.setIcon(std_icon(self, "SP_DialogSaveButton"))
        self.btn_copy_altitude.clicked.connect(self.copy_altitude_plot)

        alt_btn_row_l.addWidget(self.btn_open_altitude_inspector)
        alt_btn_row_l.addWidget(self.btn_copy_altitude)
        alt_btn_row_l.addStretch(1)
        alt_l.addWidget(alt_btn_row)

        self.alt_canvas = FigureCanvas(core.plt.figure(figsize=(7.8, 4.9)))
        alt_l.addWidget(self.alt_canvas, 1)
        right_split.addWidget(alt_container)

        finder_container = QWidget()
        finder_container.setStyleSheet(card_css)
        finder_l = QVBoxLayout(finder_container)
        finder_l.setContentsMargins(10, 10, 10, 10)
        finder_l.setSpacing(8)
        finder_l.addWidget(QLabel("Finder Charts (selected target)"))

        finder_btn_row = QWidget()
        finder_btn_row_l = QHBoxLayout(finder_btn_row)
        finder_btn_row_l.setContentsMargins(0, 0, 0, 0)
        finder_btn_row_l.setSpacing(8)

        self.btn_open_finder_1 = QPushButton("Open FOV1 Inspector")
        style_toolbar_button(self.btn_open_finder_1)
        self.btn_open_finder_1.setIcon(std_icon(self, "SP_FileDialogDetailedView"))
        self.btn_open_finder_1.clicked.connect(lambda: self.open_finder_inspector(1))

        self.btn_copy_finder_1 = QPushButton("Copy FOV1")
        style_toolbar_button(self.btn_copy_finder_1)
        self.btn_copy_finder_1.setIcon(std_icon(self, "SP_DialogSaveButton"))
        self.btn_copy_finder_1.clicked.connect(lambda: self.copy_finder_plot(1))

        self.btn_open_finder_2 = QPushButton("Open FOV2 Inspector")
        style_toolbar_button(self.btn_open_finder_2)
        self.btn_open_finder_2.setIcon(std_icon(self, "SP_FileDialogDetailedView"))
        self.btn_open_finder_2.clicked.connect(lambda: self.open_finder_inspector(2))

        self.btn_copy_finder_2 = QPushButton("Copy FOV2")
        style_toolbar_button(self.btn_copy_finder_2)
        self.btn_copy_finder_2.setIcon(std_icon(self, "SP_DialogSaveButton"))
        self.btn_copy_finder_2.clicked.connect(lambda: self.copy_finder_plot(2))

        finder_btn_row_l.addWidget(self.btn_open_finder_1)
        finder_btn_row_l.addWidget(self.btn_copy_finder_1)
        finder_btn_row_l.addSpacing(12)
        finder_btn_row_l.addWidget(self.btn_open_finder_2)
        finder_btn_row_l.addWidget(self.btn_copy_finder_2)
        finder_btn_row_l.addStretch(1)
        finder_l.addWidget(finder_btn_row)

        self.finder_tabs = QTabWidget()
        self.finder_canvas_1 = FigureCanvas(core.plt.figure(figsize=(7.8, 6.3)))
        self.finder_canvas_2 = FigureCanvas(core.plt.figure(figsize=(7.8, 6.3)))
        self.finder_tabs.addTab(self.finder_canvas_1, "FOV1")
        self.finder_tabs.addTab(self.finder_canvas_2, "FOV2")
        finder_l.addWidget(self.finder_tabs, 1)

        right_split.addWidget(finder_container)
        right_l.addWidget(right_split, 1)

        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)

        # Apply splitter sizes AFTER layout/show
        QTimer.singleShot(0, lambda: splitter.setSizes([330, 840, 520]))
        QTimer.singleShot(0, lambda: right_split.setSizes([420, 620]))

        self._bind_altitude_click()
        self._bind_finder_clicks()

        self.apply_date_location(initial=True)

    # Apply date/location
    def apply_date_location(self, initial: bool = False):
        try:
            self.statusBar().showMessage("Applying site/date…")

            qd = self.date_edit.date()
            d = date(qd.year(), qd.month(), qd.day())
            core.set_planning_date(d)

            lat = float(self.lat_spin.value())
            lon = float(self.lon_spin.value())
            h = float(self.height_spin.value())
            tz = self.tz_combo.currentText().strip()
            if not tz:
                raise ValueError("Timezone cannot be blank (example: US/Eastern).")

            ZoneInfo(tz)
            core.set_site(lat, lon, h, tz, name=None)

            self.refresh_sky()

            if (not initial) and self.plan:
                self.plan_observations()

            self.statusBar().showMessage("Ready")

        except Exception as e:
            self.statusBar().showMessage("Apply failed.")
            QMessageBox.critical(self, "Apply failed", str(e))

    # Tabs
    def _build_manual_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setSpacing(10)

        box = QGroupBox("Add Target")
        form = QFormLayout(box)
        form.setVerticalSpacing(8)

        self.in_name = QLineEdit()
        self.in_ra = QLineEdit()
        self.in_dec = QLineEdit()
        self.in_pr = QSpinBox()
        self.in_pr.setRange(1, 5)
        self.in_pr.setValue(3)

        form.addRow("Name:", self.in_name)
        form.addRow("RA (hh:mm:ss OR deg):", self.in_ra)
        form.addRow("Dec (dd:mm:ss OR deg):", self.in_dec)
        form.addRow("Priority:", self.in_pr)

        btn_add = QPushButton("Add to Plan")
        btn_add.setIcon(std_icon(self, "SP_DialogApplyButton"))
        btn_add.clicked.connect(self.add_manual)

        l.addWidget(box)
        l.addWidget(btn_add)
        l.addStretch(1)
        return w

    def _build_upload_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setSpacing(10)

        btn = QPushButton("Choose CSV/XLSX…")
        btn.setIcon(std_icon(self, "SP_DialogOpenButton"))
        btn.clicked.connect(self.upload_file)

        self.lbl_upload = QLabel("No file loaded.")
        self.lbl_upload.setWordWrap(True)

        l.addWidget(btn)
        l.addWidget(self.lbl_upload)
        l.addStretch(1)
        return w

    # Sky
    def refresh_sky(self):
        self.statusBar().showMessage("Refreshing sky conditions…")
        cond = core.sky_conditions()

        sunset = cond.get("sunset_local", None)
        self.lbl_sunset.setText(sunset.strftime("%Y-%m-%d %H:%M") if sunset else "unavailable")

        moon_alt = cond.get("moon_alt_deg", None)
        self.lbl_moon_alt.setText(f"{moon_alt:.1f}°" if moon_alt is not None else "unavailable")

        illum = cond.get("moon_illum_frac", None)
        self.lbl_moon_illum.setText(f"{illum*100:.1f}%" if illum is not None else "unavailable")

        cloud_now = cond.get("cloud_now_pct", None)
        self.lbl_cloud_now.setText(f"{cloud_now:.0f}%" if cloud_now is not None else "unavailable")

        cloud_next = cond.get("cloud_next_pct", None)
        self.lbl_cloud_next.setText(f"{cloud_next:.0f}%" if cloud_next is not None else "unavailable")

        self.statusBar().showMessage("Ready")

    # Plan manipulation
    def add_manual(self):
        try:
            name = self.in_name.text().strip()
            ra = self.in_ra.text().strip()
            dec = self.in_dec.text().strip()
            pr = int(self.in_pr.value())

            if not (name or (ra and dec)):
                raise ValueError("Enter a name (SIMBAD) or provide RA/Dec.")

            row = PlanRow(name=name, ra=ra, dec=dec, priority=pr)
            self.plan.append(row)
            self._append_table_row(row)
            self.statusBar().showMessage(f"Added target: {name or 'Unnamed'}")

        except Exception as e:
            self.statusBar().showMessage("Invalid target.")
            QMessageBox.critical(self, "Invalid target", str(e))

    def upload_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open target list", "", "Data Files (*.csv *.xlsx)")
        if not path:
            return
        try:
            self.statusBar().showMessage("Loading target list…")
            df = core.load_targets_from_file(path)
            self.lbl_upload.setText(f"Loaded {len(df)} rows from:\n{path}")

            for _, r in df.iterrows():
                vmag_val = r.get("vmag", np.nan)
                row = PlanRow(
                    name=str(r["name"]),
                    ra=str(r["ra"]),
                    dec=str(r["dec"]),
                    priority=int(r["priority"]),
                    vmag=("N/A" if pd.isna(vmag_val) else str(vmag_val)),
                )
                self.plan.append(row)
                self._append_table_row(row)

            self.statusBar().showMessage(f"Loaded {len(df)} targets.")
        except Exception as e:
            self.statusBar().showMessage("Upload failed.")
            QMessageBox.critical(self, "Upload failed", str(e))

    def clear_plan(self):
        self.plan = []
        self.tbl.setRowCount(0)
        self._last_coords = []
        self._last_names = []
        self._selected_row = None
        self._set_altitude_fig(core.plt.figure(figsize=(7.8, 4.9)))
        self._set_finder_figs(core.plt.figure(figsize=(7.8, 6.3)), core.plt.figure(figsize=(7.8, 6.3)))
        self.statusBar().showMessage("Plan cleared.")

    def remove_selected(self):
        idxs = self.tbl.selectionModel().selectedRows()
        if not idxs:
            QMessageBox.information(self, "No selection", "Select one or more rows in the table to remove.")
            return

        rows = sorted({idx.row() for idx in idxs}, reverse=True)
        names = [self.plan[r].name for r in rows if 0 <= r < len(self.plan)]

        if len(rows) == 1:
            msg = f"Remove '{names[0]}' from the plan?"
        else:
            preview = "\n".join(names[:10])
            extra = "" if len(names) <= 10 else f"\n...and {len(names) - 10} more"
            msg = f"Remove these {len(rows)} targets from the plan?\n\n{preview}{extra}"

        if QMessageBox.question(self, "Remove target(s)", msg) != QMessageBox.Yes:
            return

        for r in rows:
            if 0 <= r < len(self.plan):
                self.plan.pop(r)
                self.tbl.removeRow(r)

        self._selected_row = None

        if len(rows) == 1:
            self.statusBar().showMessage(f"Removed: {names[0]}")
        else:
            self.statusBar().showMessage(f"Removed {len(rows)} targets.")

        if not self.plan:
            self._last_coords = []
            self._last_names = []
            self._set_altitude_fig(core.plt.figure(figsize=(7.8, 4.9)))
            self._set_finder_figs(core.plt.figure(figsize=(7.8, 6.3)), core.plt.figure(figsize=(7.8, 6.3)))
            return

        new_r = min(rows[-1], self.tbl.rowCount() - 1)
        if new_r >= 0:
            self.tbl.selectRow(new_r)
            self.on_row_selected()

    # Planning
    def plan_observations(self):
        if not self.plan:
            QMessageBox.information(self, "No targets", "Add at least one target to the plan.")
            return

        min_alt = float(self.in_min_alt.value())
        max_alt = float(self.in_max_alt.value())
        if max_alt <= min_alt:
            QMessageBox.warning(self, "Settings", "Max altitude must be greater than min altitude.")
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

        worker.finished.connect(_cleanup)
        worker.failed.connect(_cleanup)
        worker.start()

    def on_plan_failed(self, msg: str):
        self.statusBar().showMessage("Planning failed.")
        QMessageBox.critical(self, "Planning failed", msg)

    def on_plan_finished(self, updated_plan: List[PlanRow], altitude_fig, coords, names):
        self.plan = updated_plan
        self._last_coords = coords
        self._last_names = names

        self.tbl.setRowCount(0)
        for row in self.plan:
            self._append_table_row(row)

        if altitude_fig is not None:
            self._set_altitude_fig(altitude_fig)

        if self.tbl.rowCount() > 0:
            self.tbl.selectRow(0)

        self.statusBar().showMessage("Planning complete.")
        
    # Finder charts
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

        row = self.plan[r]
        self._selected_row = row

        fov1 = int(self.in_fov1.value())
        fov2 = int(self.in_fov2.value())
        mode = self.in_survey.currentText()

        self.statusBar().showMessage(f"Generating finder charts for {row.name}…")

        self._finder_request_id += 1
        req_id = self._finder_request_id

        worker = FinderWorker(req_id, row.name, row.ra, row.dec, fov1, fov2, mode, roll_deg=float(self.in_roll.value()))
        self._finder_workers.add(worker)

        worker.finished.connect(self.on_finder_finished)
        worker.failed.connect(self.on_finder_failed)

        def _cleanup():
            self._finder_workers.discard(worker)
            worker.deleteLater()

        worker.finished.connect(_cleanup)
        worker.failed.connect(_cleanup)
        worker.start()

    def on_finder_failed(self, request_id: int, msg: str):
        if request_id != self._finder_request_id:
            return
        self.statusBar().showMessage("Finder chart failed.")
        QMessageBox.warning(self, "Finder chart failed", msg)

    def on_finder_finished(self, request_id: int, fig1, fig2):
        if request_id != self._finder_request_id:
            return
        self._set_finder_figs(fig1, fig2)

        if self._open_finder_dialog_request is not None:
            dlg = self._open_finder_dialog_request
            if isinstance(dlg, FinderInspectorDialog):
                dlg.set_figure(fig1 if dlg.which_fov == 1 else fig2)
            self._open_finder_dialog_request = None

        self.statusBar().showMessage("Finder charts updated.")

    def open_altitude_inspector(self):
        if not self._last_coords or not self._last_names:
            QMessageBox.information(self, "No altitude plot", "Plan at least one target first.")
            return

        if self._alt_dialog is not None and self._alt_dialog.isVisible():
            self._alt_dialog.raise_()
            self._alt_dialog.activateWindow()
            return

        dlg = AltitudeInspectorDialog(self, self._last_coords, self._last_names, self.in_min_alt.value(), self.in_max_alt.value())
        self._alt_dialog = dlg
        dlg.finished.connect(lambda _: setattr(self, "_alt_dialog", None))
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def open_finder_inspector(self, which_fov: int):
        existing = self._finder_dialog_fov1 if which_fov == 1 else self._finder_dialog_fov2
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return

        fig = None
        try:
            if self._selected_row is not None:
                row = self._selected_row
                mode = self.in_survey.currentText()
                fov = int(self.in_fov1.value()) if which_fov == 1 else int(self.in_fov2.value())
                rt = core.resolve_target(row.name, row.ra, row.dec)
                fig = core.finder_figure(rt.coord, rt.display_name, fov_arcmin=fov, mode=mode, roll_deg=float(self.in_roll.value()))
        except Exception:
            fig = None

        if fig is None:
            fig = core.plt.figure(figsize=(7.8, 6.3))

        dlg = FinderInspectorDialog(self, fig, which_fov)
        if which_fov == 1:
            self._finder_dialog_fov1 = dlg
            dlg.finished.connect(lambda _: setattr(self, "_finder_dialog_fov1", None))
        else:
            self._finder_dialog_fov2 = dlg
            dlg.finished.connect(lambda _: setattr(self, "_finder_dialog_fov2", None))

        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def copy_altitude_plot(self):
        if self.alt_canvas is None or self.alt_canvas.figure is None:
            QMessageBox.information(self, "No altitude plot", "There is no altitude plot to copy yet.")
            return
        copy_figure_to_clipboard(self.alt_canvas.figure, self, "Altitude plot copied to clipboard.")

    def copy_finder_plot(self, which_fov: int):
        canvas = self.finder_canvas_1 if which_fov == 1 else self.finder_canvas_2
        if canvas is None or canvas.figure is None:
            QMessageBox.information(self, "No finder chart", "There is no finder chart to copy yet.")
            return
        copy_figure_to_clipboard(canvas.figure, self, f"Finder FOV{which_fov} copied to clipboard.")

    # Click bindings
    def _bind_altitude_click(self):
        def _on_click(event):
            self.open_altitude_inspector()

        self.alt_canvas.mpl_connect("button_press_event", _on_click)

    def _bind_finder_clicks(self):
        self.finder_canvas_1.mpl_connect("button_press_event", lambda e: self.open_finder_inspector(1))
        self.finder_canvas_2.mpl_connect("button_press_event", lambda e: self.open_finder_inspector(2))

    # Clean shutdown
    def closeEvent(self, event):
        for w in list(self._finder_workers) + list(self._plan_workers):
            if w.isRunning():
                w.requestInterruption()
                w.wait(2000)
        event.accept()

    # Canvas replacement helpers
    def _set_altitude_fig(self, fig):
        parent = self.alt_canvas.parentWidget()
        layout = parent.layout()
        layout.removeWidget(self.alt_canvas)
        try:
            core.plt.close(self.alt_canvas.figure)
        except Exception:
            pass
        self.alt_canvas.setParent(None)
        self.alt_canvas = FigureCanvas(fig)
        layout.addWidget(self.alt_canvas, 1)
        self._bind_altitude_click()

    def _set_finder_figs(self, fig1, fig2):
        try:
            core.plt.close(self.finder_canvas_1.figure)
        except Exception:
            pass
        try:
            core.plt.close(self.finder_canvas_2.figure)
        except Exception:
            pass

        self.finder_tabs.clear()
        self.finder_canvas_1.setParent(None)
        self.finder_canvas_2.setParent(None)
        self.finder_canvas_1 = FigureCanvas(fig1)
        self.finder_canvas_2 = FigureCanvas(fig2)
        self.finder_tabs.addTab(self.finder_canvas_1, "FOV1")
        self.finder_tabs.addTab(self.finder_canvas_2, "FOV2")
        self._bind_finder_clicks()

    # Table helper
    def _append_table_row(self, row: PlanRow):
        r = self.tbl.rowCount()
        self.tbl.insertRow(r)

        vals = [
            row.name,
            row.ra,
            row.dec,
            str(row.priority),
            row.vmag,
            row.visible_windows,
            row.notes if (row.notes and row.notes.strip()) else "",
        ]

        for c, v in enumerate(vals):
            it = QTableWidgetItem(v)
            it.setFlags(it.flags() ^ Qt.ItemIsEditable)

            if c in (5, 6):
                txt = str(v or "").strip()
                if txt:
                    it.setToolTip(txt)

            self.tbl.setItem(r, c, it)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_app_style(app)
    w = MainWindow()
    w.show()

    sys.exit(app.exec())
