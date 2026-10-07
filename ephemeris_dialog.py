"""Qt dialog for generating JPL Horizons ephemerides."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from astropy.coordinates import SkyCoord
import astropy.units as u

import ephemeris
import planner_core as core


class EphemerisWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, request: ephemeris.EphemerisRequest):
        super().__init__()
        self.request = request

    def run(self):
        try:
            result = ephemeris.query_horizons_ephemeris(self.request)
            if not self.isInterruptionRequested():
                self.completed.emit(result)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(str(exc))


class EphemerisFinderWorker(QThread):
    completed = Signal(object, str)
    failed = Signal(str)

    def __init__(
        self,
        *,
        target_name: str,
        epoch_label: str,
        ra_deg: float,
        dec_deg: float,
        fov_w_arcmin: int,
        fov_h_arcmin: int,
        mode: str,
        roll_deg: float,
        flip_horizontal: bool,
        flip_vertical: bool,
    ):
        super().__init__()
        self.target_name = str(target_name)
        self.epoch_label = str(epoch_label)
        self.ra_deg = float(ra_deg)
        self.dec_deg = float(dec_deg)
        self.fov_w_arcmin = int(fov_w_arcmin)
        self.fov_h_arcmin = int(fov_h_arcmin)
        self.mode = str(mode)
        self.roll_deg = float(roll_deg)
        self.flip_horizontal = bool(flip_horizontal)
        self.flip_vertical = bool(flip_vertical)

    def run(self):
        try:
            coord = SkyCoord(
                self.ra_deg * u.deg,
                self.dec_deg * u.deg,
                frame="icrs",
            )
            data, wcs, survey_label = core.fetch_finder_raw(
                coord,
                self.fov_w_arcmin,
                self.mode,
                fov_h_arcmin=self.fov_h_arcmin,
            )
            if self.isInterruptionRequested():
                return
            if data is None or wcs is None:
                raise RuntimeError(
                    "No finder image was returned for this position and survey."
                )

            display_name = f"{self.target_name} — {self.epoch_label}"
            fig = core.render_finder_figure_from_data(
                coord,
                display_name,
                data,
                wcs,
                self.fov_w_arcmin,
                survey_label,
                self.roll_deg,
                fov_h_arcmin=self.fov_h_arcmin,
                flip_horizontal=self.flip_horizontal,
                flip_vertical=self.flip_vertical,
            )
            if not self.isInterruptionRequested():
                self.completed.emit(fig, survey_label)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(str(exc))


class EphemerisDialog(QDialog):
    TYPE_OPTIONS = (
        ("Automatic", None),
        ("Small body (asteroid/comet)", "smallbody"),
        ("Major body / satellite / spacecraft", None),
    )

    TABLE_COLUMNS = (
        ("Local time", "time_local"),
        ("RA", "ra_hms"),
        ("Dec", "dec_dms"),
        ("Alt (deg)", "alt_deg"),
        ("Az (deg)", "az_deg"),
        ("Airmass", "airmass"),
        ("V mag", "v_mag"),
        ("Sky motion (arcsec/min)", "sky_motion_arcsec_min"),
        ("Moon sep. (deg)", "moon_separation_deg"),
        ("Moon illum. (%)", "moon_illumination_pct"),
        ("Solar elong. (deg)", "solar_elongation_deg"),
    )

    def __init__(
        self,
        parent=None,
        *,
        site,
        planning_date,
        min_alt_deg: float,
        max_alt_deg: float,
        finder_fov_w_arcmin: int = 20,
        finder_fov_h_arcmin: int = 20,
        finder_mode: str = "DSS",
        finder_roll_deg: float = 0.0,
        finder_flip_horizontal: bool = False,
        finder_flip_vertical: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle("JPL Horizons Ephemeris Generator")
        self.resize(1180, 760)
        self.setMinimumSize(820, 560)

        self.site = site
        self.planning_date = planning_date
        self.min_alt_deg = float(min_alt_deg)
        self.max_alt_deg = float(max_alt_deg)
        self.finder_fov_w_arcmin = int(finder_fov_w_arcmin)
        self.finder_fov_h_arcmin = int(finder_fov_h_arcmin)
        self.finder_mode = str(finder_mode)
        self.finder_roll_deg = float(finder_roll_deg)
        self.finder_flip_horizontal = bool(finder_flip_horizontal)
        self.finder_flip_vertical = bool(finder_flip_vertical)
        self._worker = None
        self._finder_worker = None
        self._result = None

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        intro = QLabel(
            "Generate a topocentric JPL Horizons ephemeris for the planner's "
            "active observatory and observing night. Times are shown in the "
            "site timezone and cover 17:00–07:00."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        setup = QGroupBox("Ephemeris Request")
        form = QFormLayout(setup)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.target_edit = QLineEdit()
        self.target_edit.setPlaceholderText(
            "e.g. 433 Eros, 1 Ceres, Mars, C/2023 A3"
        )

        self.type_combo = QComboBox()
        for label, value in self.TYPE_OPTIONS:
            self.type_combo.addItem(label, value)

        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 60)
        self.step_spin.setValue(5)
        self.step_spin.setSuffix(" min")

        site_text = (
            f"{self.site.name} — "
            f"{self.site.lat:.6f}°, {self.site.lon:.6f}°, "
            f"{self.site.height_m:.0f} m"
        )
        self.site_label = QLabel(site_text)
        self.site_label.setWordWrap(True)

        self.date_label = QLabel(
            f"{self.planning_date.isoformat()} 17:00 → "
            f"{(self.planning_date + pd.Timedelta(days=1)).isoformat()} 07:00 "
            f"({self.site.timezone})"
        )
        self.date_label.setWordWrap(True)

        form.addRow("Target:", self.target_edit)
        form.addRow("Target type:", self.type_combo)
        form.addRow("Sampling:", self.step_spin)
        form.addRow("Observatory:", self.site_label)
        form.addRow("Night:", self.date_label)
        root.addWidget(setup)

        actions = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Ephemeris")
        self.generate_btn.clicked.connect(self.generate)
        self.generate_btn.setProperty("primaryButton", True)
        self.generate_btn.setCursor(Qt.PointingHandCursor)

        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.clicked.connect(self.export_csv)
        self.export_btn.setEnabled(False)

        actions.addWidget(self.generate_btn)
        actions.addWidget(self.export_btn)
        actions.addStretch(1)
        root.addLayout(actions)

        self.status_label = QLabel("Enter a Solar System target and generate an ephemeris.")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        table_box = QGroupBox("Ephemeris")
        table_layout = QVBoxLayout(table_box)
        self.table = QTableWidget(0, len(self.TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels([c[0] for c in self.TABLE_COLUMNS])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_epoch_selection_changed)
        self.table.setAlternatingRowColors(True)
        self.table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        header = self.table.horizontalHeader()
        for col in range(len(self.TABLE_COLUMNS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)

        table_tools = QHBoxLayout()
        self.copy_table_btn = QPushButton("Copy Ephemeris Table")
        self.copy_table_btn.clicked.connect(self.copy_table_image)
        self.copy_table_btn.setEnabled(False)
        table_tools.addWidget(self.copy_table_btn)
        table_tools.addStretch(1)
        table_layout.addLayout(table_tools)
        table_layout.addWidget(self.table)
        splitter.addWidget(table_box)

        self.plot_tabs = QTabWidget()

        altitude_page = QWidget()
        altitude_layout = QVBoxLayout(altitude_page)
        altitude_layout.setContentsMargins(6, 6, 6, 6)
        altitude_tools = QHBoxLayout()
        self.copy_altitude_btn = QPushButton("Copy Altitude Plot")
        self.copy_altitude_btn.clicked.connect(self.copy_altitude_plot)
        self.copy_altitude_btn.setEnabled(False)
        altitude_tools.addWidget(self.copy_altitude_btn)
        altitude_tools.addStretch(1)
        altitude_layout.addLayout(altitude_tools)
        self.canvas = FigureCanvas(self._empty_figure())
        altitude_layout.addWidget(self.canvas, 1)
        self.plot_tabs.addTab(altitude_page, "Altitude")

        finder_page = QWidget()
        finder_layout = QVBoxLayout(finder_page)
        finder_layout.setContentsMargins(6, 6, 6, 6)
        finder_tools = QHBoxLayout()
        self.finder_epoch_label = QLabel(
            "Select an ephemeris row to choose the finder-chart epoch."
        )
        self.finder_epoch_label.setWordWrap(True)
        self.generate_finder_btn = QPushButton("Generate Finder Chart")
        self.generate_finder_btn.clicked.connect(self.generate_finder_chart)
        self.generate_finder_btn.setEnabled(False)
        self.copy_finder_btn = QPushButton("Copy Finder Chart")
        self.copy_finder_btn.clicked.connect(self.copy_finder_chart)
        self.copy_finder_btn.setEnabled(False)
        finder_tools.addWidget(self.finder_epoch_label, 1)
        finder_tools.addWidget(self.generate_finder_btn)
        finder_tools.addWidget(self.copy_finder_btn)
        finder_layout.addLayout(finder_tools)
        self.finder_canvas = FigureCanvas(self._empty_finder_figure())
        finder_layout.addWidget(self.finder_canvas, 1)
        self.plot_tabs.addTab(finder_page, "Finder Chart")

        splitter.addWidget(self.plot_tabs)
        splitter.setSizes([390, 320])

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_row.addWidget(close_btn)
        root.addLayout(close_row)

    @staticmethod
    def _empty_finder_figure():
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.text(
            0.5,
            0.5,
            "Select an ephemeris row, then generate a finder chart.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    @staticmethod
    def _empty_figure():
        fig, ax = plt.subplots(figsize=(8.5, 3.4))
        ax.text(
            0.5, 0.5,
            "Altitude track will appear after an ephemeris is generated.",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    def _request(self):
        id_type = self.type_combo.currentData()
        return ephemeris.EphemerisRequest(
            target=self.target_edit.text().strip(),
            planning_date=self.planning_date,
            site_lat_deg=self.site.lat,
            site_lon_deg=self.site.lon,
            site_elevation_m=self.site.height_m,
            timezone_name=self.site.timezone,
            step_minutes=int(self.step_spin.value()),
            id_type=id_type,
        )

    def generate(self):
        if self._worker is not None and self._worker.isRunning():
            return

        try:
            request = self._request()
            request.validate()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Ephemeris Request", str(exc))
            return

        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("Querying JPL Horizons…")
        self.export_btn.setEnabled(False)
        self.copy_table_btn.setEnabled(False)
        self.copy_altitude_btn.setEnabled(False)
        self.generate_finder_btn.setEnabled(False)
        self.copy_finder_btn.setEnabled(False)
        self.status_label.setText(
            f"Querying JPL Horizons for {request.target!r}. "
            "This requires an internet connection."
        )

        worker = EphemerisWorker(request)
        self._worker = worker
        worker.completed.connect(self._query_finished)
        worker.failed.connect(self._query_failed)
        worker.finished.connect(self._worker_finished)
        worker.start()

    def _worker_finished(self):
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("Generate Ephemeris")
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()

    def _query_failed(self, message: str):
        self.status_label.setText("Horizons query failed.")
        QMessageBox.warning(
            self,
            "JPL Horizons Query Failed",
            "RHO Planner could not generate this ephemeris.\n\n"
            f"{message}\n\n"
            "If the target name is ambiguous, try a numeric designation and "
            "select the appropriate target type.",
        )

    def _query_finished(self, frame):
        self._result = frame.copy()
        self._populate_table(frame)
        self._set_plot(self._build_altitude_figure(frame))
        self.export_btn.setEnabled(True)
        self.copy_table_btn.setEnabled(True)
        self.copy_altitude_btn.setEnabled(True)
        self._select_default_epoch()

        windows = ephemeris.altitude_windows(
            frame,
            self.min_alt_deg,
            self.max_alt_deg,
        )
        target = str(frame.iloc[0]["target"]) if len(frame) else self.target_edit.text()
        if windows:
            window_text = "; ".join(
                f"{a.strftime('%H:%M')}–{b.strftime('%H:%M')}"
                for a, b in windows
            )
            self.status_label.setText(
                f"{target}: {len(frame)} samples. Within the planner altitude "
                f"limits ({self.min_alt_deg:.0f}°–{self.max_alt_deg:.0f}°): "
                f"{window_text}."
            )
        else:
            self.status_label.setText(
                f"{target}: {len(frame)} samples. The target does not enter "
                f"the planner altitude range of {self.min_alt_deg:.0f}°–"
                f"{self.max_alt_deg:.0f}° during this night."
            )

    @staticmethod
    def _display_value(column: str, value) -> str:
        if column == "time_local":
            try:
                return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
            except Exception:
                return str(value)
        if column in {"ra_hms", "dec_dms"}:
            return str(value)
        try:
            number = float(value)
            if not np.isfinite(number):
                return "—"
            if column == "v_mag":
                return f"{number:.2f}"
            if column in {"airmass"}:
                return f"{number:.3f}"
            if "motion" in column:
                return f"{number:.3f}"
            return f"{number:.2f}"
        except Exception:
            text = str(value).strip()
            return text if text else "—"

    def _populate_table(self, frame):
        self.table.setRowCount(len(frame))
        for r, (_, row) in enumerate(frame.iterrows()):
            for c, (_, column) in enumerate(self.TABLE_COLUMNS):
                value = row.get(column, "")
                item = QTableWidgetItem(self._display_value(column, value))
                self.table.setItem(r, c, item)


    def _select_default_epoch(self):
        if self._result is None or self._result.empty:
            return

        alt = pd.to_numeric(
            self._result["alt_deg"], errors="coerce"
        ).to_numpy(dtype=float)
        finite = np.isfinite(alt)
        allowed = (
            finite
            & (alt >= self.min_alt_deg)
            & (alt <= self.max_alt_deg)
        )
        candidates = np.flatnonzero(allowed)
        if len(candidates):
            row = int(candidates[np.argmax(alt[candidates])])
        else:
            candidates = np.flatnonzero(finite)
            row = int(candidates[np.argmax(alt[candidates])]) if len(candidates) else 0

        row = max(0, min(row, self.table.rowCount() - 1))
        self.table.setCurrentCell(row, 0)
        self.table.selectRow(row)
        item = self.table.item(row, 0)
        if item is not None:
            self.table.scrollToItem(item)

    def _selected_row(self):
        if self._result is None or self._result.empty:
            return None
        row = self.table.currentRow()
        if row < 0 or row >= len(self._result):
            return None
        return self._result.iloc[row]

    def _on_epoch_selection_changed(self):
        row = self._selected_row()
        if row is None:
            self.finder_epoch_label.setText(
                "Select an ephemeris row to choose the finder-chart epoch."
            )
            self.generate_finder_btn.setEnabled(False)
            return

        timestamp = pd.Timestamp(row["time_local"])
        ra = str(row.get("ra_hms", ""))
        dec = str(row.get("dec_dms", ""))
        alt = self._display_value("alt_deg", row.get("alt_deg", np.nan))
        self.finder_epoch_label.setText(
            f"Finder epoch: {timestamp.strftime('%Y-%m-%d %H:%M %Z')}  |  "
            f"RA {ra}  Dec {dec}  |  Alt {alt}°"
        )
        self.generate_finder_btn.setEnabled(True)

    def generate_finder_chart(self):
        if self._finder_worker is not None and self._finder_worker.isRunning():
            return

        row = self._selected_row()
        if row is None:
            QMessageBox.information(
                self,
                "Select an Epoch",
                "Select an ephemeris row before generating a finder chart.",
            )
            return

        try:
            ra_deg = float(row["ra_deg"])
            dec_deg = float(row["dec_deg"])
            if not np.isfinite(ra_deg) or not np.isfinite(dec_deg):
                raise ValueError("The selected ephemeris row has no usable RA/Dec.")
        except Exception as exc:
            QMessageBox.warning(self, "Finder Chart", str(exc))
            return

        timestamp = pd.Timestamp(row["time_local"])
        epoch_label = timestamp.strftime("%Y-%m-%d %H:%M %Z")
        target = str(row.get("target", "")).strip() or self.target_edit.text().strip()

        self.generate_finder_btn.setEnabled(False)
        self.generate_finder_btn.setText("Loading Finder…")
        self.copy_finder_btn.setEnabled(False)
        self.finder_epoch_label.setText(
            f"Loading {self.finder_mode} finder for {target} at {epoch_label}…"
        )

        worker = EphemerisFinderWorker(
            target_name=target,
            epoch_label=epoch_label,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            fov_w_arcmin=self.finder_fov_w_arcmin,
            fov_h_arcmin=self.finder_fov_h_arcmin,
            mode=self.finder_mode,
            roll_deg=self.finder_roll_deg,
            flip_horizontal=self.finder_flip_horizontal,
            flip_vertical=self.finder_flip_vertical,
        )
        self._finder_worker = worker
        worker.completed.connect(
            lambda fig, survey: self._finder_finished(
                fig, survey, target, epoch_label
            )
        )
        worker.failed.connect(self._finder_failed)
        worker.finished.connect(self._finder_worker_finished)
        worker.start()

    def _finder_finished(self, figure, survey_label: str, target: str, epoch_label: str):
        self._set_finder_plot(figure)
        self.copy_finder_btn.setEnabled(True)
        self.plot_tabs.setCurrentIndex(1)
        self.finder_epoch_label.setText(
            f"{target} at {epoch_label} — {survey_label}; "
            f"FOV {self.finder_fov_w_arcmin}′ × {self.finder_fov_h_arcmin}′"
        )

    def _finder_failed(self, message: str):
        QMessageBox.warning(
            self,
            "Finder Chart Failed",
            "RHO Planner could not generate the finder chart.\n\n"
            f"{message}",
        )
        self._on_epoch_selection_changed()

    def _finder_worker_finished(self):
        self.generate_finder_btn.setEnabled(self._selected_row() is not None)
        self.generate_finder_btn.setText("Generate Finder Chart")
        worker = self._finder_worker
        self._finder_worker = None
        if worker is not None:
            worker.deleteLater()

    def _set_finder_plot(self, figure):
        parent = self.finder_canvas.parentWidget()
        layout = parent.layout()
        layout.removeWidget(self.finder_canvas)
        try:
            plt.close(self.finder_canvas.figure)
        except Exception:
            pass
        self.finder_canvas.setParent(None)
        self.finder_canvas = FigureCanvas(figure)
        layout.addWidget(self.finder_canvas, 1)

    def _copy_figure(self, figure, success_message: str):
        try:
            buf = BytesIO()
            figure.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            image = QImage.fromData(buf.getvalue(), "PNG")
            if image.isNull():
                raise RuntimeError("Failed to create clipboard image.")
            QGuiApplication.clipboard().setPixmap(QPixmap.fromImage(image))
            self.status_label.setText(success_message)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Clipboard Error",
                f"Could not copy image.\n\n{exc}",
            )

    def copy_table_image(self):
        if self._result is None or self._result.empty:
            return
        try:
            pixmap = self.table.grab()
            if pixmap.isNull():
                raise RuntimeError("Failed to capture the ephemeris table.")
            QGuiApplication.clipboard().setPixmap(pixmap)
            self.status_label.setText(
                "Visible ephemeris table copied to the clipboard."
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Clipboard Error",
                f"Could not copy the ephemeris table.\n\n{exc}",
            )

    def copy_altitude_plot(self):
        self._copy_figure(
            self.canvas.figure,
            "Altitude plot copied to the clipboard.",
        )

    def copy_finder_chart(self):
        if not self.copy_finder_btn.isEnabled():
            return
        self._copy_figure(
            self.finder_canvas.figure,
            "Finder chart copied to the clipboard.",
        )

    def _build_altitude_figure(self, frame):
        fig, ax = plt.subplots(figsize=(8.5, 3.4))
        times = pd.DatetimeIndex(frame["time_local"])
        alt = pd.to_numeric(frame["alt_deg"], errors="coerce").to_numpy(dtype=float)

        ax.plot(times.to_pydatetime(), alt, linewidth=2)
        ax.axhline(
            self.min_alt_deg,
            linestyle="--",
            linewidth=1,
            label=f"Min altitude {self.min_alt_deg:.0f}°",
        )
        ax.axhline(
            self.max_alt_deg,
            linestyle="--",
            linewidth=1,
            label=f"Max altitude {self.max_alt_deg:.0f}°",
        )
        ax.fill_between(
            times.to_pydatetime(),
            self.min_alt_deg,
            self.max_alt_deg,
            alpha=0.08,
        )
        ax.set_ylim(0, 90)
        ax.set_ylabel("Altitude (deg)")
        ax.set_xlabel(f"Local time ({self.site.timezone})")
        target = str(frame.iloc[0]["target"]) if len(frame) else "Target"
        ax.set_title(f"{target} — topocentric altitude")
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M", tz=ZoneInfo(self.site.timezone))
        )
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.autofmt_xdate(rotation=0)
        fig.tight_layout()
        return fig

    def _set_plot(self, figure):
        parent = self.canvas.parentWidget()
        layout = parent.layout()
        layout.removeWidget(self.canvas)
        try:
            plt.close(self.canvas.figure)
        except Exception:
            pass
        self.canvas.setParent(None)
        self.canvas = FigureCanvas(figure)
        layout.addWidget(self.canvas)

    def export_csv(self):
        if self._result is None or self._result.empty:
            QMessageBox.information(self, "No Ephemeris", "Generate an ephemeris first.")
            return

        target = str(self._result.iloc[0]["target"]).strip() or "ephemeris"
        safe_target = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in target
        ).strip("_")
        suggested = f"{safe_target}_{self.planning_date.isoformat()}_ephemeris.csv"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Ephemeris CSV",
            suggested,
            "CSV files (*.csv)",
        )
        if not path:
            return

        export = self._result.copy()
        export["time_local"] = pd.DatetimeIndex(export["time_local"]).astype(str)
        export["time_utc"] = pd.DatetimeIndex(export["time_utc"]).astype(str)
        export.to_csv(Path(path), index=False)
        self.status_label.setText(f"Exported ephemeris to {path}")

    @staticmethod
    def _stop_worker(worker):
        if worker is None or not worker.isRunning():
            return
        worker.requestInterruption()
        if not worker.wait(1200):
            worker.terminate()
            worker.wait(500)

    def closeEvent(self, event):
        self._stop_worker(self._worker)
        self._stop_worker(self._finder_worker)
        self._worker = None
        self._finder_worker = None
        event.accept()
