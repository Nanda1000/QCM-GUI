
        
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import threading
import time
from datetime import datetime
from functools import partial
from typing import Optional
from Acquire_data import NanoVNA



import pyqtgraph as pg
import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, QTimer, QSize, QAbstractTableModel, QModelIndex
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QFileDialog, QGroupBox, QSplitter, QWidget, QTableView, QLineEdit, QToolBar,
    QHBoxLayout, QMessageBox, QPushButton, QLabel, QVBoxLayout, QMainWindow,
    QApplication, QStatusBar, QFormLayout, QComboBox, QSizePolicy, QFrame
)
from Models.Butterworth import parameter as bvd_parameter, butterworth
from Models.Avrami import compute_X_t, fit as avrami_fit, formula as avrami_formula
from Models.Sauerbrey import sauerbrey
from Models.konazawa import konazawa



# ---------------------------
# Table model ()
# ---------------------------
class TableModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._data = df.reset_index(drop=True)

    def rowCount(self, parent=QModelIndex()):
        return int(self._data.shape[0])

    def columnCount(self, parent=QModelIndex()):
        return int(self._data.shape[1])

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iat[index.row(), index.column()]
            if pd.isna(value):
                return ""
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._data.columns[section])
        return str(section)

    def flags(self, index):
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def set_dataframe(self, df: pd.DataFrame):
        self.beginResetModel()
        self._data = df.reset_index(drop=True)
        self.endResetModel()

    def dataframe(self):
        return self._data


# ---------------------------
# Main application window
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NanoVNA Data Logger & Analysis")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)

        # state / models
        self.vna: Optional[NanoVNA] = None

        self.impedance = None
        self.freqs = np.linspace(1e6, 10e6, 201)
        self.is_dark = False
        


        # GUI build
        
        self._build_actions()
        self._build_toolbar()

        self._build_statusbar()
        self._build_main_layout()

        # QTimer for fallback polling (not active by default)
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(2000)
        self.poll_timer.timeout.connect(self.run_single_sweep)

        # minimal attempt to import pyvisa resources; non-fatal
        try:
            import pyvisa
            self.pyvisa_rm = pyvisa.ResourceManager()
        except Exception:
            self.pyvisa_rm = None

        # initial populate of device list
        QTimer.singleShot(50, self.rescan_ports)

    # ---------------------------
    # UI: actions & toolbar
    # ---------------------------
    def _build_actions(self):
        self.act_connect = QAction(QIcon("icons/connect.png"), "Connect", self)
        self.act_connect.triggered.connect(self.connect_to_instrument)

        self.act_rescan = QAction(QIcon("icons/rescan.png"), "Rescan", self)
        self.act_rescan.triggered.connect(self.rescan_ports)

        self.act_start = QAction(QIcon("icons/start.png"), "Start Logging", self)
        self.act_start.triggered.connect(self.start_logging_button)

        self.act_stop = QAction(QIcon("icons/stop.png"), "Stop Logging", self)
        self.act_stop.triggered.connect(self.stop_logging_button)

        self.act_export = QAction(QIcon("icons/export.png"), "Export CSV", self)
        self.act_export.triggered.connect(self.export_csv)

        self.act_toggle_theme = QAction(QIcon("icons/theme.png"), "Toggle Dark Mode", self)
        self.act_toggle_theme.triggered.connect(self.toggle_theme)

    def _build_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        toolbar.addAction(self.act_connect)
        toolbar.addAction(self.act_rescan)
        toolbar.addSeparator()
        toolbar.addAction(self.act_start)
        toolbar.addAction(self.act_stop)
        toolbar.addSeparator()
        toolbar.addAction(self.act_export)
        toolbar.addAction(self.act_toggle_theme)

 
        
            # Toolbar
        toolbar = QToolBar("Main toolbar")
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)

        # Menu
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction(QAction("New File", self))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Open File", self))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Save", self))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Save As", self))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Share as", self))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Print", self))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Close", self))
        file_menu.addMenu("Share")
        edit_menu = menu.addMenu("&Edit")
        view_menu = menu.addMenu("&View")
        self.button_action8 = QAction("Add Table", self)
        view_menu.addAction(self.button_action8)
        self.button_action8.triggered.connect(self.insert_button_clicked)
        self.act_toggle_theme = QAction("Toggle Dark Mode", self)
        self.act_toggle_theme.triggered.connect(self.toggle_theme)
        edit_menu.addAction(self.act_toggle_theme)
        self.act_export = QAction("Export CSV", self)
        self.act_export.triggered.connect(self.export_csv)
        view_menu.addAction(self.act_export)

    def _build_statusbar(self):
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        self.statusBar().showMessage("Ready")

    # ---------------------------
    # Main layout (left controls, right plot/table)
    # ---------------------------
    def _build_main_layout(self):
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main_splitter)

        # --- LEFT: controls ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(12)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # Metadata group
        meta_grp = QGroupBox("Metadata")
        meta_layout = QFormLayout()
        self.sample_name = QLineEdit()
        self.batch_number = QLineEdit()
        self.operator_name = QLineEdit()
        self.notes = QLineEdit()
        meta_layout.addRow("Sample:", self.sample_name)
        meta_layout.addRow("Batch No.:", self.batch_number)
        meta_layout.addRow("Operator:", self.operator_name)
        meta_layout.addRow("Notes:", self.notes)
        meta_grp.setLayout(meta_layout)
        left_layout.addWidget(meta_grp)

        # Acquisition group
        acq_grp = QGroupBox("Data Acquisition")
        acq_layout = QVBoxLayout()
        h_rescan = QHBoxLayout()
        self.combo_ports = QComboBox()
        self.combo_ports.setEditable(True)
        self.combo_ports.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        h_rescan.addWidget(QLabel("Instrument:"))
        h_rescan.addWidget(self.combo_ports)
        btn_rescan = QPushButton("Rescan")
        btn_rescan.clicked.connect(self.rescan_ports)
        h_rescan.addWidget(btn_rescan)
        acq_layout.addLayout(h_rescan)

        freq_form = QFormLayout()
        self.start_frequency = QLineEdit("1000000")
        self.end_frequency = QLineEdit("10000000")
        self.sweep_points = QLineEdit("201")
        freq_form.addRow("Start (Hz):", self.start_frequency)
        freq_form.addRow("Stop (Hz):", self.end_frequency)
        freq_form.addRow("Points:", self.sweep_points)
        acq_layout.addLayout(freq_form)

        btns = QHBoxLayout()
        self.btn_connect = QPushButton("Connect")
        self.btn_connect.clicked.connect(self.connect_to_instrument)
        self.btn_single = QPushButton("Single Sweep")
        self.btn_single.clicked.connect(self.run_single_sweep)
        btns.addWidget(self.btn_connect)
        btns.addWidget(self.btn_single)
        acq_layout.addLayout(btns)

        acq_grp.setLayout(acq_layout)
        left_layout.addWidget(acq_grp)

        # Logging controls
        control_grp = QGroupBox("Logging Controls")
        ctl_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Logging")
        self.btn_stop = QPushButton("Stop Logging")
        self.upload_button = QPushButton("Upload Data")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start_logging_button)
        self.btn_stop.clicked.connect(self.stop_logging_button)
        self.upload_button.clicked.connect(self.upload_button_clicked)
        ctl_layout.addWidget(self.btn_start)
        ctl_layout.addWidget(self.btn_stop)
        ctl_layout.addWidget(self.upload_button)
        control_grp.setLayout(ctl_layout)
        left_layout.addWidget(control_grp)

        # Analysis shortcuts
        analysis_grp = QGroupBox("Analysis Tools")
        an_layout = QVBoxLayout()
        self.btn_sauer = QPushButton("Sauerbrey / Konazawa")
        self.btn_sauer.clicked.connect(self.sauerbrey_konazawa)
        self.btn_cryst_dyn = QPushButton("Crystallization Dynamics")
        self.btn_cryst_dyn.clicked.connect(self.crystallizationdynamics)
        self.btn_cryst_kin = QPushButton("Crystallization Kinetics")
        self.btn_cryst_kin.clicked.connect(self.crystallizationkinetics)
        an_layout.addWidget(self.btn_sauer)
        an_layout.addWidget(self.btn_cryst_dyn)
        an_layout.addWidget(self.btn_cryst_kin)
        analysis_grp.setLayout(an_layout)
        left_layout.addWidget(analysis_grp)

        left_layout.addStretch()
        main_splitter.addWidget(left_widget)
        left_widget.setMinimumWidth(320)

        # --- RIGHT: table + plot (vertical) ---
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Table
        table_frame = QFrame()
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(4, 4, 4, 4)

        self.table_view = QTableView()
        self.data = pd.DataFrame(columns=["Time", "Frequency(Hz)", "Resistance(Ω)", "Phase"])
        self.table_model = TableModel(self.data)
        self.table_model.dataChanged.connect(self.update_plot)
        self.table_view.setModel(self.table_model)

        self.table_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        table_layout.addWidget(QLabel("Data Table"))
        table_layout.addWidget(self.table_view)
        right_splitter.addWidget(table_frame)

        # Plot
        plot_frame = QFrame()
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(4, 4, 4, 4)
        plot_layout.addWidget(QLabel("Resistance vs Frequency"))
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_curve = self.plot_widget.plot(self.freqs, np.zeros_like(self.freqs), pen='b')
        self.plot_widget.setLabel('left', 'Resistance (Ω)')
        self.plot_widget.setLabel('bottom', 'Frequency (Hz)')
        self.plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        plot_layout.addWidget(self.plot_widget)
        right_splitter.addWidget(plot_frame)

        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

    # ---------------------------
    # Utilities: safe notifications
    # ---------------------------
    def _notify(self, level: str, title: str, message: str):
        """Schedule a messagebox or statusbar text on main thread."""
        def _show():
            if level == "info":
                # short info: status + optional messagebox for explicit info
                self.statusBar().showMessage(f"{title}: {message}", 8000)
            elif level == "warning":
                QMessageBox.warning(self, title, message)
            elif level == "critical":
                QMessageBox.critical(self, title, message)
            else:
                QMessageBox.information(self, title, message)
        QTimer.singleShot(0, _show)

    # This callback is passed to NanoVNA to receive messages from worker threads
    def notify_callback(self, level, title, message):
        # ensure scheduled on GUI thread
        self._notify(level, title, message)

    # ---------------------------
    # Device scanning & connect
    # ---------------------------
    def rescan_ports(self):
        """Populate device combobox. Adds 'NanoVNA (Auto-detect)' entry."""
        self.combo_ports.clear()
        self.combo_ports.addItem("NanoVNA (Auto-detect)")
        # try pyvisa devices if available (best-effort)
        if self.pyvisa_rm:
            try:
                res = self.pyvisa_rm.list_resources()
                for r in res:
                    self.combo_ports.addItem(str(r))
            except Exception:
                # ignore errors from pyvisa
                pass
        self.statusBar().showMessage("Device list updated", 3000)

    def _connect_worker(self, port_hint: Optional[str] = None):
        """Background connect worker — uses NanoVNA.connect and does one quick scan."""
        try:
            # create NanoVNA instance if not present
            if self.vna is None:
                self.vna = NanoVNA(port=port_hint, notify_callback=self.notify_callback)

            ok = self.vna.connect()
            if not ok:
                self._notify("critical", "Connect", "Failed to connect to NanoVNA")
                return

            # do a quick single scan (in thread) to prime data and update GUI via callback
            try:
                start = float(self.start_frequency.text() or 1e6)
                stop = float(self.end_frequency.text() or 10e6)
                points = int(self.sweep_points.text() or 201)
            except Exception:
                start, stop, points = 1e6, 10e6, 201

            try:
                freqs, s11 = self.vna.scan(start, stop, points)
                impedance = self.vna.s11_to_impedance(s11)
                resistance = impedance.real
                data_package = {
                    "timestamp": pd.Timestamp.now(),
                    "scan_count": 0,
                    "frequencies": freqs,
                    "impedance": impedance,
                    "resistance": resistance,
                    "phase": np.angle(impedance, deg=True),
                    "s11": s11
                }
                # schedule UI update (use same path as acquisition)
                QTimer.singleShot(0, lambda pkg=data_package: self._on_vna_data(pkg))
                QTimer.singleShot(0, lambda: self.statusBar().showMessage("Connected to NanoVNA", 5000))
            except Exception as e:
                self._notify("warning", "Initial Scan", f"Connected but initial scan failed: {e}")

        except Exception as e:
            self._notify("critical", "Connect Error", str(e))

    def connect_to_instrument(self):
        """Public slot — triggered from UI. Spawns background thread to connect."""
        selected = self.combo_ports.currentText()
        # if user selected explicit visa resource, pass as hint (NanoVNA handler will auto-detect if None)
        port_hint = None
        if selected and selected.strip() and "NanoVNA" not in selected:
            port_hint = selected
        # spawn background connect to avoid freeze
        threading.Thread(target=self._connect_worker, args=(None if "Auto-detect" in selected else port_hint,), daemon=True).start()
        
    def insert_button_clicked(self):
        """Initializes table with empty rows and shows Insert/Delete buttons."""
        # Create empty DataFrame
        self.data = pd.DataFrame({
            "Timestamp": ["" for _ in range(10)],
            "Frequency(Hz)": ["" for _ in range(10)],
            "Resistance(Ω)": ["" for _ in range(10)],
            "Phase": ["" for _ in range(10)]
        })

        # Create model and connect to plot
        self.model = TableModel(self.data)
        self.model.dataChanged.connect(self.update_plot)
        self.table.setModel(self.model)

        # Create buttons only once
        if not hasattr(self, 'insert_btn'):
            self.insert_btn = QPushButton("Insert Row")
            self.insert_btn.clicked.connect(self.insert_row)
            self.delete_btn = QPushButton("Delete Row")
            self.delete_btn.clicked.connect(self.delete_row)

            # Add buttons below existing table in the right-side layout
            btn_layout = QHBoxLayout()
            btn_layout.addWidget(self.insert_btn)
            btn_layout.addWidget(self.delete_btn)

            # self.table is inside the table_frame from _build_main_layout()
            parent_layout = self.table.parentWidget().layout()
            parent_layout.addLayout(btn_layout)


    def insert_row(self):
        """Insert a new blank row at the selected position or at the end."""
        new_row = pd.DataFrame({
            "Timestamp": [""],
            "Frequency(Hz)": [""],
            "Resistance(Ω)": [""],
            "Phase": [""]
        })
        curr_row = self.table.currentIndex().row()
        if curr_row == -1:
            self.data = pd.concat([self.data, new_row], ignore_index=True)
        else:
            top = self.data.iloc[:curr_row + 1]
            bottom = self.data.iloc[curr_row + 1:]
            self.data = pd.concat([top, new_row, bottom], ignore_index=True)

        # Refresh model
        self.model = TableModel(self.data)
        self.model.dataChanged.connect(self.update_plot)
        self.table.setModel(self.model)

    def delete_row(self):
        """Delete the currently selected row."""
        curr_row = self.table.currentIndex().row()
        if curr_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a row to delete")
            return

        confirm = QMessageBox.question(
            self,
            "Confirm Deletion",
            "Are you sure you want to delete this row?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.data = self.data.drop(index=curr_row).reset_index(drop=True)
            self.model = TableModel(self.data)
            self.model.dataChanged.connect(self.update_plot)
            self.table.setModel(self.model)

    def upload_button_clicked(self):
        """Load CSV or Excel file into the table."""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "CSV (*.csv);;Excel (*.xlsx)"
        )
        if not file:
            return
        try:
            self.data = pd.read_csv(file) if file.endswith(".csv") else pd.read_excel(file)
            for col in ["Timestamp", "Frequency(Hz)", "Resistance(Ω)", "Phase"]:
                if col not in self.data.columns:
                    self.data[col] = ""
            self.model = TableModel(self.data)
            self.model.dataChanged.connect(self.update_plot)
            self.table.setModel(self.model)
            self.update_plot()
        except Exception as e:
            QMessageBox.warning(self, "File Error", str(e))

    # ---------------------------
    # Data handling (thread-safe)
    # ---------------------------
    def _on_vna_data(self, data_package: dict):
        """
        Called from background callback or initial scan thread.
        Must only perform light-weight operations and schedule heavy UI work on main thread.
        """
        def _process():
            try:
                freqs = np.asarray(data_package.get("frequencies", []))
                resistance = np.asarray(data_package.get("resistance", np.zeros_like(freqs)))
                phase = np.asarray(data_package.get("phase", np.angle(data_package.get("impedance", np.zeros_like(freqs)), deg=True)))

                # build new rows
                timestamp = data_package.get("timestamp", pd.Timestamp.now())
                rows = [{"Timestamp": timestamp, "Frequency(Hz)": float(f), "Resistance(Ω)": float(r), "Phase": float(p)}
                        for f, r, p in zip(freqs, resistance, phase)]

                if rows:
                    new_df = pd.DataFrame(rows)
                    self.data = pd.concat([self.data, new_df], ignore_index=True)
                    # update table model
                    self.table_model.set_dataframe(self.data)
                    self.table.setModel(self.table_model)
                    self.table_view.resizeColumnsToContents()
                    # update plot
                    self.update_plot()
                    self.statusBar().showMessage(f"Received scan #{data_package.get('scan_count', '?')} ({len(freqs)} pts)", 5000)
            except Exception as e:
                self._notify("warning", "Data Processing Error", str(e))

        # ensure runs on GUI thread
        QTimer.singleShot(0, _process)

    # ---------------------------
    # Single sweep
    # ---------------------------
    def run_single_sweep(self):
        """Synchronous single sweep wrapper — runs in GUI thread only if small; prefer background use."""
        # run in a background thread to keep UI snappy
        def worker():
            try:
                if self.vna is None or not getattr(self.vna, "is_connected", False):
                    self._notify("warning", "Connection", "Device not connected.")
                    return
                start = float(self.start_frequency.text() or 1e6)
                stop = float(self.end_frequency.text() or 10e6)
                points = int(self.sweep_points.text() or 201)
                freqs, s11 = self.vna.scan(start, stop, points)
                impedance = self.vna.s11_to_impedance(s11)
                resistance = impedance.real
                data_package = {
                    "timestamp": pd.Timestamp.now(),
                    "scan_count": 0,
                    "frequencies": freqs,
                    "impedance": impedance,
                    "resistance": resistance,
                    "phase": np.angle(impedance, deg=True),
                    "s11": s11
                }
                # push to UI
                self._on_vna_data(data_package)
            except Exception as e:
                self._notify("warning", "Sweep Error", str(e))

        threading.Thread(target=worker, daemon=True).start()

    # ---------------------------
    # Start / Stop continuous logging
    # ---------------------------
    def start_logging_button(self):
        """Start continuous logging: prefer NanoVNA.start_acquisition (threaded), else fallback to QTimer polling."""
        if self.vna is None or not getattr(self.vna, "is_connected", False):
            QMessageBox.warning(self, "Connection Error", "Please connect to the NanoVNA first.")
            return

        try:
            start = float(self.start_frequency.text() or 1e6)
            stop = float(self.end_frequency.text() or 10e6)
            points = int(self.sweep_points.text() or 201)
        except Exception:
            QMessageBox.warning(self, "Input Error", "Invalid sweep parameters.")
            return

        started = False
        try:
            # pass callback that will be invoked from worker thread — callback should not touch GUI directly
            started = self.vna.start_acquisition(start, stop, points, interval=2.0, callback=self.vna_callback_wrapper)
        except Exception as e:
            started = False
            self._notify("warning", "Acquisition", f"start_acquisition raised: {e}")

        if started:
            # thread-based acquisition started
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.act_start.setEnabled(False)
            self.act_stop.setEnabled(True)
            self.statusBar().showMessage("Threaded logging started...", 5000)
            return

        # fallback to QTimer polling
        try:
            if not self.poll_timer.isActive():
                self.poll_timer.start()
                self.btn_start.setEnabled(False)
                self.btn_stop.setEnabled(True)
                self.act_start.setEnabled(False)
                self.act_stop.setEnabled(True)
                self.statusBar().showMessage("Logging started (timer fallback)...", 5000)
        except Exception as e:
            QMessageBox.warning(self, "Logging Error", str(e))

    def vna_callback_wrapper(self, data_package):
        """
        This wrapper is passed to NanoVNA.start_acquisition.
        NanoVNA will call it from a worker thread — the wrapper must schedule GUI updates on the main thread.
        """
        # pass to our main handler (it marshals to GUI thread)
        try:
            self._on_vna_data(data_package)
        except Exception as e:
            # do not let exceptions propagate to worker thread
            self._notify("warning", "Callback Error", str(e))

    def stop_logging_button(self):
        """Stop threaded acquisition and timer fallback."""
        try:
            # stop NanoVNA threaded acquisition if active
            if self.vna and getattr(self.vna, "acquisition_active", False):
                try:
                    self.vna.stop_acquisition()
                except Exception:
                    pass

            # stop timer fallback
            if self.poll_timer.isActive():
                self.poll_timer.stop()

            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.act_start.setEnabled(True)
            self.act_stop.setEnabled(False)

            self.statusBar().showMessage("Logging stopped.", 5000)
        except Exception as e:
            QMessageBox.warning(self, "Stop Error", str(e))

    # ---------------------------
    # Plot update
    # ---------------------------
    def update_plot(self):
        """Update the Resistance vs Frequency plot from current self.data."""
        try:
            if self.data.empty or "Frequency(Hz)" not in self.data.columns or "Resistance(Ω)" not in self.data.columns:
                # clear plot
                self.plot_widget.clear()
                return

            x = pd.to_numeric(self.data["Frequency(Hz)"], errors='coerce')
            y = pd.to_numeric(self.data["Resistance(Ω)"], errors='coerce')
            mask = x.notnull() & y.notnull()
            if mask.sum() == 0:
                self.plot_widget.clear()
                return
            xvals = x[mask].astype(float).values
            yvals = y[mask].astype(float).values
            self.plot_widget.clear()
            self.plot_widget.plot(xvals, yvals, pen='r', symbol='o', symbolSize=5)
        except Exception as e:
            self._notify("warning", "Plot Error", str(e))

    # ---------------------------
    # Export
    # ---------------------------
    def export_csv(self):
        try:
            if self.data.empty:
                QMessageBox.information(self, "Export", "No data to export.")
                return
            fname, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV files (*.csv)")
            if not fname:
                return
            self.data.to_csv(fname, index=False)
            self.statusBar().showMessage(f"Saved {len(self.data)} rows to {fname}", 100000)
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    # ---------------------------
    # Theme toggle
    # ---------------------------
    def toggle_theme(self):
        self.is_dark = not self.is_dark
        if self.is_dark:
            # simple dark stylesheet
            self.setStyleSheet("""
                QWidget { background: #202020; color: #e0e0e0; }
                QLineEdit, QComboBox, QTableView { background: #2a2a2a; color: #e0e0e0; }
                QToolBar { background: #2a2a2a; }
                QStatusBar { background: #252525; color: #cfcfcf; }
            """)
            self.plot_widget.setBackground('#2a2a2a')
        else:
            self.setStyleSheet("")
            self.plot_widget.setBackground('w')

    # ---------------------------
    # Analysis windows & calculators
    # ---------------------------
    def sauerbrey_konazawa(self):
        """Open Sauerbrey & Konazawa dialog (simple implementation)."""
        dlg = QWidget()
        dlg.setWindowTitle("Sauerbrey & Konazawa")
        dlg.resize(700, 480)
        layout = QVBoxLayout(dlg)

        form = QFormLayout()
        f0 = QLineEdit("5000000")
        density = QLineEdit("2650")
        shear = QLineEdit("2.947e10")
        area = QLineEdit("1e-4")
        form.addRow("Resonant freq (Hz):", f0)
        form.addRow("Quartz density (kg/m³):", density)
        form.addRow("Shear modulus (Pa):", shear)
        form.addRow("Electrode area (m²):", area)

        calc_btn = QPushButton("Calculate Sauerbrey Mass")
        result = QLineEdit()
        result.setReadOnly(True)

        def _calc():
            try:
                ff0 = float(f0.text())
                dd = float(density.text())
                ss = float(shear.text())
                aa = float(area.text())
                # latest frequency from data
                if self.data.empty or "Frequency(Hz)" not in self.data.columns:
                    QMessageBox.warning(self, "Data Error", "No frequency data available.")
                    return
                ft = float(pd.to_numeric(self.data["Frequency(Hz)"], errors='coerce').dropna().iloc[-1])
                mass_change = sauerbrey(ff0, dd, ss, ft, aa)
                result.setText(f"{mass_change:.6e}")
            except Exception as e:
                QMessageBox.warning(self, "Calculation Error", str(e))

        calc_btn.clicked.connect(_calc)

        layout.addLayout(form)
        layout.addWidget(calc_btn)
        layout.addWidget(QLabel("Mass Change (kg):"))
        layout.addWidget(result)
        dlg.setLayout(layout)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()

    def crystallizationdynamics(self):
        """Open Crystallization Dynamics window with multi-plot + BVD params."""
        try:
            # Close if already open
            if hasattr(self, 'cryst_dyn_win') and self.cryst_dyn_win:
                self.cryst_dyn_win.close()

            # New window
            self.cryst_dyn_win = QWidget()
            self.cryst_dyn_win.setWindowTitle("Crystallization Dynamics")
            self.cryst_dyn_win.resize(1400, 900)
            layout = QVBoxLayout(self.cryst_dyn_win)

            # Toolbar for this window
            toolbar = QToolBar()
            toolbar.setIconSize(QSize(16, 16))
            act_view_table = QAction("View Table", self.cryst_dyn_win)
            act_view_table.triggered.connect(self.Viewtable)
            act_export = QAction("Export Table", self.cryst_dyn_win)
            act_export.triggered.connect(lambda: self._export_df(self.cryst_dyn_df, "cryst_dynamics.csv"))
            toolbar.addAction(act_view_table)
            toolbar.addAction(act_export)
            layout.addWidget(toolbar)

            splitter = QSplitter(Qt.Orientation.Horizontal)

            # Left plots
            left_widget = QWidget()
            left_layout = QVBoxLayout(left_widget)
            left_layout.setContentsMargins(0, 0, 0, 0)
            self.plot_resistance = pg.PlotWidget(title="Motional Resistance vs Time")
            self.plot_resistance.setLabel("left", "Rm (Ω)")
            self.plot_resistance.setLabel("bottom", "Time (s)")

            self.plot_frequency = pg.PlotWidget(title="Resonance Frequency vs Time")
            self.plot_frequency.setLabel("left", "Fs (Hz)")
            self.plot_frequency.setLabel("bottom", "Time (s)")

            self.plot_inductance = pg.PlotWidget(title="Motional Inductance vs Time")
            self.plot_inductance.setLabel("left", "Lm (H)")
            self.plot_inductance.setLabel("bottom", "Time (s)")

            self.plot_capacitance = pg.PlotWidget(title="Motional Capacitance vs Time")
            self.plot_capacitance.setLabel("left", "Cm (F)")
            self.plot_capacitance.setLabel("bottom", "Time (s)")

            for pw in (self.plot_resistance, self.plot_frequency, self.plot_inductance, self.plot_capacitance):
                left_layout.addWidget(pw)

            splitter.addWidget(left_widget)

            # Right controls
            right_widget = QWidget()
            right_layout = QVBoxLayout(right_widget)
            group_box = QGroupBox("Latest BVD Parameters")
            form = QFormLayout()
            self.rm_edit = QLineEdit(); self.rm_edit.setReadOnly(True)
            self.lm_edit = QLineEdit(); self.lm_edit.setReadOnly(True)
            self.cm_edit = QLineEdit(); self.cm_edit.setReadOnly(True)
            self.c0_edit = QLineEdit(); self.c0_edit.setReadOnly(True)
            self.f_edit = QLineEdit();  self.f_edit.setReadOnly(True)
            form.addRow("Fs (Hz):", self.f_edit)
            form.addRow("Rm (Ω):", self.rm_edit)
            form.addRow("Lm (H):", self.lm_edit)
            form.addRow("Cm (F):", self.cm_edit)
            form.addRow("C0 (F):", self.c0_edit)
            group_box.setLayout(form)
            right_layout.addWidget(group_box)
            right_layout.addStretch()
            splitter.addWidget(right_widget)

            layout.addWidget(splitter)
            self.cryst_dyn_win.show()

            self.plot_crystallization_data()

        except Exception as e:
            QMessageBox.warning(self, "Window Error", str(e))

    def plot_crystallization_data(self):
        """Extract sweep-by-sweep BVD params and update dynamics plots."""
        try:
            from Models.Butterworth import parameter  # local import to avoid crash if missing
        except ImportError:
            QMessageBox.warning(self, "Missing Model", "Butterworth model not found.")
            return

        try:
            if self.data.empty:
                QMessageBox.warning(self, "Data Error", "No data to analyze.")
                return

            points = int(self.sweep_points.text() or 201)
            if points <= 0:
                QMessageBox.warning(self, "Input Error", "Invalid sweep points.")
                return

            total_rows = len(self.data)
            if total_rows < points:
                QMessageBox.warning(self, "Data Error", "Insufficient data for one sweep.")
                return

            if total_rows % points != 0:
                QMessageBox.warning(self, "Data Error", "Data length not divisible by sweep points.")
                return

            num_sweeps = total_rows // points
            t_seconds, rm_values, fs_values, lm_values, cm_values = [], [], [], [], []

            for i in range(num_sweeps):
                block = self.data.iloc[i * points:(i + 1) * points]
                freqs = pd.to_numeric(block["Frequency(Hz)"], errors='coerce').dropna().values
                resist = pd.to_numeric(block["Resistance(Ω)"], errors='coerce').fillna(0).values
                if len(freqs) == 0:
                    continue
                impedance = resist + 0j
                try:
                    Rm, Lm, Cm, C0, fs = parameter(freqs, impedance, resist)
                    rm_values.append(Rm)
                    fs_values.append(fs)
                    lm_values.append(Lm)
                    cm_values.append(Cm)
                    t_seconds.append(i * 2)
                except Exception:
                    continue

            if not rm_values:
                QMessageBox.warning(self, "Processing Error", "No sweeps could be processed.")
                return

            # Save for table/export
            self.cryst_dyn_df = pd.DataFrame({
                "Time(s)": t_seconds,
                "Rm(Ω)": rm_values,
                "Lm(H)": lm_values,
                "Cm(F)": cm_values,
                "Fs(Hz)": fs_values
            })

            # Update latest params
            self.rm_edit.setText(f"{rm_values[-1]:.6f}")
            self.lm_edit.setText(f"{lm_values[-1]:.6e}")
            self.cm_edit.setText(f"{cm_values[-1]:.6e}")
            self.c0_edit.setText(f"{C0:.6e}" if 'C0' in locals() else "")
            self.f_edit.setText(f"{fs_values[-1]:.2f}")

            # Plot
            self.plot_resistance.clear()
            self.plot_resistance.plot(t_seconds, rm_values, pen='b', symbol='o')

            self.plot_frequency.clear()
            self.plot_frequency.plot(t_seconds, fs_values, pen='r', symbol='o')

            self.plot_inductance.clear()
            self.plot_inductance.plot(t_seconds, lm_values, pen='g', symbol='o')

            self.plot_capacitance.clear()
            self.plot_capacitance.plot(t_seconds, cm_values, pen='m', symbol='o')

        except Exception as e:
            QMessageBox.warning(self, "Plotting Error", str(e))

    def Viewtable(self):
        """Show computed dynamics table in a separate window."""
        try:
            if not hasattr(self, 'cryst_dyn_df') or self.cryst_dyn_df.empty:
                QMessageBox.warning(self, "No Data", "Run dynamics analysis first.")
                return
            table_win = QWidget()
            table_win.setWindowTitle("Crystallization Dynamics Table")
            table_win.resize(800, 600)
            layout = QVBoxLayout(table_win)
            model = TableModel(self.cryst_dyn_df)
            table = QTableView()
            table.setModel(model)
            table.resizeColumnsToContents()
            layout.addWidget(table)
            table_win.show()
        except Exception as e:
            QMessageBox.warning(self, "Table Error", str(e))

    def crystallizationkinetics(self):
        """Open Crystallization Kinetics (Avrami) window."""
        try:
            if hasattr(self, 'cryst_kin_win') and self.cryst_kin_win:
                self.cryst_kin_win.close()

            self.cryst_kin_win = QWidget()
            self.cryst_kin_win.setWindowTitle("Crystallization Kinetics (Avrami)")
            self.cryst_kin_win.resize(1000, 700)
            layout = QVBoxLayout(self.cryst_kin_win)

            splitter = QSplitter(Qt.Orientation.Horizontal)

            # Left plot
            left_widget = QWidget()
            left_layout = QVBoxLayout(left_widget)
            self.plot_crystallization_fraction = pg.PlotWidget(title="X(t) vs Time")
            self.plot_crystallization_fraction.setLabel("left", "X(t)")
            self.plot_crystallization_fraction.setLabel("bottom", "Time (s)")
            left_layout.addWidget(self.plot_crystallization_fraction)
            splitter.addWidget(left_widget)

            # Right params
            right_widget = QWidget()
            right_layout = QVBoxLayout(right_widget)
            group = QGroupBox("Avrami Fit Parameters")
            form = QFormLayout()
            self.f0_edit = QLineEdit(); self.f0_edit.setReadOnly(True)
            self.finf_edit = QLineEdit(); self.finf_edit.setReadOnly(True)
            self.k_edit = QLineEdit(); self.n_edit = QLineEdit()
            form.addRow("f₀:", self.f0_edit)
            form.addRow("f∞:", self.finf_edit)
            form.addRow("k:", self.k_edit)
            form.addRow("n:", self.n_edit)
            fit_btn = QPushButton("Fit Avrami")
            fit_btn.clicked.connect(self.plot_kinetics_data)
            form.addRow(fit_btn)
            group.setLayout(form)
            right_layout.addWidget(group)
            splitter.addWidget(right_widget)

            layout.addWidget(splitter)
            self.cryst_kin_win.show()

            self.plot_kinetics_data()

        except Exception as e:
            QMessageBox.warning(self, "Window Error", str(e))

    def plot_kinetics_data(self):
        """Compute and plot crystallization fraction + Avrami fit."""
        try:
            from Models.Avrami import compute_X_t, fit as avrami_fit, formula as avrami_formula
        except ImportError:
            QMessageBox.warning(self, "Missing Model", "Avrami model not found.")
            return

        try:
            if self.data.empty:
                QMessageBox.warning(self, "Data Error", "No data for kinetics.")
                return

            timestamps = pd.to_datetime(self.data["Timestamp"], errors="coerce")
            freqs = pd.to_numeric(self.data["Frequency(Hz)"], errors="coerce")
            mask = ~(timestamps.isna() | freqs.isna())
            if mask.sum() < 3:
                QMessageBox.warning(self, "Data Error", "Too few points for analysis.")
                return

            t_seconds = (timestamps[mask] - timestamps[mask].min()).dt.total_seconds().values
            fvals = freqs[mask].values
            f0, finf = fvals[0], fvals[-1]
            self.f0_edit.setText(f"{f0:.2f}")
            self.finf_edit.setText(f"{finf:.2f}")

            X_actual = compute_X_t(fvals, f0, finf)
            self.plot_crystallization_fraction.clear()
            self.plot_crystallization_fraction.plot(t_seconds, X_actual, pen='b', symbol='o')

            # Fit Avrami
            try:
                k, n = avrami_fit(t_seconds, fvals)
                self.k_edit.setText(f"{k:.3e}")
                self.n_edit.setText(f"{n:.3f}")
                X_fit = avrami_formula(k, n, t_seconds)
                self.plot_crystallization_fraction.plot(t_seconds, X_fit, pen='r')
            except Exception as e:
                QMessageBox.warning(self, "Fit Error", str(e))

        except Exception as e:
            QMessageBox.warning(self, "Plotting Error", str(e))

    def _export_df(self, df, default_name):
        """Utility: export a DataFrame to CSV."""
        try:
            if df is None or df.empty:
                QMessageBox.warning(self, "Export", "No data to export.")
                return
            fname, _ = QFileDialog.getSaveFileName(self, "Save CSV", default_name, "CSV files (*.csv)")
            if not fname:
                return
            df.to_csv(fname, index=False)
            self.statusBar().showMessage(f"Saved {len(df)} rows to {fname}", 6000)
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    # ---------------------------
    # Cleanup on close
    # ---------------------------
    def closeEvent(self, event):
        try:
            if self.poll_timer and self.poll_timer.isActive():
                self.poll_timer.stop()
            if self.vna:
                try:
                    self.vna.stop_acquisition()
                except Exception:
                    pass
                try:
                    self.vna.disconnect()
                except Exception:
                    pass
            event.accept()
        except Exception:
            event.accept()


# ---------------------------
# Run application
# ---------------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
