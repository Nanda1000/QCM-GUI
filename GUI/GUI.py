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
import serial.tools.list_ports
import pyqtgraph as pg
import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, QTimer, QSize, QAbstractTableModel, QModelIndex, pyqtSignal, QObject
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
# Signal handler for thread-safe communication
# ---------------------------
class DataSignalHandler(QObject):
    data_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str, str)
    status_update = pyqtSignal(str)


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
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
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
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        col = self._data.columns[index.column()]
        # convert empty string to NA
        if value == "":
            val = pd.NA
        else:
            # try to coerce numeric when appropriate
            try:
                # if column dtype is numeric or value looks numeric, convert
                val = float(value)
            except Exception:
                val = value
        self._data.iat[index.row(), index.column()] = val
        # notify views
        self.dataChanged.emit(index, index)
        return True

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
        self.acquisition_thread: Optional[threading.Thread] = None
        self.acquisition_active = False
        self.acquisition_stop_flag = threading.Event()

        self.impedance = None
        self.freqs = np.linspace(1e6, 10e6, 201)
        self.is_dark = False
        
        # Signal handler for thread-safe communication
        self.signal_handler = DataSignalHandler()
        self.signal_handler.data_received.connect(self._on_vna_data)
        self.signal_handler.error_occurred.connect(self._on_error)
        self.signal_handler.status_update.connect(self._on_status_update)

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
    # Helper method to check connection status
    # ---------------------------
    def _is_vna_connected(self):
        """Check if VNA is connected by checking serial port status."""
        return (self.vna is not None and 
                self.vna.ser is not None and 
                self.vna.ser.is_open)

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
        self.data = pd.DataFrame(columns=["Timestamp", "Frequency(Hz)", "Resistance(Ω)", "Reactance(Ω)", "Magnitude(Ω)", "Phase(°)"])
        self.table_model = TableModel(self.data)
        self.table_model.dataChanged.connect(self.update_plot)
        self.table_view.setModel(self.table_model)

        self.table_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        table_layout.addWidget(QLabel("Data Table"))
        table_layout.addWidget(self.table_view)
        # Add Insert/Delete buttons
        btn_layout = QHBoxLayout()
        self.insert_btn = QPushButton("Insert Row")
        self.insert_btn.clicked.connect(self.insert_row)
        self.delete_btn = QPushButton("Delete Row")
        self.delete_btn.clicked.connect(self.delete_row)
        btn_layout.addWidget(self.insert_btn)
        btn_layout.addWidget(self.delete_btn)
        table_layout.addLayout(btn_layout)

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
    # Thread-safe signal handlers
    # ---------------------------
    def _on_vna_data(self, data_package: dict):
        """Handle data received from NanoVNA in main thread."""
        try:
            timestamp = data_package.get("timestamp", pd.Timestamp.now())
            freqs = data_package.get("frequencies", np.array([]))
            impedance = data_package.get("impedance", np.array([]))
            
            if len(freqs) == 0 or len(impedance) == 0:
                return
                
            resistance = impedance.real
            reactance = impedance.imag
            magnitude = np.abs(impedance)
            phase = np.angle(impedance, deg=True)
            
            # Create new rows for the data table
            rows = []
            for f, r, x, m, p in zip(freqs, resistance, reactance, magnitude, phase):
                rows.append({
                    "Timestamp": timestamp,
                    "Frequency(Hz)": float(f),
                    "Resistance(Ω)": float(r),
                    "Reactance(Ω)": float(x),
                    "Magnitude(Ω)": float(m),
                    "Phase(°)": float(p)
                })
            
            if rows:
                new_df = pd.DataFrame(rows)
                self.data = pd.concat([self.data, new_df], ignore_index=True)
                self.table_model.set_dataframe(self.data)
                self.table_view.resizeColumnsToContents()
                self.update_plot()
                
                scan_count = data_package.get("scan_count", "?")
                self.statusBar().showMessage(f"Received scan #{scan_count} ({len(freqs)} points)", 3000)
                
        except Exception as e:
            QMessageBox.warning(self, "Data Processing Error", str(e))

    def _on_error(self, title: str, message: str):
        """Handle errors from background threads."""
        QMessageBox.warning(self, title, message)

    def _on_status_update(self, message: str):
        """Handle status updates from background threads."""
        self.statusBar().showMessage(message, 5000)

    # ---------------------------
    # Device scanning & connect
    # ---------------------------
    def rescan_ports(self):
        """Populate device combobox with available serial ports."""
        self.combo_ports.clear()
        self.combo_ports.addItem("NanoVNA (Auto-detect)")
        
        # Add available serial ports - FIXED FORMAT
        try:
            ports = list(serial.tools.list_ports.comports())
            for port in ports:
                # Store port info in a way that's easy to extract later
                port_desc = f"{port.device} ({port.description or 'Unknown'})"
                self.combo_ports.addItem(port_desc)
        except Exception as e:
            print(f"Error scanning ports: {e}")
        
        # Try pyvisa devices if available (best-effort)
        if self.pyvisa_rm:
            try:
                res = self.pyvisa_rm.list_resources()
                for r in res:
                    self.combo_ports.addItem(str(r))
            except Exception:
                pass
                
        self.statusBar().showMessage("Device list updated", 3000)

    def connect_to_instrument(self):
        """Connect to the selected NanoVNA device."""
        selected = self.combo_ports.currentText()
        
        # Extract port from selection
        port = None
        if selected and "Auto-detect" not in selected:
            # Extract port device name from "COM3 (Description)" format
            if "(" in selected and selected.count("(") == 1:
                port = selected.split("(")[0].strip()
            else:
                # If no parentheses, assume the whole string is the port
                port = selected.strip()
        
        def connect_worker():
            try:
                if port is None:
                    # Auto-detect mode - scan available ports and test each one
                    self.signal_handler.status_update.emit("Auto-detecting NanoVNA...")
                    ports = list(serial.tools.list_ports.comports())
                    
                    connected = False
                    for port_info in ports:
                        test_port = port_info.device
                        self.signal_handler.status_update.emit(f"Testing {test_port}...")
                        
                        try:
                            # Create NanoVNA instance with specific port
                            test_vna = NanoVNA(port=test_port)
                            if test_vna.test_connection(test_port):
                                # Found a working connection
                                self.vna = test_vna
                                self.vna.port = test_port
                                if self.vna.connect():
                                    self.signal_handler.status_update.emit(f"Connected to NanoVNA on {test_port}")
                                    connected = True
                                    break
                        except Exception:
                            continue  # Try next port
                    
                    if not connected:
                        self.signal_handler.error_occurred.emit("Auto-detect Failed", "No NanoVNA found on any port")
                        return
                else:
                    # Specific port selected
                    self.signal_handler.status_update.emit(f"Connecting to {port}...")
                    self.vna = NanoVNA(port=port)
                    
                    if not self.vna.connect():
                        self.signal_handler.error_occurred.emit("Connection Failed", f"Could not connect to NanoVNA on {port}")
                        return
                    
                    self.signal_handler.status_update.emit(f"Connected to NanoVNA on {port}")
                
                # Test connection with initial sweep
                try:
                    start = float(self.start_frequency.text() or "1000000")
                    stop = float(self.end_frequency.text() or "10000000") 
                    points = int(self.sweep_points.text() or "201")
                    
                    if self.vna.sweep(start, stop, points):
                        result = self.vna.acquire()
                        
                        if result[0] is not None:
                            freqs, s11_values, impedances, resistance, reactance, magnitude, phase = result
                            
                            data_package = {
                                "timestamp": pd.Timestamp.now(),
                                "scan_count": 0,
                                "frequencies": freqs,
                                "impedance": impedances
                            }
                            self.signal_handler.data_received.emit(data_package)
                            self.signal_handler.status_update.emit("Initial sweep completed successfully")
                        else:
                            self.signal_handler.error_occurred.emit("Connection Warning", "Connected but no data received")
                    else:
                        self.signal_handler.error_occurred.emit("Sweep Failed", "Could not set up initial sweep")
                        
                except Exception as e:
                    self.signal_handler.error_occurred.emit("Initial Sweep Failed", str(e))
                    
            except Exception as e:
                self.signal_handler.error_occurred.emit("Connection Error", str(e))
        
        # Run connection in background thread
        threading.Thread(target=connect_worker, daemon=True).start()
        self.statusBar().showMessage("Connecting to NanoVNA...", 2000)
    # ---------------------------
    # Data acquisition
    # ---------------------------
    def run_single_sweep(self):
        """Perform a single sweep measurement."""
        if not self._is_vna_connected():
            QMessageBox.warning(self, "Connection Error", "Please connect to NanoVNA first")
            return
        
        def sweep_worker():
            try:
                start = float(self.start_frequency.text() or "1000000")
                stop = float(self.end_frequency.text() or "10000000")
                points = int(self.sweep_points.text() or "201")
                
                # Perform sweep and acquire data
                if self.vna.sweep(start, stop, points):
                    result = self.vna.acquire()
                    
                    if result[0] is not None:  # Check if frequencies are available
                        freqs, s11_values, impedances, resistance, reactance, magnitude, phase = result
                        
                        data_package = {
                            "timestamp": pd.Timestamp.now(),
                            "scan_count": 1,
                            "frequencies": freqs,
                            "impedance": impedances
                        }
                        self.signal_handler.data_received.emit(data_package)
                    else:
                        self.signal_handler.error_occurred.emit("Sweep Error", "No data received from sweep")
                else:
                    self.signal_handler.error_occurred.emit("Sweep Error", "Failed to set up sweep")
                    
            except Exception as e:
                self.signal_handler.error_occurred.emit("Sweep Error", str(e))
        
        threading.Thread(target=sweep_worker, daemon=True).start()
        self.statusBar().showMessage("Performing single sweep...", 2000)

    def start_logging_button(self):
        """Start continuous data logging."""
        if not self._is_vna_connected():
            QMessageBox.warning(self, "Connection Error", "Please connect to NanoVNA first")
            return
        
        if self.acquisition_active:
            return
        
        try:
            start = float(self.start_frequency.text() or "1000000")
            stop = float(self.end_frequency.text() or "10000000")
            points = int(self.sweep_points.text() or "201")
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Invalid sweep parameters")
            return
        
        # Start acquisition thread
        self.acquisition_active = True
        self.acquisition_stop_flag.clear()
        self.acquisition_thread = threading.Thread(
            target=self._acquisition_worker, 
            args=(start, stop, points), 
            daemon=True
        )
        self.acquisition_thread.start()
        
        # Update UI
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.act_start.setEnabled(False)
        self.act_stop.setEnabled(True)
        
        self.statusBar().showMessage("Continuous logging started", 3000)

    def _acquisition_worker(self, start_freq, stop_freq, points):
        """Background worker for continuous data acquisition."""
        scan_count = 0
        interval = 2.0  # seconds between scans
        
        while self.acquisition_active and not self.acquisition_stop_flag.is_set():
            try:
                # Perform sweep and acquire data
                if self.vna.sweep(start_freq, stop_freq, points):
                    result = self.vna.acquire()
                    
                    if result[0] is not None:  # Check if frequencies are available
                        freqs, s11_values, impedances, resistance, reactance, magnitude, phase = result
                        scan_count += 1
                        
                        data_package = {
                            "timestamp": pd.Timestamp.now(),
                            "scan_count": scan_count,
                            "frequencies": freqs,
                            "impedance": impedances
                        }
                        self.signal_handler.data_received.emit(data_package)
                    else:
                        self.signal_handler.error_occurred.emit("Acquisition Error", "No data received from sweep")
                else:
                    self.signal_handler.error_occurred.emit("Acquisition Error", "Failed to set up sweep")
                
                # Wait for next scan or check stop flag
                for _ in range(int(interval * 10)):  # Check stop flag every 0.1 seconds
                    if self.acquisition_stop_flag.is_set():
                        break
                    time.sleep(0.1)
                    
            except Exception as e:
                self.signal_handler.error_occurred.emit("Acquisition Error", str(e))
                break
        
        self.acquisition_active = False

    def stop_logging_button(self):
        """Stop continuous data logging."""
        if not self.acquisition_active:
            return
        
        self.acquisition_active = False
        self.acquisition_stop_flag.set()
        
        # Wait for thread to finish (with timeout)
        if self.acquisition_thread and self.acquisition_thread.is_alive():
            self.acquisition_thread.join(timeout=2.0)
        
        # Update UI
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.act_start.setEnabled(True)
        self.act_stop.setEnabled(False)
        
        self.statusBar().showMessage("Logging stopped", 3000)

    # ---------------------------
    # Table management
    # ---------------------------
    def insert_button_clicked(self):
        """Initialize table with empty rows."""
        self.data = pd.DataFrame({
            "Timestamp": [pd.NaT for _ in range(10)],
            "Frequency(Hz)": [np.nan for _ in range(10)],
            "Resistance(Ω)": [np.nan for _ in range(10)],
            "Reactance(Ω)": [np.nan for _ in range(10)],
            "Magnitude(Ω)": [np.nan for _ in range(10)],
            "Phase(°)": [np.nan for _ in range(10)]
        })
        
        self.table_model = TableModel(self.data)
        self.table_model.dataChanged.connect(self.update_plot)
        self.table_view.setModel(self.table_model)
        self.table_view.resizeColumnsToContents()
        self.update_plot()

    def insert_row(self):
        """Insert a new blank row."""
        new_row = pd.DataFrame({
            "Timestamp": [pd.Timestamp.now()],
            "Frequency(Hz)": [np.nan],
            "Resistance(Ω)": [np.nan],
            "Reactance(Ω)": [np.nan],
            "Magnitude(Ω)": [np.nan],
            "Phase(°)": [np.nan]
        })
        
        curr_row = self.table_view.currentIndex().row()
        if curr_row == -1 or curr_row >= self.data.shape[0]:
            self.data = pd.concat([self.data, new_row], ignore_index=True)
        else:
            top = self.data.iloc[:curr_row + 1]
            bottom = self.data.iloc[curr_row + 1:]
            self.data = pd.concat([top, new_row, bottom], ignore_index=True)
        
        self.table_model.set_dataframe(self.data)
        self.table_view.resizeColumnsToContents()
        self.update_plot()

    def delete_row(self):
        """Delete the selected row."""
        curr_row = self.table_view.currentIndex().row()
        if curr_row < 0:
            QMessageBox.warning(self, "Warning", "Please select a row to delete")
            return
        
        confirm = QMessageBox.question(
            self, "Confirm Deletion", "Delete this row?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if confirm == QMessageBox.StandardButton.Yes:
            self.data = self.data.drop(index=curr_row).reset_index(drop=True)
            self.table_model.set_dataframe(self.data)
            self.table_view.resizeColumnsToContents()
            self.update_plot()

    def upload_button_clicked(self):
        """Load CSV or Excel file into the table."""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "CSV (*.csv);;Excel (*.xlsx)"
        )
        if not file:
            return
        try:
            self.data = pd.read_csv(file) if file.endswith(".csv") else pd.read_excel(file)
            # Ensure required columns exist
            required_cols = ["Timestamp", "Frequency(Hz)", "Resistance(Ω)", "Reactance(Ω)", "Magnitude(Ω)", "Phase(°)"]
            for col in required_cols:
                if col not in self.data.columns:
                    self.data[col] = np.nan
            
            self.table_model = TableModel(self.data)
            self.table_model.dataChanged.connect(self.update_plot)
            self.table_view.setModel(self.table_model)
            self.table_view.resizeColumnsToContents()
            self.update_plot()
        except Exception as e:
            QMessageBox.warning(self, "File Error", str(e))

    # ---------------------------
    # Plot update
    # ---------------------------
    def update_plot(self):
        """Update the Resistance vs Frequency plot from current data."""
        try:
            if self.data.empty or "Frequency(Hz)" not in self.data.columns or "Resistance(Ω)" not in self.data.columns:
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
            self.plot_widget.plot(xvals, yvals, pen='r', symbol='o', symbolSize=3)
            
        except Exception as e:
            QMessageBox.warning(self, "Plot Error", str(e))

    # ---------------------------
    # Export
    # ---------------------------
    def export_csv(self):
        """Export data to CSV file."""
        try:
            if self.data.empty:
                QMessageBox.information(self, "Export", "No data to export.")
                return
            fname, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV files (*.csv)")
            if not fname:
                return
            self.data.to_csv(fname, index=False)
            self.statusBar().showMessage(f"Saved {len(self.data)} rows to {fname}", 5000)
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    # ---------------------------
    # Theme toggle
    # ---------------------------
    def toggle_theme(self):
        """Toggle between light and dark themes."""
        self.is_dark = not self.is_dark
        if self.is_dark:
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
        """Open Sauerbrey & Konazawa dialog."""
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
            if hasattr(self, 'cryst_dyn_win') and self.cryst_dyn_win:
                self.cryst_dyn_win.close()

            self.cryst_dyn_win = QWidget()
            self.cryst_dyn_win.setWindowTitle("Crystallization Dynamics")
            self.cryst_dyn_win.resize(1400, 900)
            layout = QVBoxLayout(self.cryst_dyn_win)

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
            from Models.Butterworth import parameter
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
                react = pd.to_numeric(block["Reactance(Ω)"], errors='coerce').fillna(0).values
                
                if len(freqs) == 0:
                    continue
                    
                impedance = resist + 1j * react
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
        """Clean up resources when closing the application."""
        try:
            # Stop any ongoing acquisition
            if self.acquisition_active:
                self.acquisition_active = False
                self.acquisition_stop_flag.set()
                if self.acquisition_thread and self.acquisition_thread.is_alive():
                    self.acquisition_thread.join(timeout=2.0)
            
            # Stop timer if running
            if hasattr(self, 'poll_timer') and self.poll_timer.isActive():
                self.poll_timer.stop()
            
            # Close NanoVNA connection
            if self.vna and self._is_vna_connected():
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