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
import numpy as np
import pandas as pd
from Models.phaseshift import phaseshiftmethod

from PyQt6.QtCore import Qt, QTimer, QSize, QAbstractTableModel, QModelIndex, pyqtSignal, QObject
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QFileDialog, QGroupBox, QSplitter, QWidget, QTableView, QLineEdit, QToolBar,
    QHBoxLayout, QMessageBox, QPushButton, QLabel, QVBoxLayout, QMainWindow,
    QApplication, QStatusBar, QFormLayout, QComboBox, QSizePolicy, QFrame, QTabWidget
)
from Models.Butterworth import parameter, butterworth
from Models.Avrami import compute_X_t, fit as avrami_fit, formula as avrami_formula
from Models.Sauerbrey import sauerbrey
from Models.konazawa import konazawa
from Models.Sauerbrey import sauerbrey, parameter_sauerbrey as sauer_param
from Models.konazawa import konazawa, parameter_konazawa as konaz_param
#Matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import (
  NavigationToolbar2QT as NavigationToolbar,
)
#ReportLab pdf
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
#zip file
import zipfile

# PLotting
#Resistance vs Frequency
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.tight_layout()
        super().__init__(fig)
        
    def clear(self):
        """Clear the plot and Tables"""
        try:
            self.axes.clear()
            self.axes.set_xlabel("Frequency(Hz)")
            self.axes.set_ylabel("Resistance(Ω)")
            self.draw()
        except Exception:
            pass
        
# Other plots from crystallization dynamics
class MplMultiCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes_Rm = fig.add_subplot(221)
        self.axes_Fs = fig.add_subplot(222)
        self.axes_Lm = fig.add_subplot(223)
        self.axes_Cm = fig.add_subplot(224)
        fig.tight_layout()
        super().__init__(fig)
        
    def clear(self):
        """Clear all subplots"""
        try:
            self.axes_Rm.clear()
            self.axes_Fs.clear()
            self.axes_Lm.clear()
            self.axes_Cm.clear()
            
            # Reset titles and labels
            self.axes_Rm.set_title("Motional Resistance vs time")
            self.axes_Rm.set_xlabel("Time(s)")
            self.axes_Rm.set_ylabel("Rm(Ω)")
            
            self.axes_Fs.set_title("Resonance frequency vs time")
            self.axes_Fs.set_xlabel("Time(s)")
            self.axes_Fs.set_ylabel("fs(Hz)")
            
            self.axes_Lm.set_title("Motional Inductance vs time")
            self.axes_Lm.set_xlabel("Time(s)")
            self.axes_Lm.set_ylabel("Lm(H)")
            
            self.axes_Cm.set_title("Motional Capacitance vs time")
            self.axes_Cm.set_xlabel("Time(s)")
            self.axes_Cm.set_ylabel("Cm(F)")
            
            self.draw()
        except Exception:
            pass
        
        

# Signal handler for thread-safe communication

class DataSignalHandler(QObject):
    data_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str, str)
    status_update = pyqtSignal(str)



# Table model ()

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



# Main application window

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crystallization Analyzer")
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
        
        # FIX: Initialize analysis data structures
        self.crystdynamics = pd.DataFrame()
        self.plot_crystallization_fraction = None
        self.multiplot = None
        self.cryst_dyn_win = None
        self.cryst_kin_win = None
        
        # Signal handler for thread-safe communication
        self.signal_handler = DataSignalHandler()
        self.signal_handler.data_received.connect(self.vna_data)
        self.signal_handler.error_occurred.connect(self.error)
        self.signal_handler.status_update.connect(self.status_update)

        # GUI build
        self._build_actions()
        self._build_toolbar()
        self._build_statusbar()
        self._build_main_layout()

        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(2000)
        self.poll_timer.timeout.connect(self.run_single_sweep)

        

        

        try:
            import pyvisa
            self.pyvisa_rm = pyvisa.ResourceManager()
        except Exception:
            self.pyvisa_rm = None

        # initial populate of device list
        QTimer.singleShot(50, self.rescan_ports)

    # check connection status
    def _is_vna_connected(self):
        """Check if VNA is connected by checking serial port status."""
        return (self.vna is not None and 
                self.vna.ser is not None and 
                self.vna.ser.is_open)


    # Actions & toolbar
  
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
        self.button_action = QAction("New File", self)
        file_menu.addAction(self.button_action)
        file_menu.addSeparator()
        self.button_action1 = QAction("Open File", self)
        file_menu.addAction(self.button_action1)
        file_menu.addSeparator()
        self.button_action3 = QAction("Save As", self)
        file_menu.addAction(self.button_action3)
        file_menu.addSeparator()
        file_menu.addAction(QAction("Close", self))
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
        
        #Toggling file menu buttons
        self.button_action.triggered.connect(self.new_file)
        self.button_action1.triggered.connect(self.open_file)
        
        self.button_action3.triggered.connect(self.saveas)

    def _build_statusbar(self):
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        self.statusBar().showMessage("Ready")


    # Main layout

    def _build_main_layout(self):
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main_splitter)

        # Controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(12)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # Metadata
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

        # Acquisition
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
        self.btn_combobox = QComboBox()
        self.btn_combobox.addItems(["S11 (1-Port)", "S21 (2-port)"])
        self.btn_combobox2 = QComboBox()
        self.btn_combobox2.addItems(["Series Set-Up", "Parallel Set-Up"])
        btns.addWidget(self.btn_combobox)
        btns.addWidget(self.btn_combobox2)
        
        btns.addWidget(self.btn_connect)
        btns.addWidget(self.btn_single)
        acq_layout.addLayout(btns)

        acq_grp.setLayout(acq_layout)
        left_layout.addWidget(acq_grp)

        # Logging controls
        control_grp = QGroupBox("Logging Controls")
        logging_layout = QVBoxLayout()
        ctl_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Logging")
        self.btn_stop = QPushButton("Stop Logging")
        self.upload_button = QPushButton("Upload Data")
        
        interval_value = QHBoxLayout()
        interval_value.addWidget(QLabel("Interval (s):"))
        self.interval = QComboBox()
        self.interval.addItems([str(i) for i in [2, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 180, 300, 600]])
        self.interval.setCurrentText("2")
        interval_value.addWidget(self.interval)
        interval_value.addStretch()
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start_logging_button)
        self.btn_stop.clicked.connect(self.stop_logging_button)
        self.upload_button.clicked.connect(self.upload_button_clicked)
        ctl_layout.addWidget(self.btn_start)
        ctl_layout.addWidget(self.btn_stop)
        ctl_layout.addWidget(self.upload_button)
        logging_layout.addLayout(ctl_layout)
        logging_layout.addLayout(interval_value)
        control_grp.setLayout(logging_layout)
        left_layout.addWidget(control_grp)

        # Analysis
        analysis_grp = QGroupBox("Analysis Tools")
        an_layout = QVBoxLayout()
        self.btn_sauer = QPushButton("Sauerbrey / Konazawa / PhaseShift")
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
        self.delete_tab = QPushButton("Delete Table")
        self.delete_tab.clicked.connect(self.delete_table)
        btn_layout.addWidget(self.insert_btn)
        btn_layout.addWidget(self.delete_btn)
        btn_layout.addWidget(self.delete_tab)
        table_layout.addLayout(btn_layout)

        right_splitter.addWidget(table_frame)

        # Plot
        plot_frame = QFrame()
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(4, 4, 4, 4)
        plot_layout.addWidget(QLabel("Resistance vs Frequency"))
        self.plot_widget = MplCanvas(self, width=5, height=4, dpi=100)
        self.plot_widget.axes.set_xlabel("Frequency(Hz)")
        self.plot_widget.axes.set_ylabel("Resistance(Ω)")
        self.plot_widget.axes.plot(self.freqs, np.zeros_like(self.freqs), 'b-')
        
        self.plot_toolbar = NavigationToolbar(self.plot_widget, self)
        plot_layout.addWidget(self.plot_toolbar)
        self.plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        plot_layout.addWidget(self.plot_widget)
        right_splitter.addWidget(plot_frame)

        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        
        
    #File menu toggling buttons
    def new_file(self):
        confirm = QMessageBox.question(
            self, "Save", "Do you want to save this file?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if confirm == QMessageBox.StandardButton.Yes:
            self.saveas()
            
        
        self.data = pd.DataFrame(columns=["Timestamp", "Frequency(Hz)", "Resistance(Ω)", "Reactance(Ω)", "Magnitude(Ω)", "Phase(°)"])
        self.table_model.set_dataframe(self.data)
        
        try:
            self.plot_widget.clear()
        except Exception:
            try:
                self.plot_widget.axes.clear()
                self.plot_widget.axes.set_xlabel("Frequency(Hz)")
                self.plot_widget.axes.set_ylabel("Resistance(Ω)")
                self.plot_widget.draw()
            except Exception:
                pass
                
        
        if hasattr(self, 'plot_crystallization_fraction') and self.plot_crystallization_fraction:
            try:
                self.plot_crystallization_fraction.clear()
            except Exception:
                try:
                    self.plot_crystallization_fraction.axes.clear()
                    self.plot_crystallization_fraction.draw()
                except Exception:
                    pass
            
        
        if hasattr(self, 'multiplot') and self.multiplot:
            try:
                self.multiplot.clear()
            except Exception:
                try:
                    self.multiplot.axes_Rm.clear()
                    self.multiplot.axes_Fs.clear()
                    self.multiplot.axes_Lm.clear()
                    self.multiplot.axes_Cm.clear()
                    self.multiplot.draw()
                except Exception:
                    pass

        
        fields_to_clear = ['rm_edit', 'lm_edit', 'c0_edit', 'cm_edit', 'f0_edit', 'f_edit', 'finf_edit']
        for field in fields_to_clear:
            if hasattr(self, field):
                try:
                    getattr(self, field).clear()
                except Exception:
                    pass
        
        
        self.crystdynamics = pd.DataFrame()
        
        
    def open_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select File of Resistance vs Frequency Data Table", "", "CSV (*.csv);;Excel (*.xlsx)"
        )
        if not file:
            return
        try:
            self.data = pd.read_csv(file) if file.endswith(".csv") else pd.read_excel(file)
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
        
        
    #Save File    
    def saveas(self):
        file = QFileDialog.getExistingDirectory(self, "Select folder to Save Report")
        if not file:
            return
        
        try:
            # File paths
            csv_path = os.path.join(file, "Resistance vs Frequency.csv")
            csv_path1 = os.path.join(file, "Crystallization Dynamics Table.csv")
            bvd_path = os.path.join(file, "BVD Parameters.csv")
            analysis_path = os.path.join(file, "Analysis Results.csv")
            pdf_path = os.path.join(file, "Report.pdf")
            zip_path = os.path.join(file, "Crystallization Report.zip")
            
            doc = SimpleDocTemplate(pdf_path, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []
            
            # Title
            elements.append(Paragraph("Crystallization Analysis Report", styles["Title"]))
            elements.append(Spacer(1, 12))
            
            # Metadata
            elements.append(Paragraph("Metadata", styles["Heading2"]))
            elements.append(Paragraph(f"Sample: {self.sample_name.text()}", styles["Normal"]))
            elements.append(Paragraph(f"Batch No.: {self.batch_number.text()}", styles["Normal"]))
            elements.append(Paragraph(f"Operator: {self.operator_name.text()}", styles["Normal"]))
            elements.append(Paragraph(f"Notes: {self.notes.text()}", styles["Normal"]))
            elements.append(Spacer(1, 12))
            
            # Raw Data table in csv
            if not self.data.empty:
                self.data.to_csv(csv_path, index=False)
                elements.append(Paragraph("Resistance vs Frequency Data Table saved to CSV file", styles["Normal"]))
                elements.append(Spacer(1, 12))
                
                # Resistance vs Frequency Plot
                plot_filename = os.path.join(file, "plot.png")
                self.plot_widget.figure.savefig(plot_filename, dpi=300, bbox_inches='tight')
                img = Image(plot_filename, width=400, height=300)
                elements.append(Paragraph("Resistance vs Frequency Plot", styles["Heading3"]))
                elements.append(img)
                elements.append(Spacer(1, 12))
            
            # BVD Parameters
            bvd_data = []
            if hasattr(self, 'rm_edit') and self.rm_edit.text():
                bvd_data.append(['Parameter', 'Value', 'Unit'])
                bvd_data.append(['Motional Resistance (Rm)', self.rm_edit.text(), 'Ω'])
                bvd_data.append(['Motional Inductance (Lm)', self.lm_edit.text(), 'H'])
                bvd_data.append(['Motional Capacitance (Cm)', self.cm_edit.text(), 'F'])
                bvd_data.append(['Static Capacitance (C0)', self.c0_edit.text(), 'F'])
                bvd_data.append(['Resonant Frequency (Fs)', self.f_edit.text(), 'Hz'])
                
                # Save BVD parameters to CSV
                bvd_df = pd.DataFrame(bvd_data[1:], columns=bvd_data[0])
                bvd_df.to_csv(bvd_path, index=False)
                elements.append(Paragraph("BVD Parameters", styles["Heading2"]))
                elements.append(Paragraph("BVD parameters saved to CSV file", styles["Normal"]))
                elements.append(Spacer(1, 12))
            
            # Analysis Results (Sauerbrey and Konazawa)
            analysis_data = []
            has_analysis_data = False
            
            # Check for Sauerbrey results
            if (hasattr(self, 'sauerb_freq_shift') and self.sauerb_freq_shift.text() and
                hasattr(self, 'sauerb_mass_change') and self.sauerb_mass_change.text()):
                if not has_analysis_data:
                    analysis_data.append(['Analysis Type', 'Parameter', 'Value', 'Unit'])
                    has_analysis_data = True
                
                analysis_data.extend([
                    ['Sauerbrey', 'Initial Frequency (f₀)', getattr(self, 'sauerb_f0', QLineEdit()).text() or 'N/A', 'Hz'],
                    ['Sauerbrey', 'Current Frequency (ft)', getattr(self, 'sauerb_current_freq', QLineEdit()).text() or 'N/A', 'Hz'],
                    ['Sauerbrey', 'Frequency Shift (Δf)', self.sauerb_freq_shift.text(), 'Hz'],
                    ['Sauerbrey', 'Mass Change (Δm)', self.sauerb_mass_change.text(), 'kg'],
                    ['Sauerbrey', 'Mass Change per Area', self.sauerb_mass_per_area.text(), 'ng/cm²'],
                    ['Sauerbrey', 'Quartz Density', getattr(self, 'sauerb_density', QLineEdit()).text() or 'N/A', 'kg/m³'],
                    ['Sauerbrey', 'Shear Modulus', getattr(self, 'sauerb_shear', QLineEdit()).text() or 'N/A', 'Pa'],
                    ['Sauerbrey', 'Electrode Area', getattr(self, 'sauerb_area', QLineEdit()).text() or 'N/A', 'm²']
                ])
            
            # Check for Konazawa results
            if (hasattr(self, 'konaz_freq_shift') and self.konaz_freq_shift.text() and
                hasattr(self, 'konaz_result') and self.konaz_result.text()):
                if not has_analysis_data:
                    analysis_data.append(['Analysis Type', 'Parameter', 'Value', 'Unit'])
                    has_analysis_data = True
                
                analysis_data.extend([
                    ['Konazawa', 'Initial Frequency (f₀)', getattr(self, 'konaz_f0', QLineEdit()).text() or 'N/A', 'Hz'],
                    ['Konazawa', 'Current Frequency (ft)', getattr(self, 'konaz_current_freq', QLineEdit()).text() or 'N/A', 'Hz'],
                    ['Konazawa', 'Frequency Shift (Δf)', self.konaz_freq_shift.text(), 'Hz'],
                    ['Konazawa', 'Konazawa Result (n)', self.konaz_result.text(), '-'],
                    ['Konazawa', 'Quartz Density', getattr(self, 'konaz_p', QLineEdit()).text() or 'N/A', 'kg/m³'],
                    ['Konazawa', 'Shear Modulus', getattr(self, 'konaz_u', QLineEdit()).text() or 'N/A', 'Pa'],
                    ['Konazawa', 'Liquid Density', getattr(self, 'konaz_p1', QLineEdit()).text() or 'N/A', 'kg/m³']
                ])
            
            
            if (hasattr(self, 'k_edit') and self.k_edit.text() and
                hasattr(self, 'n_edit') and self.n_edit.text()):
                if not has_analysis_data:
                    analysis_data.append(['Analysis Type', 'Parameter', 'Value', 'Unit'])
                    has_analysis_data = True
                
                analysis_data.extend([
                    ['Avrami', 'Initial Frequency (f₀)', getattr(self, 'f0_edit', QLineEdit()).text() or 'N/A', 'Hz'],
                    ['Avrami', 'Final Frequency (f∞)', getattr(self, 'finf_edit', QLineEdit()).text() or 'N/A', 'Hz'],
                    ['Avrami', 'Rate Constant (k)', self.k_edit.text(), 's⁻ⁿ'],
                    ['Avrami', 'Avrami Exponent (n)', self.n_edit.text(), '-']
                ])
            
            if has_analysis_data:
                # Save analysis results to CSV
                analysis_df = pd.DataFrame(analysis_data[1:], columns=analysis_data[0])
                analysis_df.to_csv(analysis_path, index=False)
                elements.append(Paragraph("Analysis Results", styles["Heading2"]))
                elements.append(Paragraph("Analysis results saved to CSV file", styles["Normal"]))
                elements.append(Spacer(1, 12))
            
            # Crystallization Dynamics table and plots
            if hasattr(self, 'crystdynamics') and not self.crystdynamics.empty:
                self.crystdynamics.to_csv(csv_path1, index=False)
                elements.append(Paragraph("Crystallization Dynamics Data table saved to CSV file", styles["Normal"]))
                elements.append(Spacer(1, 12))
                
                if hasattr(self, 'multiplot') and self.multiplot:
                    # Individual plot saves
                    plot_files = [
                        ("Rm vs Time.png", self.multiplot.axes_Rm, "Motional Resistance vs Time"),
                        ("Cm vs Time.png", self.multiplot.axes_Cm, "Motional Capacitance vs Time"),
                        ("Lm vs Time.png", self.multiplot.axes_Lm, "Motional Inductance vs Time"),
                        ("Fs vs Time.png", self.multiplot.axes_Fs, "Resonance Frequency vs Time")
                    ]
                    
                    for filename, axis, title in plot_files:
                        plot_path = os.path.join(file, filename)
                        fig = axis.figure
                        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                        elements.append(Paragraph(title, styles["Heading3"]))
                        img = Image(plot_path, width=400, height=300)
                        elements.append(img)
                        elements.append(Spacer(1, 12))
            
            # Crystallization kinetics plot
            if hasattr(self, 'plot_crystallization_fraction') and self.plot_crystallization_fraction:
                plot_filename5 = os.path.join(file, "Crystallinity vs Time.png")
                self.plot_crystallization_fraction.figure.savefig(plot_filename5, dpi=300, bbox_inches='tight')
                elements.append(Paragraph("Crystallinity vs Time", styles["Heading3"]))
                img5 = Image(plot_filename5, width=400, height=300)
                elements.append(img5)
                elements.append(Spacer(1, 12))
            
            # Build PDF
            doc.build(elements)
            
            # Create Zip File
            with zipfile.ZipFile(zip_path, "w") as zipf:
                # Add all CSV files
                csv_files = [csv_path, csv_path1, bvd_path, analysis_path]
                for csv_file in csv_files:
                    if os.path.exists(csv_file):
                        zipf.write(csv_file, arcname=os.path.basename(csv_file))
                
                # Add PDF
                zipf.write(pdf_path, arcname=os.path.basename(pdf_path))
                
                # Add plot images to zip
                for root, dirs, files in os.walk(file):
                    for f in files:
                        if f.endswith('.png'):
                            file_path = os.path.join(root, f)
                            zipf.write(file_path, arcname=f)
            
            self.statusBar().showMessage(f"Complete report saved to {zip_path}", 5000)
            QMessageBox.information(self, "Save Complete", f"All files saved and zipped to:\n{zip_path}")
            
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))
            
        

    # Thread-safe signal handlers
    
    def vna_data(self, data_package: dict):
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
                if self.data.empty:
                    self.data = new_df.copy()
                else:
                    self.data = pd.concat([self.data, new_df], ignore_index=True)
                
                self.table_model.set_dataframe(self.data)
                self.table_view.resizeColumnsToContents()
                self.update_plot()
                
                scan_count = data_package.get("scan_count", "?")
                self.statusBar().showMessage(f"Received scan #{scan_count} ({len(freqs)} points)", 3000)
                
        except Exception as e:
            QMessageBox.warning(self, "Data Processing Error", str(e))

    def error(self, title: str, message: str):
        """Handle errors from background threads."""
        QMessageBox.warning(self, title, message)

    def status_update(self, message: str):
        """Handle status updates from background threads."""
        self.statusBar().showMessage(message, 5000)

    # Device scanning & connect
    
    def rescan_ports(self):
        """Populate device combobox with available serial ports."""
        self.combo_ports.clear()
        self.combo_ports.addItem("NanoVNA (Auto-detect)")
        
        # Add available serial ports
        try:
            ports = list(serial.tools.list_ports.comports())
            for port in ports:
                # Store port info
                port_desc = f"{port.device} ({port.description or 'Unknown'})"
                self.combo_ports.addItem(port_desc)
        except Exception as e:
            print(f"Error scanning ports: {e}")
        
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
            # Extract port device name
            if "(" in selected and selected.count("(") == 1:
                port = selected.split("(")[0].strip()
            else:
                port = selected.strip()
        
        def connect_worker():
            try:
                if port is None:
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
                
            except Exception as e:
                self.signal_handler.error_occurred.emit("Connection Error", str(e))
        
        # Run connection in background thread
        threading.Thread(target=connect_worker, daemon=True).start()
        self.statusBar().showMessage("Connecting to NanoVNA...", 2000)
        
    # Data acquisition

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
                
                # Fixed string comparison issue
                if self.btn_combobox.currentText() == "S11 (1-Port)":
                    try:
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
                        
                else:
                    try:
                        # Perform sweep and acquire data
                        if self.vna.sweep(start, stop, points):
                            result = self.vna.acquire_s21()
                            
                            if result[0] is not None:  # Check if frequencies are available
                                freqs, s21_values, impedances, resistance, reactance, magnitude, phase = result
                                
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
                        
            except Exception as e:
                self.signal_handler.error_occurred.emit("Sweep Error", str(e))
                
        # Fixed indentation issue
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
            
            if start >= stop:
                QMessageBox.warning(self, "Parameter Error", "Start Frequency must be less than Stop Frequency")
                return
            
            if points < 2 or points > 1001:
                QMessageBox.warning(self, "Sweep Points Error", "Please Choose sweep points in the range[2, 1001]")
                return
            
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Invalid sweep parameters")
            return
        
        # Start acquisition thread
        self.acquisition_active = True
        self.acquisition_stop_flag.clear()
        self.acquisition_thread = threading.Thread(
            target=self.acquisition, 
            args=(start, stop, points), 
            daemon=True
        )
        self.acquisition_thread.start()
        
        # Buttons in the screen
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.act_start.setEnabled(False)
        self.act_stop.setEnabled(True)
        
        self.statusBar().showMessage("Continuous logging started", 3000)

    def acquisition(self, start_freq, stop_freq, points):
        scan_count = 0
        try:
            interval = float(self.interval.currentText())
        except (ValueError, AttributeError):
            interval = 2
        
        while self.acquisition_active and not self.acquisition_stop_flag.is_set():
            # Fixed string comparison issue
            if self.btn_combobox.currentText() == "S11 (1-Port)":
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
                    
                    for _ in range(int(interval * 10)):
                        if self.acquisition_stop_flag.is_set():
                            break
                        time.sleep(0.1)
                        
                except Exception as e:
                    self.signal_handler.error_occurred.emit("Acquisition Error", str(e))
                    break
                
            else:
                try:
                    # Perform sweep and acquire data
                    if self.vna.sweep(start_freq, stop_freq, points):
                        result = self.vna.acquire_s21()
                        
                        if result[0] is not None:  # Check if frequencies are available
                            freqs, s21_values, impedances, resistance, reactance, magnitude, phase = result
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
                    
                    for _ in range(int(interval * 10)):
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
        
        if self.acquisition_thread and self.acquisition_thread.is_alive():
            self.acquisition_thread.join(timeout=2.0)
        
        # Updating buttons
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.act_start.setEnabled(True)
        self.act_stop.setEnabled(False)
        
        self.statusBar().showMessage("Logging stopped", 3000)

    # Inserting the table

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
        if self.data.empty or curr_row == -1 or curr_row >= self.data.shape[0]:
            if self.data.empty:
                self.data = new_row.copy()
            else:
                self.data = pd.concat([self.data, new_row], ignore_index=True)
        else:
            top = self.data.iloc[:curr_row + 1].copy()
            bottom = self.data.iloc[curr_row + 1:].copy()
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
            
    def delete_table(self):
        if self.data.empty:
            QMessageBox.warning(self, "Warning", "There is no Data Table to delete")
            return
        
        confirm = QMessageBox.question(
            self, "Confirm Deletion", "Delete this table?", 
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if confirm == QMessageBox.StandardButton.Yes:
            self.data = pd.DataFrame(columns=["Timestamp", "Frequency(Hz)", "Resistance(Ω)", "Reactance(Ω)", "Magnitude(Ω)", "Phase(°)"])
            self.table_model.set_dataframe(self.data)
            self.table_view.resizeColumnsToContents()
            self.update_plot()

    def upload_button_clicked(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "CSV (*.csv);;Excel (*.xlsx)"
        )
        if not file:
            return
        try:
            self.data = pd.read_csv(file) if file.endswith(".csv") else pd.read_excel(file)
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

    # Plot update

    def update_plot(self):
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
            
            self.plot_widget.axes.clear()
            self.plot_widget.axes.set_xlabel("Frequency(Hz)")
            self.plot_widget.axes.set_ylabel("Resistance(Ω)")
            self.plot_widget.axes.plot(xvals, yvals, 'ro-', markersize=3)
            self.plot_widget.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "Plot Error", str(e))

    # Export data into table as csv files
    
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

    # Theme toggle

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
            self.plot_widget.axes.set_facecolor("#1c1c1c")
            if hasattr(self, 'multiplot') and self.multiplot:
                self.multiplot.axes_Cm.set_facecolor("#1c1c1c")
                self.multiplot.axes_Rm.set_facecolor("#1c1c1c")
                self.multiplot.axes_Lm.set_facecolor("#1c1c1c")
                self.multiplot.axes_Fs.set_facecolor("#1c1c1c")
                self.multiplot.draw()
            self.plot_widget.draw()
        else:
            self.setStyleSheet("")
            self.plot_widget.axes.set_facecolor('w')
            self.plot_widget.draw()

    # Analysis windows & calculators

    def sauerbrey_konazawa(self):
        """Open Sauerbrey & Konazawa analysis window"""
        try:
            if hasattr(self, 'sauerbrey_win') and self.sauerbrey_win is not None:
                try:
                    self.sauerbrey_win.close()
                    self.sauerbrey_win.deleteLater()
                except Exception:
                    pass
                self.sauerbrey_win = None

            # Create new window
            self.sauerbrey_win = QWidget()
            self.sauerbrey_win.setWindowTitle("Sauerbrey & Konazawa Analysis")
            self.sauerbrey_win.resize(800, 600)
            self.sauerbrey_win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
            
            main_layout = QVBoxLayout(self.sauerbrey_win)
            tab_widget = QTabWidget()
            
            # Sauerbrey tab
            sauerbrey_tab = QWidget()
            sauerbrey_layout = QVBoxLayout(sauerbrey_tab)
            
            sauerbrey_group = QGroupBox("Sauerbrey Parameters")
            sauerbrey_form = QFormLayout()
            
            # Input fields
            self.sauerb_f0 = QLineEdit("5000000")  # 5 MHz default
            self.sauerb_density = QLineEdit("2650")  # Quartz density kg/m³
            self.sauerb_shear = QLineEdit("2.947e10")  # Shear modulus Pa
            self.sauerb_area = QLineEdit("1e-4")  # Electrode area m²
            
            sauerbrey_form.addRow("Initial Resonant Frequency f₀ (Hz):", self.sauerb_f0)
            sauerbrey_form.addRow("Quartz Density ρ (kg/m³):", self.sauerb_density)
            sauerbrey_form.addRow("Shear Modulus μ (Pa):", self.sauerb_shear)
            sauerbrey_form.addRow("Electrode Area A (m²):", self.sauerb_area)
            
            sauerbrey_group.setLayout(sauerbrey_form)
            sauerbrey_layout.addWidget(sauerbrey_group)
            
            # Calculation section
            calc_group = QGroupBox("Mass Change Calculation")
            calc_layout = QVBoxLayout()
            
            # Current frequency display
            freq_layout = QHBoxLayout()
            freq_layout.addWidget(QLabel("Current Frequency ft (Hz):"))
            self.sauerb_current_freq = QLineEdit()
            self.sauerb_current_freq.setReadOnly(True)
            freq_layout.addWidget(self.sauerb_current_freq)
            calc_layout.addLayout(freq_layout)
            
            # Auto-detect frequency button
            detect_freq_btn = QPushButton("Auto-Detect Current Frequency")
            detect_freq_btn.clicked.connect(lambda: self.auto_detect_frequency('sauerbrey'))
            calc_layout.addWidget(detect_freq_btn)
            
            # Calculate button
            calc_btn = QPushButton("Calculate Mass Change")
            calc_btn.clicked.connect(self.calculate_sauerbrey_mass)
            calc_layout.addWidget(calc_btn)
            
            # Results
            result_layout = QFormLayout()
            self.sauerb_freq_shift = QLineEdit()
            self.sauerb_freq_shift.setReadOnly(True)
            self.sauerb_mass_change = QLineEdit()
            self.sauerb_mass_change.setReadOnly(True)
            self.sauerb_mass_per_area = QLineEdit()
            self.sauerb_mass_per_area.setReadOnly(True)
            
            result_layout.addRow("Frequency Shift Δf (Hz):", self.sauerb_freq_shift)
            result_layout.addRow("Mass Change Δm (kg):", self.sauerb_mass_change)
            result_layout.addRow("Mass Change Δm (ng/cm²):", self.sauerb_mass_per_area)
            
            calc_layout.addLayout(result_layout)
            calc_group.setLayout(calc_layout)
            sauerbrey_layout.addWidget(calc_group)
            
            # Add Sauerbrey tab
            tab_widget.addTab(sauerbrey_tab, "Sauerbrey Analysis")
            
            # Konazawa tab
            konazawa_tab = QWidget()
            konazawa_layout = QVBoxLayout(konazawa_tab)
            
            konazawa_group = QGroupBox("Konazawa Parameters")
            konazawa_form = QFormLayout()
            
            self.konaz_f0 = QLineEdit("5000000")  # 5 MHz default
            self.konaz_p = QLineEdit("2650")  # Quartz density kg/m³
            self.konaz_u = QLineEdit("2.947e10")  # Shear Modulus of quartz in Pascals
            self.konaz_p1 = QLineEdit("1000")  # Liquid density kg/m³, can be editable, if any other liquid used
            
            konazawa_form.addRow("Initial Frequency f₀ (Hz):", self.konaz_f0)
            konazawa_form.addRow("Quartz Density p (kg/m³):", self.konaz_p)
            konazawa_form.addRow("Shear Modulus μ (Pa):", self.konaz_u)
            konazawa_form.addRow("Liquid Density p1 (kg/m³):", self.konaz_p1)
            
            konazawa_group.setLayout(konazawa_form)
            konazawa_layout.addWidget(konazawa_group)
            
            # Konazawa calculation
            konaz_calc_group = QGroupBox("Konazawa Calculation")
            konaz_calc_layout = QVBoxLayout()
            
            # Current frequency display
            konaz_freq_layout = QHBoxLayout()
            konaz_freq_layout.addWidget(QLabel("Current Frequency ft (Hz):"))
            self.konaz_current_freq = QLineEdit()
            self.konaz_current_freq.setReadOnly(True)
            konaz_freq_layout.addWidget(self.konaz_current_freq)
            konaz_calc_layout.addLayout(konaz_freq_layout)
            
            konaz_detect_freq_btn = QPushButton("Auto-Detect Current Frequency")
            konaz_detect_freq_btn.clicked.connect(lambda: self.auto_detect_frequency('konazawa'))
            konaz_calc_layout.addWidget(konaz_detect_freq_btn)
            
            konaz_calc_btn = QPushButton("Calculate Konazawa")
            konaz_calc_btn.clicked.connect(self.calculate_konazawa_result)
            konaz_calc_layout.addWidget(konaz_calc_btn)
            
            # Results
            konaz_result_layout = QFormLayout()
            self.konaz_freq_shift = QLineEdit()
            self.konaz_freq_shift.setReadOnly(True)
            self.konaz_result = QLineEdit()
            self.konaz_result.setReadOnly(True)
            
            konaz_result_layout.addRow("Frequency Shift Δf (Hz):", self.konaz_freq_shift)
            konaz_result_layout.addRow("Konazawa Result n:", self.konaz_result)
            
            konaz_calc_layout.addLayout(konaz_result_layout)
            konaz_calc_group.setLayout(konaz_calc_layout)
            konazawa_layout.addWidget(konaz_calc_group)
            
            tab_widget.addTab(konazawa_tab, "Konazawa Analysis")
            
            # Phase shift tab 
            # Phase shift tab 
            phase_shift_tab = QWidget()
            phase_layout = QVBoxLayout(phase_shift_tab)
            phase_grp = QGroupBox("Phase Shift Analysis")
            phase_form = QFormLayout()

            # Add input fields and button instead of immediate calculation
            phase_form.addRow(QLabel("Connect to NanoVNA and run a measurement first"))

            self.phase_calc_btn = QPushButton("Calculate Phase Shift Parameters")
            self.phase_calc_btn.clicked.connect(self.calculate_phase_shift)
            phase_form.addRow(self.phase_calc_btn)

            # Result display fields
            self.phase_rm_result = QLineEdit()
            self.phase_rm_result.setReadOnly(True)
            self.phase_reff_result = QLineEdit()
            self.phase_reff_result.setReadOnly(True)
            self.phase_cm_result = QLineEdit()
            self.phase_cm_result.setReadOnly(True)
            self.phase_lm_result = QLineEdit()
            self.phase_lm_result.setReadOnly(True)

            phase_form.addRow("Rm (Ω):", self.phase_rm_result)
            phase_form.addRow("Reff (Ω):", self.phase_reff_result)
            phase_form.addRow("Cm (F):", self.phase_cm_result)
            phase_form.addRow("Lm (H):", self.phase_lm_result)

            phase_grp.setLayout(phase_form)
            phase_layout.addWidget(phase_grp)

            
            tab_widget.addTab(phase_shift_tab, "Phase Shift")
            
            
            main_layout.addWidget(tab_widget)
            
            # Auto-detect frequencies on startup
            self.auto_detect_frequency('both')
            
            # Show the window
            self.sauerbrey_win.show()
            
        except ImportError as e:
            QMessageBox.warning(self, "Import Error", f"Could not import required modules: {str(e)}\nMake sure Models/Sauerbrey.py and Models/konazawa.py are available.")
        except Exception as e:
            QMessageBox.warning(self, "Window Error", f"Error opening Sauerbrey/Konazawa window: {str(e)}")
            
    def calculate_phase_shift(self):
        """Calculate phase shift parameters from current data"""
        try:
            if not self._is_vna_connected():
                QMessageBox.warning(self, "Connection Error", "Please connect to NanoVNA first")
                return
                
            if self.data.empty:
                QMessageBox.warning(self, "No Data", "Please run a sweep first.")
                return
            
            # Get the latest sweep data
            points = int(self.sweep_points.text() or 201)
            if len(self.data) < points:
                QMessageBox.warning(self, "Insufficient Data", "Please run a complete sweep first")
                return
            try:
                if self.vna.sweep(float(self.start_frequency.text() or "1000000"), 
                                float(self.end_frequency.text() or "10000000"), 
                                points):
                    result = self.vna.acquire_s21()
                    
                    if result[0] is not None:
                        freqs, s21_values, *_ = result
                        
                        # Phase shift analysis
                        from Models.phaseshift import phaseshiftmethod
                        method = phaseshiftmethod()
                        phase_deg, s21, s21_db = method.phaseshift(s21_values)
                        results = method.phaseshiftfrequency(freqs, phase_deg, s21_db)
                        Cm, Rm, Lm, Reff, delta_f = method.phaseshiftcalculation(results)
                        
                        # Update result fields
                        self.phase_rm_result.setText(f"{Rm:.6f}")
                        self.phase_reff_result.setText(f"{Reff:.6f}")
                        self.phase_cm_result.setText(f"{Cm:.6e}")
                        self.phase_lm_result.setText(f"{Lm:.6e}")
                        
                        QMessageBox.information(self, "Phase Shift Analysis Complete", 
                                            f"Phase shift parameters calculated successfully")
                    else:
                        QMessageBox.warning(self, "Data Error", "No S21 data received from NanoVNA")
                else:
                    QMessageBox.warning(self, "Sweep Error", "Failed to perform sweep")
                    
            except ImportError:
                QMessageBox.warning(self, "Missing Model", "Phase shift model not found")
            except Exception as e:
                QMessageBox.warning(self, "Calculation Error", f"Error in phase shift calculation: {str(e)}")
                
        except Exception as e:
            QMessageBox.warning(self, "Phase Shift Error", f"Error calculating phase shift: {str(e)}")

    def auto_detect_frequency(self, target='both'):
        """Auto-detect current frequency from impedance data using your parameter functions"""
        try:
            if self.data.empty:
                QMessageBox.warning(self, "No Data", "No measurement data available.")
                return
            
            # Get the latest sweep data
            points = int(self.sweep_points.text() or 201)
            if len(self.data) < points:
                QMessageBox.warning(self, "Insufficient Data", "Please run again")
                return
            
            last_sweep = self.data.tail(points)
            freqs = pd.to_numeric(last_sweep["Frequency(Hz)"], errors='coerce').dropna().values
            resist = pd.to_numeric(last_sweep["Resistance(Ω)"], errors='coerce').fillna(0).values
            react = pd.to_numeric(last_sweep["Reactance(Ω)"], errors='coerce').fillna(0).values
            
            if len(freqs) == 0:
                QMessageBox.warning(self, "Data Error", "No valid frequency data")
                return
            
            # Create impedance array
            impedance = resist + 1j * react
            
            # Get resonant frequency using your parameter functions
            ft_sauer = sauer_param(freqs, impedance)
            ft_konaz = konaz_param(freqs, impedance)
            
            if target in ['sauerbrey', 'both'] and hasattr(self, 'sauerb_current_freq'):
                self.sauerb_current_freq.setText(f"{ft_sauer:.2f}")
                
            if target in ['konazawa', 'both'] and hasattr(self, 'konaz_current_freq'):
                self.konaz_current_freq.setText(f"{ft_konaz:.2f}")
                
            if target == 'both':
                self.statusBar().showMessage(f"Auto-detected frequencies - Sauerbrey: {ft_sauer:.2f} Hz, Konazawa: {ft_konaz:.2f} Hz", 3000)
            else:
                self.statusBar().showMessage(f"Auto-detected frequency: {ft_sauer if target == 'sauerbrey' else ft_konaz:.2f} Hz", 3000)
                
        except ImportError as e:
            QMessageBox.warning(self, "Import Error", f"Could not import parameter functions: {str(e)}")
        except Exception as e:
            QMessageBox.warning(self, "Detection Error", f"Error detecting frequency: {str(e)}")

    def calculate_sauerbrey_mass(self):
        """Calculate mass change using your Sauerbrey function"""
        try:
            # Get parameters
            f0 = float(self.sauerb_f0.text())
            p = float(self.sauerb_density.text())  # density
            u = float(self.sauerb_shear.text())   # shear modulus
            A = float(self.sauerb_area.text())    # area
            
            # Get current frequency
            ft_text = self.sauerb_current_freq.text()
            if not ft_text:
                QMessageBox.warning(self, "Data Error", "No current frequency available. Please auto-detect or run a measurement first.")
                return
            
            ft = float(ft_text)
            
            # Calculate using your Sauerbrey function: sauerbrey(f0, p, u, ft, A)
            mass_change = sauerbrey(f0, p, u, ft, A)
            freq_shift = f0 - ft
            
            # Calculate mass per unit area in ng/cm²
            mass_per_area = abs(mass_change) / A * 1e4 * 1e9  # Convert kg/m² to ng/cm²
            
            # Update results
            self.sauerb_freq_shift.setText(f"{freq_shift:.2f}")
            self.sauerb_mass_change.setText(f"{mass_change:.6e}")
            self.sauerb_mass_per_area.setText(f"{mass_per_area:.2f}")
            
            QMessageBox.information(self, "Sauerbrey Calculation Complete", 
                                f"Frequency Shift: {freq_shift:.2f} Hz\n"
                                f"Mass Change: {mass_change:.6e} kg\n"
                                f"Mass Change: {mass_per_area:.2f} ng/cm²")
            
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input parameters: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Calculation Error", f"Error calculating Sauerbrey: {e}")

    def calculate_konazawa_result(self):
        """Calculate result using your Konazawa function"""
        try:
            # Get parameters
            f0 = float(self.konaz_f0.text())
            p = float(self.konaz_p.text())    # quartz density
            u = float(self.konaz_u.text())   # shear modulus
            p1 = float(self.konaz_p1.text()) # liquid density
            
            # Get current frequency
            ft_text = self.konaz_current_freq.text()
            if not ft_text:
                QMessageBox.warning(self, "Data Error", "No current frequency available. Please auto-detect or run a measurement first.")
                return
            
            ft = float(ft_text)
            
            # Calculate using your Konazawa function: konazawa(f0, p, u, ft, p1)
            konaz_result = konazawa(f0, p, u, ft, p1)
            freq_shift = f0 - ft
            
            # Update results
            self.konaz_freq_shift.setText(f"{freq_shift:.2f}")
            self.konaz_result.setText(f"{konaz_result:.6e}")
            
            QMessageBox.information(self, "Konazawa Calculation Complete", 
                                f"Frequency Shift: {freq_shift:.2f} Hz\n"
                                f"Konazawa Result n: {konaz_result:.6e}")
            
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input parameters: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Calculation Error", f"Error calculating Konazawa: {e}")

    # Crystallization dynamics using BVD model
    def crystallizationdynamics(self):
        try:
            if hasattr(self, 'cryst_dyn_win') and self.cryst_dyn_win is not None:
                try:
                    self.cryst_dyn_win.close()
                    self.cryst_dyn_win.deleteLater()
                except Exception:
                    pass
                self.cryst_dyn_win = None

            self.cryst_dyn_win = QWidget()
            self.cryst_dyn_win.setWindowTitle("Crystallization Dynamics")
            self.cryst_dyn_win.resize(1400, 900)
            layout = QVBoxLayout(self.cryst_dyn_win)

            toolbar = QToolBar()
            toolbar.setIconSize(QSize(16, 16))
            act_view_table = QAction("View Table", self.cryst_dyn_win)
            act_view_table.triggered.connect(self.Viewtable)
            act_export = QAction("Export Table", self.cryst_dyn_win)
            act_export.triggered.connect(lambda: self.exportdynamics(self.crystdynamics, "cryst_dynamics.csv"))
            toolbar.addAction(act_view_table)
            toolbar.addAction(act_export)
            layout.addWidget(toolbar)

            splitter = QSplitter(Qt.Orientation.Horizontal)

            # Left plots
            left_widget = QWidget()
            left_layout = QVBoxLayout(left_widget)
            left_layout.setContentsMargins(0, 0, 0, 0)
            
            self.multiplot = MplMultiCanvas(self, width=8, height=6, dpi=100)
            multi_toolbar = NavigationToolbar(self.multiplot, self.cryst_dyn_win)
            left_layout.addWidget(multi_toolbar)
            
 
            self.multiplot.axes_Rm.set_xlabel("Time(s)")
            self.multiplot.axes_Rm.set_ylabel("Rm(Ω)")
            
            self.multiplot.axes_Fs.set_xlabel("Time(s)")
            self.multiplot.axes_Fs.set_ylabel("fs(Hz)")
            

            self.multiplot.axes_Lm.set_xlabel("Time(s)")
            self.multiplot.axes_Lm.set_ylabel("Lm(H)")
            
            self.multiplot.axes_Cm.set_xlabel("Time(s)")
            self.multiplot.axes_Cm.set_ylabel("Cm(F)")
            
            left_layout.addWidget(self.multiplot)
            splitter.addWidget(left_widget)

            # Right controls
            right_widget = QWidget()
            right_layout = QVBoxLayout(right_widget)
            group_box = QGroupBox("Latest BVD Parameters")
            form = QFormLayout()
            
            if not hasattr(self, 'rm_edit'):
                self.rm_edit = QLineEdit()
                self.rm_edit.setReadOnly(True)
            if not hasattr(self, 'lm_edit'):
                self.lm_edit = QLineEdit()
                self.lm_edit.setReadOnly(True)
            if not hasattr(self, 'cm_edit'):
                self.cm_edit = QLineEdit()
                self.cm_edit.setReadOnly(True)
            if not hasattr(self, 'c0_edit'):
                self.c0_edit = QLineEdit()
                self.c0_edit.setReadOnly(True)
            if not hasattr(self, 'f_edit'):
                self.f_edit = QLineEdit()
                self.f_edit.setReadOnly(True)
                
            form.addRow("Fs (Hz):", self.f_edit)
            form.addRow("Rm (Ω):", self.rm_edit)
            form.addRow("Lm (H):", self.lm_edit)
            form.addRow("Cm (F):", self.cm_edit)
            form.addRow("C0 (F):", self.c0_edit)
            
            self.fit_button = QPushButton("Fit BVD")
            form.addRow(self.fit_button)
            group_box.setLayout(form)
            right_layout.addWidget(group_box)
            right_layout.addStretch()
            splitter.addWidget(right_widget)

            layout.addWidget(splitter)
            
            self.cryst_dyn_win.show()
            
            self.fit_button.clicked.connect(self.fit_data)
            self.plotcrystallization()

        except Exception as e:
            QMessageBox.warning(self, "Window Error", str(e))

    def fit_data(self):
        """Fit BVD parameters to current data"""
        try:
            if self.data.empty:
                QMessageBox.warning(self, "No Data", "No data available for fitting")
                return
            
            # Get the latest sweep data
            points = int(self.sweep_points.text() or 201)
            if len(self.data) < points:
                QMessageBox.warning(self, "Insufficient Data", "Need at least one complete sweep")
                return
            
            # Use the last complete sweep
            last_sweep = self.data.tail(points)
            freqs = pd.to_numeric(last_sweep["Frequency(Hz)"], errors='coerce').dropna().values
            resist = pd.to_numeric(last_sweep["Resistance(Ω)"], errors='coerce').fillna(0).values
            react = pd.to_numeric(last_sweep["Reactance(Ω)"], errors='coerce').fillna(0).values
            
            if len(freqs) == 0:
                QMessageBox.warning(self, "Data Error", "No valid frequency data")
                return
            
            impedance = resist + 1j * react
            
            try:
                Rm, Lm, Cm, C0, fs, Q = parameter(freqs, impedance, resist)
                
                # Update the display fields
                self.rm_edit.setText(f"{Rm:.6f}")
                self.lm_edit.setText(f"{Lm:.6e}")
                self.cm_edit.setText(f"{Cm:.6e}")
                self.c0_edit.setText(f"{C0:.6e}")
                self.f_edit.setText(f"{fs:.2f}")
                
                QMessageBox.information(self, "Fit Complete", "BVD parameters fitted successfully")
                
            except ImportError:
                QMessageBox.warning(self, "Missing Model", "Butterworth model not found")
            except Exception as e:
                QMessageBox.warning(self, "Fit Error", f"Error fitting BVD parameters: {str(e)}")
                
        except Exception as e:
            QMessageBox.warning(self, "Fit Error", str(e))

    def plotcrystallization(self):
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
            self.crystdynamics = pd.DataFrame({
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
            if 'C0' in locals():
                self.c0_edit.setText(f"{C0:.6e}")
            self.f_edit.setText(f"{fs_values[-1]:.2f}")

            # Plot
            self.multiplot.axes_Rm.clear()
            self.multiplot.axes_Rm.plot(t_seconds, rm_values, 'bo-')

            self.multiplot.axes_Fs.clear()
            self.multiplot.axes_Fs.plot(t_seconds, fs_values, 'ro-')

            self.multiplot.axes_Lm.clear()
            self.multiplot.axes_Lm.plot(t_seconds, lm_values, 'go-')

            self.multiplot.axes_Cm.clear()
            self.multiplot.axes_Cm.plot(t_seconds, cm_values, 'mo-')
            
            self.multiplot.draw()

        except Exception as e:
            QMessageBox.warning(self, "Plotting Error", str(e))

    def Viewtable(self):
        """Show computed dynamics table in a separate window."""
        try:
            if not hasattr(self, 'crystdynamics') or self.crystdynamics.empty:
                QMessageBox.warning(self, "No Data", "Run dynamics analysis first.")
                return
                
            table_win = QWidget()
            table_win.setWindowTitle("Crystallization Dynamics Table")
            table_win.resize(800, 600)
            layout = QVBoxLayout(table_win)
            model = TableModel(self.crystdynamics)
            table = QTableView()
            table.setModel(model)
            table.resizeColumnsToContents()
            layout.addWidget(table)
            table_win.show()
            
            self.table_win = table_win
            
        except Exception as e:
            QMessageBox.warning(self, "Table Error", str(e))

    # Crystallization kinetics (Avrami)
    def crystallizationkinetics(self):
        try:
            if hasattr(self, 'cryst_kin_win') and self.cryst_kin_win is not None:
                try:
                    self.cryst_kin_win.close()
                    self.cryst_kin_win.deleteLater()
                except Exception:
                    pass
                
            self.cryst_kin_win = None
            self.cryst_kin_win = QWidget()
            self.cryst_kin_win.setWindowTitle("Crystallization Kinetics (Avrami)")
            self.cryst_kin_win.resize(1000, 700)
            layout = QVBoxLayout(self.cryst_kin_win)

            splitter = QSplitter(Qt.Orientation.Horizontal)

            # Left plot
            left_widget = QWidget()
            left_layout = QVBoxLayout(left_widget)
            self.plot_crystallization_fraction = MplCanvas(self, width=5, height=4, dpi=100)
            kinetics_toolbar = NavigationToolbar(self.plot_crystallization_fraction, self.cryst_kin_win)
            left_layout.addWidget(kinetics_toolbar)
            self.plot_crystallization_fraction.axes.set_title("Crystallinity vs Time")
            self.plot_crystallization_fraction.axes.set_ylabel("X(t)")
            self.plot_crystallization_fraction.axes.set_xlabel("Time (s)")
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
            fit_btn.clicked.connect(self.plotkineticsdata)
            form.addRow(fit_btn)
            group.setLayout(form)
            right_layout.addWidget(group)
            splitter.addWidget(right_widget)

            layout.addWidget(splitter)
            self.cryst_kin_win.show()

            self.plotkineticsdata()

        except Exception as e:
            QMessageBox.warning(self, "Window Error", str(e))

    def plotkineticsdata(self):
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
            self.plot_crystallization_fraction.axes.clear()
            self.plot_crystallization_fraction.axes.plot(t_seconds, X_actual, 'bo-')
            self.plot_crystallization_fraction.draw()

            # Fit Avrami
            try:
                k, n = avrami_fit(t_seconds, fvals)
                self.k_edit.setText(f"{k:.3e}")
                self.n_edit.setText(f"{n:.3f}")
                X_fit = avrami_formula(k, n, t_seconds)
                self.plot_crystallization_fraction.axes.plot(t_seconds, X_fit, 'r-')
                self.plot_crystallization_fraction.draw()
            except Exception as e:
                QMessageBox.warning(self, "Fit Error", str(e))

        except Exception as e:
            QMessageBox.warning(self, "Plotting Error", str(e))

    def exportdynamics(self, df, default_name):
        try:
            if df is None or df.empty:
                QMessageBox.warning(self, "Export", "No data to export.")
                return
            fname, _ = QFileDialog.getSaveFileName(self, "Save CSV", default_name, "CSV files (*.csv)")
            if not fname:
                return
            df.to_csv(fname, index=False)
            self
        except Exception as e:
            QMessageBox.warning(self, "Exporting Error", str(e))

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