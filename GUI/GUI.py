import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import pyvisa
from VNA_Data.Acquire_data import acquire_data
from Models.Butterworth import half_power_threshold, parameter, fit_data, butterworth
from Models.Avrami import compute_X_t, fit, formula, validate_data
from Models.Sauerbrey import parameter as sauerbrey_parameter, sauerbrey
from Models.konazawa import parameter as konazawa_parameter, konazawa
from PyQt6.QtCore import Qt, QTimer, QSize, QAbstractTableModel
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFileDialog, QGroupBox, QSplitter, QWidget, QTableView, QLineEdit, QToolBar,
    QHBoxLayout, QMessageBox, QPushButton, QLabel, QVBoxLayout, QMainWindow,
    QApplication, QStatusBar, QFormLayout, QComboBox
)

class TableModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)
        return None

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Orientation.Vertical:
                return str(section)
        return None

    def flags(self, index):
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEditable

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role == Qt.ItemDataRole.EditRole:
            self._data.iloc[index.row(), index.column()] = value
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole])
            return True
        return False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GUI for Network Analyzer")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)

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
        

        self.setStatusBar(QStatusBar(self))

        view_menu.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        self.crystallization_action = QAction("Crystallization Dynamics & Kinetics", self)
        self.sauerbrey_action = QAction("Sauerbrey & Konazawa", self)
        view_menu.addAction(self.crystallization_action)
        view_menu.addAction(self.sauerbrey_action)


        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(main_splitter)

        # Left panel (controls)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        # Metadata group
        group_box = QGroupBox("Metadata")
        layout = QFormLayout()
        self.sample_name = QLineEdit()
        self.batch_number = QLineEdit()
        self.coating_solution = QLineEdit()
        self.notes = QLineEdit()
        self.operator_name = QLineEdit()
        layout.addRow("Sample Name:", self.sample_name)
        layout.addRow("Batch Number:", self.batch_number)
        layout.addRow("Coating solution:", self.coating_solution)
        layout.addRow("Notes:", self.notes)
        layout.addRow("Operator Name:", self.operator_name)
        group_box.setLayout(layout)
        left_layout.addWidget(group_box)

        # Sweep Frequency Range
        group_box2 = QGroupBox("Sweep Frequency Range")
        range_layout = QHBoxLayout()
        self.label2 = QLabel("Start")
        self.start_frequency = QLineEdit()
        self.label3 = QLabel("End")
        self.end_frequency = QLineEdit()
        self.label_points = QLabel("Points")
        self.sweep_points = QLineEdit()
        self.sweep_points.setPlaceholderText("e.g., 201")
        self.update_button = QPushButton("Update")
        range_layout.addWidget(self.label_points)
        range_layout.addWidget(self.sweep_points)
        range_layout.addWidget(self.label2)
        range_layout.addWidget(self.start_frequency)
        range_layout.addWidget(self.label3)
        range_layout.addWidget(self.end_frequency)
        range_layout.addWidget(self.update_button)
        group_box2.setLayout(range_layout)
        left_layout.addWidget(group_box2)

        # Controls
        group_box4 = QGroupBox("Controls")
        table_button = QHBoxLayout()
        self.start_logging = QPushButton("Start Logging")
        self.stop_logging = QPushButton("Stop Logging")
        self.upload_button = QPushButton("Upload Data")
        table_button.addWidget(self.upload_button)
        table_button.addWidget(self.start_logging)
        table_button.addWidget(self.stop_logging)
        self.start_logging.setEnabled(True)
        self.stop_logging.setEnabled(False)
        group_box4.setLayout(table_button)
        left_layout.addWidget(group_box4)
        left_layout.addStretch()


        # Data Acquisition group
        group_box1 = QGroupBox("Data Acquisition")
        layout1 = QVBoxLayout()
        self.label = QLabel("Select the Instrument")
        layout1.addWidget(self.label)
        self.combo = QComboBox()
        self.combo.setEditable(True)
        layout1.addWidget(self.combo)
        button_layout = QHBoxLayout()
        self.connect_button = QPushButton("Connect")
        self.rescan_button = QPushButton("Rescan")
        button_layout.addWidget(self.connect_button)
        button_layout.addWidget(self.rescan_button)
        layout1.addLayout(button_layout)
        group_box1.setLayout(layout1)
        left_layout.addWidget(group_box1)

        # Right panel (table and plot)
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)

        # Sauerbrey and Konazawa data
        self.label1_sauer = QLabel("Resonant Frequency(f0)")
        self.resonant = QLineEdit()
        self.label4_sauer = QLabel("Density of Quartz(ρ)")
        self.density = QLineEdit()
        self.label5_sauer = QLabel("Shear Modulus(µ)")
        self.shear = QLineEdit()
        self.label6_sauer = QLabel("Active Area of Electrode(A)")
        self.area = QLineEdit()

        # Table
        self.data = pd.DataFrame(columns=["Time", "Frequency(Hz)", "Resistance(Ω)", "Phase"])
        self.model = TableModel(self.data)
        self.model.dataChanged.connect(self.update_plot)
        self.table = QTableView()
        self.table.setModel(self.model)

        right_splitter.addWidget(self.table)

        self.freqs = np.linspace(1e6, 10e6, 201)
        self.resistance = np.zeros_like(self.freqs)
        self.phase = np.zeros_like(self.freqs)
        self.time_array = np.linspace(0, 10, len(self.freqs))
        self.fs = []
        self.rm = []


        # Avrami calculation
        self.f0 = self.freqs[0] if len(self.freqs) > 0 else 0
        self.finf = self.freqs[-1] if len(self.freqs) > 0 else 0
        try:
         self.X = compute_X_t(self.freqs, self.f0, self.finf)
        except Exception as e:
         self.X = np.zeros_like(self.freqs)  # fallback if compute_X_t fails
         print("Avrami X(t) calculation error:", e)

        # Plot the data
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("Resistance vs Frequency")
        self.plot_widget.setLabel("left", "Resistance (Ω)")
        self.plot_widget.setLabel("bottom", "Frequency(Hz)")
        self.plot_widget.plot(self.freqs, self.resistance, pen='b')
        right_splitter.addWidget(self.plot_widget)

        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([400, 1000])

        self.timer = QTimer()
        self.timer.timeout.connect(self.continuous_logging)

        self.rm = pyvisa.ResourceManager()
        self.resource_map = {}

        self.connect_button.clicked.connect(self.connect_to_instrument)
        self.rescan_button.clicked.connect(self.rescan_button_clicked)
        self.button_action8.triggered.connect(self.insert_button_clicked)
        self.upload_button.clicked.connect(self.upload_button_clicked)
        self.start_logging.clicked.connect(self.start_logging_button)
        self.stop_logging.clicked.connect(self.stop_logging_button)
        self.update_button.clicked.connect(self.update_buttons)
        self.crystallization_action.triggered.connect(self.crystallizationdynamicskinetics)
        self.sauerbrey_action.triggered.connect(self.sauerbrey_konazawa)

        self.rescan_button_clicked()

    def update_buttons(self):
        start = self.start_frequency.text()
        end = self.end_frequency.text()
        if not start or not end:
            QMessageBox.warning(self, "Input Error", "Please enter both start and end frequencies.")
            return
        try:
            start_val = float(start)
            end_val = float(end)
            if start_val >= end_val:
                QMessageBox.warning(self, "Input Error", "Start frequency must be less than end frequency.")
                return
            QMessageBox.information(self, "Sweep Range Updated", f"Sweep range set: {start_val} Hz to {end_val} Hz.")
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values for frequencies.")
        try:
            points_val = int(self.sweep_points.text())
            if points_val < 10 or points_val > 2001:
                QMessageBox.warning(self, "Input Error", "Sweep points should be between 10 and 2001.")
                return
        except ValueError:
                QMessageBox.warning(self, "Input Error", "Please enter a valid integer for sweep points.")
                return




    def start_logging_button(self):
        try:
            if self.sweep_points.text():
                QMessageBox.warning(self, "Missing Reference", "Please enter the sweep points before starting logging.")
                return
            self.timer.start(2000)
            self.start_logging.setEnabled(False)
            self.stop_logging.setEnabled(True)
        except Exception as e:
            QMessageBox.warning(self, "Error in continuous logging", str(e))

    def stop_logging_button(self):
        try:
            self.timer.stop()
            self.start_logging.setEnabled(True)
            self.stop_logging.setEnabled(False)
        except Exception as e:
            QMessageBox.warning(self, "Error in stop_logging", str(e))

    def update_plot(self):
        self.plot_widget.clear()
        if not self.data.empty and "Frequency(Hz)" in self.data.columns and "Resistance(Ω)" in self.data.columns:
            try:
                x = pd.to_numeric(self.data["Frequency(Hz)"], errors='coerce')
                y = pd.to_numeric(self.data["Resistance(Ω)"], errors='coerce')
                mask = x.notnull() & y.notnull()
                self.plot_widget.plot(x[mask], y[mask], pen='r', symbol='o', symbolSize=5, symbolBrush='b')
            except Exception:
                pass

    def continuous_logging(self):
        try:
            timestamps = pd.to_datetime(self.data["Timestamp"], errors="coerce")
            t_seconds = (timestamps - timestamps.min()).dt.total_seconds()
            start = float(self.start_frequency.text() or 1e6)
            stop = float(self.end_frequency.text() or 10e6)
            points = int(self.sweep_points.text() or "201")
            freqs, resistances, impedance, _, _, _ = acquire_data(self.combo.currentText(), start, stop, points)
            self.impedance = np.array(impedance)
            Rm, Lm, Cm, C0, fs = parameter(freqs, self.impedance)
            initial_guess = [Rm, Lm, Cm, C0]
            result = least_squares(
            lambda params: np.concatenate([
                np.real(butterworth(freqs, *params) - self.impedance),
                np.imag(butterworth(freqs, *params) - self.impedance)
           ]),
            initial_guess
         )
            self.Z_fit = butterworth(freqs, *result.x)
            self.fit_btn.setEnabled(True)

            abs_freq = float(self.abs_frequency.text())
            abs_res = float(self.abs_resistance.text())

            rows = [{
                "Time": t_seconds,
                "Frequency(Hz)": f,
                "Resistance(Ω)": r,
                "Phase": 0
            } for f, r in zip(freqs, resistances)]
            
            self.rm.append(Rm)
            self.fs.append(fs)

            self.data = pd.concat([self.data, pd.DataFrame(rows)], ignore_index=True)
            self.model = TableModel(self.data)
            self.model.dataChanged.connect(self.update_plot)
            self.table.setModel(self.model)

            self.update_plot()
        except Exception as e:
            QMessageBox.warning(self, "Error in logging", str(e))

    def rescan_button_clicked(self):
        self.combo.clear()
        self.resource_map = {}
        try:
            self.resources = self.rm.list_resources()
            if not self.resources:
                QMessageBox.critical(self, "No Instruments", "No instrument found.")
                return
            for res in self.resources:
                try:
                    instr = self.rm.open_resource(res)
                    idn = instr.query("*IDN?")
                    display = f"{idn.strip()} ({res})"
                    self.resource_map[display] = res
                    self.combo.addItem(display)
                except Exception:
                    continue
        except Exception as e:
            QMessageBox.critical(self, "VISA Error", str(e))

    def connect_to_instrument(self):
        selected = self.combo.currentText()
        resource = self.resource_map.get(selected, selected)
        try:
            instr = self.rm.open_resource(resource)
            idn = instr.query("*IDN?")
            QMessageBox.information(self, "Connected", f"Instrument ID: {idn}")

            start = float(self.start_frequency.text() or 1e6)
            stop = float(self.end_frequency.text() or 10e6)
            points = int(self.sweep_points.text() or "201")
            freqs, resistances, impedance, _, _, _ = acquire_data(resource, start, stop, points)
            self.impedance = np.array(impedance)

            Rm, Lm, Cm, C0, fs = parameter(freqs, self.impedance)
            self.Rm, self.Lm, self.Cm, self.C0, self.fs = parameter(freqs, self.impedance)

            # Update the UI with calculated parameters
            if hasattr(self, 'rm_edit'):
               self.rm_edit.setText(f"{self.Rm:.6f}")
               self.lm_edit.setText(f"{self.Lm:.6e}")
               self.cm_edit.setText(f"{self.Cm:.6e}")
               self.c0_edit.setText(f"{self.C0:.6e}")

            initial_guess = [Rm, Lm, Cm, C0]
            result = least_squares(
                lambda params: np.concatenate([
                    np.real(butterworth(freqs, *params) - self.impedance),
                    np.imag(butterworth(freqs, *params) - self.impedance)
                ]),
                initial_guess
            )
            self.Z_fit = butterworth(freqs, *result.x)
            self.fit_btn.setEnabled(True)
            timestamps = pd.to_datetime(self.data["Timestamp"], errors="coerce")
            t_seconds = (timestamps - timestamps.min()).dt.total_seconds()
            self.data = pd.DataFrame({
                "Timestamp": [t_seconds] * len(freqs),
                "Frequency(Hz)": freqs,
                "Resistance(Ω)": resistances,
                "Phase": [0] * len(freqs)
            })

            self.model = TableModel(self.data)
            self.model.dataChanged.connect(self.update_plot)
            self.table.setModel(self.model)

            self.update_plot()

        except Exception as e:
            QMessageBox.critical(self, "Connection Error", str(e))

    def insert_button_clicked(self):
        self.data = pd.DataFrame({
            "Timestamp": ["" for _ in range(10)],
            "Frequency(Hz)": ["" for _ in range(10)],
            "Resistance(Ω)": ["" for _ in range(10)],
            "Phase": ["" for _ in range(10)]
        })
        self.model = TableModel(self.data)
        self.model.dataChanged.connect(self.update_plot)
        self.table.setModel(self.model)
        
        self.button1 = QPushButton("Insert Row")
        self.button1.clicked.connect(self.insert_row)
        self.button2 = QPushButton("Delete Row")
        self.button2.clicked.connect(self.delete_row)
        
    def insert_row(self):
        new_row = pd.DataFrame({
            "Timestamp": [""],
            "Frequency": [""],
            "Resistance(Ω)": [""],
            "Phase": [""]
        })
        curr_row = self.table.currentRow()
        if curr_row == -1:
            self.data = pd.concat([self.data, new_row], ignore_index=True)
        else:
            top = self.data.iloc[:curr_row + 1]
            bottom= self.data.iloc[curr_row + 1:]
            self.data = pd.concat([top, new_row, bottom], ignore_index=True)
            
        
            
    def delete_row(self):
        curr_row = self.table.currentRow().row()
        if curr_row < 0:
            return QMessageBox.warning("Please select a Row to delete", self)
        button = QMessageBox.question(self, "Are you sure, You want to Delete this row", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if button == QMessageBox.StandardButton.Yes:
            self.data = self.data.drop(index=curr_row).reset_index(drop=True)
            
            self.model = TableModel(self.data)
            self.model.dataChanged.connect(self.update_plot)
            self.table.setModel(self.model)
            
            
            
            

    def upload_button_clicked(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select File", "", "CSV (*.csv);;Excel (*.xlsx)")
        try:
            if not file:
                return
            self.data = pd.read_csv(file) if file.endswith(".csv") else pd.read_excel(file)
            for col in ["Timestamp", "Frequency(Hz)", "Resistance(Ω)", "Phase"]:
                if col not in self.data.columns:
                    self.data[col] = ""
            self.model = TableModel(self.data)
            self.model.dataChanged.connect(self.update_plot)
            self.table.setModel(self.model)
            
            self.sweep_points.setText("1")

            self.update_plot()
        except Exception as e:
            QMessageBox.warning(self, "File Error", str(e))

    def crystallizationdynamicskinetics(self):
        try:
            # Create a new window for crystallization dynamics & kinetics
            self.crystallization_widget = QWidget()
            self.crystallization_widget.setWindowTitle("Crystallization Dynamics & Kinetics")
            self.crystallization_widget.resize(1400, 1200)
            main_layout = QVBoxLayout(self.crystallization_widget)
            main_splitter = QSplitter(Qt.Orientation.Horizontal)
            

              # Left panel (controls)
            left_widget = QWidget()
            left_layout = QVBoxLayout(left_widget)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(5)
            
            right_splitter = QSplitter(Qt.Orientation.Vertical)
            right_widget = QWidget()
            right_layout = QVBoxLayout(right_widget)
            right_layout.setSpacing(10)

            # Section: Plots
            plot_layout = QVBoxLayout()

            self.plot_resistance = pg.PlotWidget()
            self.plot_resistance.setTitle("Motional Resistance vs Time")
            self.plot_resistance.setLabel("left", "Motional Resistance (Ω)")
            self.plot_resistance.setLabel("bottom", "Time (s)")
            plot_layout.addWidget(self.plot_resistance)

            self.plot_frequency = pg.PlotWidget()
            self.plot_frequency.setTitle("Resonance Frequency vs Time")
            self.plot_frequency.setLabel("left", "Resonance Frequency (Hz)")
            self.plot_frequency.setLabel("bottom", "Time (s)")
            plot_layout.addWidget(self.plot_frequency)

            self.plot_crystallization_fraction = pg.PlotWidget()
            self.plot_crystallization_fraction.setTitle("Crystallization Fraction X(t) vs Time")
            plot_layout.addWidget(self.plot_crystallization_fraction)

            # Section: Parameter Inputs
            #BVD
            group_box = QGroupBox("Butterworth Van Dyke Model")

            param_form1 = QFormLayout()
            self.rm_edit = QLineEdit()
            self.lm_edit = QLineEdit()
            self.cm_edit = QLineEdit()
            self.c0_edit = QLineEdit()
            self.f_edit = QLineEdit()

            param_form1.addRow("Frequency (f):", self.f_edit)
            param_form1.addRow("Motional Resistance (Rm):", self.rm_edit)
            param_form1.addRow("Motional Inductance (Lm):", self.lm_edit)
            param_form1.addRow("Motional Capacitance (Cm):", self.cm_edit)
            param_form1.addRow("Static Capacitance (C0):", self.c0_edit)
            self.fit_btn = QPushButton("Fit BVD Model")
            param_form1.addRow(self.fit_btn)
            
            
            #Avrami
            group_box1 = QGroupBox("Avrami Model")

            param_form2 = QFormLayout()
            self.f0_edit = QLineEdit()
            self.finf_edit = QLineEdit()
            self.t_edit = QLineEdit()
            self.k_edit = QLineEdit()
            self.n_edit = QLineEdit()
            param_form2.addRow("Initial Frequency (f₀):", self.f0_edit)
            param_form2.addRow("Final Frequency (f_inf):", self.finf_edit)
            param_form2.addRow("Time (t):", self.t_edit)
            param_form2.addRow("Crystallization Rate (k):", self.k_edit)
            param_form2.addRow("Avrami Exponent (n):", self.n_edit)
            self.fit_btn1 = QPushButton("Fit Avrami")
            param_form2.addRow(self.fit_btn1)

            group_box.setLayout(param_form1)
            group_box1.setLayout(param_form2)
            right_splitter.addWidget(group_box)
            right_splitter.addWidget(group_box1)

            # Fit Button
            self.fit_btn1.clicked.connect(self.fit_data1)
            self.fit_btn.clicked.connect(self.fit_data)
            self.fit_btn1.setEnabled(True)
            self.fit_btn.setEnabled(True)

            # Add to main layout
            left_layout.addLayout(plot_layout)
            left_widget.setLayout(left_layout)
            main_splitter.addWidget(left_widget)
            main_splitter.addWidget(right_splitter)
            main_layout.addWidget(main_splitter)

            # Plot data from main table
            if not self.data.empty:
                try:
                    points = int(self.sweep_points.text() or "201")
                    total_rows = len(self.data)

                    if points > 1 and total_rows % points != 0:
                        QMessageBox.warning(self, "Data Error", "Data length is not divisible by number of sweep points.")
                        return

                    num_sweeps = total_rows // points
                    if num_sweeps == 0:
                        QMessageBox.warning(self, "Data Error", "No data available for plotting.")
                        return

                    t_seconds = []
                    rm_values = []
                    fs_values = []

                    for i in range(num_sweeps):
                        start = i * points
                        end = start + points
                        sweep = self.data.iloc[start:end]

                        freq = pd.to_numeric(sweep["Frequency(Hz)"].to_numpy(), errors='coerce')

                        if "Resistance(Ω)" in sweep.columns and "Reactance(Ω)" in sweep.columns:
                            resistance = pd.to_numeric(sweep["Resistance(Ω)"], errors='coerce')
                            reactance = pd.to_numeric(sweep["Reactance(Ω)"], errors='coerce')
                            impedance = resistance + 1j * reactance
                        elif "Impedance(Ω)" in sweep.columns:
                            impedance = pd.to_numeric(sweep["Impedance(Ω)"], errors='coerce')
                            resistance = impedance.real
                        else:
                            QMessageBox.warning(self, "Data Error", "No valid impedance data found for fitting.")
                            return

                        try:
                            Rm, Lm, Cm, C0, fs = parameter(freq, impedance, resistance)
                        except Exception as e:
                            QMessageBox.warning(self, "Fitting Error", f"Error in parameter extraction at sweep {i+1}: {str(e)}")
                            continue

                        self.Rm, self.Lm, self.Cm, self.C0, self.fs = Rm, Lm, Cm, C0, fs

                        rm_values.append(Rm)
                        fs_values.append(fs)

                        ts = pd.to_datetime(sweep["Timestamp"], errors="coerce")
                        avg_time = (ts - ts.min()).dt.total_seconds().mean() if ts.notna().all() else i
                        t_seconds.append(avg_time)

                    # Convert to arrays
                    t_seconds = np.array(t_seconds)
                    F = np.array(fs_values)
                    R = np.array(rm_values)
                    timestamps = np.array(t_seconds)

                    # Plot
                    self.plot_resistance.clear()
                    self.plot_resistance.plot(t_seconds, rm_values, pen='b', symbol='o', symbolSize=5, symbolBrush='r', name='Motional Resistance')

                    self.plot_frequency.clear()
                    self.plot_frequency.plot(t_seconds, fs_values, pen='r', symbol='o', symbolSize=5, symbolBrush='g', name='Resonance Frequency')

                    # Set fitted values in UI
                    if hasattr(self, "Rm"):
                        self.rm_edit.setText(f"{self.Rm:.6f}")
                        self.lm_edit.setText(f"{self.Lm:.6e}")
                        self.cm_edit.setText(f"{self.Cm:.6e}")
                        self.c0_edit.setText(f"{self.C0:.6e}")
                        self.f_edit.setText(f"{self.fs:.2f}")
                    else:
                        raise AttributeError("Parameters not available for BVD fit.")

                    # BVD fitting (optional)
                    if self.fit_btn.isEnabled():
                        try:
                           mask_f = np.isfinite(F)
                           mask_r = np.isfinite(R)

                           popt = fit_data(t_seconds[mask_f], F[mask_f], R[mask_r])
                           self.rm_edit.setText(f"{popt[0]:.6f}")
                           self.lm_edit.setText(f"{popt[1]:.6e}")
                           self.cm_edit.setText(f"{popt[2]:.6e}")
                           self.c0_edit.setText(f"{popt[3]:.6e}")
                           self.f_edit.setText(f"{popt[4]:.2f}")
                           self.plot_resistance.plot(t_seconds[mask_r], butterworth(F[mask_f], *popt), pen='m', name='BVD Fit')
                           QMessageBox.information(self.crystallization_widget, "BVD Fit", "BVD Model Fitting completed successfully!")
                        except Exception as e:
                            QMessageBox.warning(self.crystallization_widget, "BVD Fit Error", str(e))

                    # Avrami fitting
                    try:
                        points = int(self.sweep_points.text() or "201")
                        total_rows = len(self.data)
                        num_sweeps = total_rows // points

                        k_values = []
                        n_values = []
                        sweep_indices = []

                        for i in range(num_sweeps):
                            start = i * points
                            end = start + points

                            t_sweep = self.data['Time'][start:end]
                            freq_sweep = self.data['Frequency'][start:end]

                            try:
                                k, n = fit(t_sweep, freq_sweep)
                                k_values.append(k)
                                n_values.append(n)
                                sweep_indices.append(i)

                            except ValueError as e:
                                print(f"Sweep {i} skipped: {e}")

                        mask_f_valid = np.isfinite(F) & np.isfinite(t_seconds)
                        if mask_f_valid.sum() >= 3:
                            t_fit = t_seconds[mask_f_valid]
                            F_fit = F[mask_f_valid]

                            popt = fit(t_fit, F_fit)
                            k_val, n_val = popt
                            self.k_edit.setText(f"{k_val:.4e}")
                            self.n_edit.setText(f"{n_val:.2f}")

                            X_t = formula(k_val, n_val, t_seconds)
                            self.plot_crystallization_fraction.plot(t_seconds, X_t, pen='m')
                            QMessageBox.information(self.crystallization_widget, "Avrami Fit", f"Crystallization Rate k: {k_val:.4e}, Exponent n: {n_val:.2f}")
                        else:
                            QMessageBox.warning(self.crystallization_widget, "Avrami Fit Error", "Insufficient valid frequency data for fitting (need at least 3 points)")
                    except Exception as e:
                             QMessageBox.warning(self.crystallization_widget, "Avrami Fit Error", str(e))

                    self.crystallization_widget.show()

                except Exception as e:
                    QMessageBox.critical(self, "Processing Error", f"Unexpected error: {str(e)}")


        except Exception as e:
            QMessageBox.warning(self, "Window Error", str(e))
    
    def _validate_avrami_data(self, timestamps, freqs):
        """
        Helper method to validate data for Avrami fitting.
        """
        try:
            # Check basic data requirements
            if timestamps.empty or freqs.empty:
                return False, "No data available"
            
            # Check for sufficient valid data points
            mask = ~(timestamps.isna() | freqs.isna())
            valid_count = mask.sum()
            
            if valid_count < 3:
                return False, f"Insufficient valid data points ({valid_count}). Need at least 3."
            
            # Check for frequency variation
            freqs_clean = freqs[mask]
            freq_range = freqs_clean.max() - freqs_clean.min()
            
            if freq_range < 1e-6:
                return False, "Frequency data shows insufficient variation for meaningful fitting"
            
            return True, "Data validation passed"
            
        except Exception as e:
            return False, f"Data validation error: {str(e)}"
        
    def fit_data1(self):
        try:
            # Check if data is loaded
            if self.data.empty:
                QMessageBox.warning(self, "Data Error", "No data loaded. Please upload data first.")
                return
            
            timestamps = pd.to_datetime(self.data["Timestamp"], errors="coerce")
            t_seconds = (timestamps - timestamps.min()).dt.total_seconds()
            freqs = pd.to_numeric(self.data["Frequency(Hz)"], errors="coerce")

            # Validate data first
            is_valid, error_msg = self._validate_avrami_data(timestamps, freqs)
            if not is_valid:
                QMessageBox.warning(self, "Data Validation Error", error_msg)
                return

            # Remove NaN values
            mask = ~(timestamps.isna() | freqs.isna())
            t_clean = t_seconds[mask]
            freqs_clean = freqs[mask]

            # Check if manual f0 and finf are provided
            if self.f0_edit.text() and self.finf_edit.text():
                try:
                    f0 = float(self.f0_edit.text())
                    finf = float(self.finf_edit.text())
                    
                    if abs(f0 - finf) < 1e-10:
                        QMessageBox.warning(self, "Input Error", "Initial and final frequencies must be different.")
                        return
                        
                except ValueError:
                    QMessageBox.warning(self, "Input Error", "Initial and Final Frequency must be valid numbers.")
                    return
            else:
                # Use automatic detection from data
                f0 = freqs_clean[0]
                finf = freqs_clean[-1]
                self.f0_edit.setText(f"{f0:.2f}")
                self.finf_edit.setText(f"{finf:.2f}")

            # Perform the fit
            k, n = fit(t_clean, freqs_clean)
            self.k_edit.setText(f"{k:.4e}")
            self.n_edit.setText(f"{n:.2f}")

            # Plot the fitted curve
            X_fit = formula(k, n, t_seconds)
            self.plot_crystallization_fraction.clear()
            self.plot_crystallization_fraction.plot(t_seconds, X_fit, pen='m', name='Fitted Curve')
            
            # Also plot the actual crystallization fraction for comparison
            X_actual = compute_X_t(freqs, f0, finf)
            self.plot_crystallization_fraction.plot(t_seconds, X_actual, pen='b', symbol='o', symbolBrush='b', name='Actual Data')
            
            QMessageBox.information(self, "Avrami Fit Success", f"Fitting completed successfully!\nCrystallization Rate k: {k:.4e}\nAvrami Exponent n: {n:.2f}")

        except Exception as e:
            QMessageBox.warning(self, "Avrami Fit Error", str(e))
        

    def fit_data(self):
        try:
            if hasattr(self, "fit_plot_window") and self.fit_plot_window:
                self.fit_plot_window.close()
            self.fit_plot_window = pg.GraphicsLayoutWidget(show=True, title="Butterworth Model Fit")
            self.fit_plot_window.resize(800, 600)

            plot_widget = self.fit_plot_window.addPlot(title="Measured and Fitted Impedance vs Frequency")
            plot_widget.setLabel("left", "Impedance (Ohms)")
            plot_widget.setLabel("bottom", "Frequency (Hz)")

            if hasattr(self, "impedance") and hasattr(self, "Z_fit") and self.impedance is not None and self.Z_fit is not None:
                plot_widget.plot(self.freqs, np.abs(self.impedance), pen='b', name='Measured Impedance')
                plot_widget.plot(self.freqs, np.abs(self.Z_fit), pen='r', name='Fitted Impedance')
            else:
                QMessageBox.warning(self, "Fit Error", "Impedance data or fit result is missing.")
        except Exception as e:
            QMessageBox.warning(self, "Fit Error", str(e))

    def sauerbrey_konazawa(self):
        try:
            layout = QVBoxLayout()
            self.sauerbrey_konazawa_widget = QWidget()
            self.sauerbrey_konazawa_widget.setLayout(layout)
            self.sauerbrey_konazawa_widget.setWindowTitle("Sauerbrey & Konazawa")
            self.sauerbrey_konazawa_widget.resize(900, 700)
            self.sauerbrey_konazawa_widget.show()

            group_box = QGroupBox("Sauerbrey Equation Parameters")
            group_layout = QFormLayout()
            group_layout.addRow(self.label1_sauer, self.resonant)
            group_layout.addRow(self.label4_sauer, self.density)
            group_layout.addRow(self.label5_sauer, self.shear)
            group_layout.addRow(self.label6_sauer, self.area)
            group_box.setLayout(group_layout)
            layout.addWidget(group_box)

            group_box1 = QGroupBox("Konazawa Equation Parameters")
            group_layout1 = QFormLayout()
            group_layout1.addRow(self.label1_sauer, self.resonant)
            group_layout1.addRow(self.label4_sauer, self.density)
            group_layout1.addRow(self.label5_sauer, self.shear)
            group_layout1.addRow(self.label6_sauer, self.area)
            group_box1.setLayout(group_layout1)
            layout.addWidget(group_box1)

        except Exception as e:
            QMessageBox.warning(self, "Try Again by opening again, If the error persists, reinstall the GUI", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
