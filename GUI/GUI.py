import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import pyvisa
from test.test import acquire_data
from Models.Butterworth import half_power_threshold, parameter, fit_data, butterworth
from Models.Avrami import avrami, fit, formula
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
        self.calculate_button = QPushButton("Calculate Remaining values")
        table_button.addWidget(self.calculate_button)
        table_button.addWidget(self.upload_button)
        table_button.addWidget(self.start_logging)
        table_button.addWidget(self.stop_logging)
        self.start_logging.setEnabled(True)
        self.stop_logging.setEnabled(False)
        group_box4.setLayout(table_button)
        left_layout.addWidget(group_box4)
        left_layout.addStretch()

        # Reference Values
        ref_box = QGroupBox("Reference Values")
        ref_layout = QFormLayout()
        self.abs_frequency = QLineEdit()
        self.abs_resistance = QLineEdit()
        ref_layout.addRow("Absolute Frequency:", self.abs_frequency)
        ref_layout.addRow("Absolute Resistance:", self.abs_resistance)
        ref_box.setLayout(ref_layout)
        left_layout.addWidget(ref_box)

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
        self.data = pd.DataFrame(columns=["Timestamp", "Time", "Frequency(Hz)", "Resistance(Ω)", "Phase"])
        self.model = TableModel(self.data)
        self.model.dataChanged.connect(self.update_plot)
        self.table = QTableView()
        self.table.setModel(self.model)

        right_splitter.addWidget(self.table)

        self.freqs = np.linspace(1e6, 10e6, 201)
        self.resistance = np.zeros_like(self.freqs)
        self.phase = np.zeros_like(self.freqs)
        self.time_array = np.linspace(0, 10, len(self.freqs))
        self.fs = 0
        self.Rm = 0


        # Avrami calculation
        self.f0 = self.freqs[0] if len(self.freqs) > 0 else 0
        self.fmax = self.freqs[-1] if len(self.freqs) > 0 else 0
        self.ft = self.freqs[len(self.freqs)//2] if len(self.freqs) > 0 else 0
        self.X = avrami(self.freqs, self.f0, self.fmax, self.ft)
        self.X = np.full_like(self.freqs, self.X)

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
        self.calculate_button.clicked.connect(self.calculate_remaining_values)
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


    def calculate_remaining_values(self):
        try:
            self.data["Rel Frequency"] = pd.to_numeric(self.data["Frequency(Hz)"], errors='coerce') - float(self.abs_frequency.text())
            self.data["Rel Resistance"] = pd.to_numeric(self.data["Resistance(Ω)"], errors='coerce') - float(self.abs_resistance.text())

            self.model = TableModel(self.data)
            self.model.dataChanged.connect(self.update_plot)
            self.table.setModel(self.model)

            self.update_plot()


        except Exception as e:
            QMessageBox.warning(self, "Calculation Error", str(e))

    def start_logging_button(self):
        try:
            if not self.abs_frequency.text() or not self.abs_resistance.text():
                QMessageBox.warning(self, "Missing Reference", "Please enter the Absolute Frequency and Absolute Resistance before starting logging.")
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
            if hasattr(self, 'fit_btn'):
                self.fit_btn.setEnabled(True)

            abs_freq = float(self.abs_frequency.text())
            abs_res = float(self.abs_resistance.text())

            # Create proper timestamp
            current_time = pd.Timestamp.now()
            
            # Calculate time in seconds from start of logging
            if self.data.empty:
                t_seconds = 0
            else:
                if "Timestamp" in self.data.columns and not self.data["Timestamp"].empty:
                    first_timestamp = pd.to_datetime(self.data["Timestamp"].iloc[0])
                    t_seconds = (current_time - first_timestamp).total_seconds()
                else:
                    t_seconds = 0

            rows = [{
                "Timestamp": current_time,
                "Time": t_seconds,
                "Frequency(Hz)": f,
                "Resistance(Ω)": r,
                "Phase": 0
            } for f, r in zip(freqs, resistances)]

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
            initial_guess = [Rm, Lm, Cm, C0]
            result = least_squares(
                lambda params: np.concatenate([
                    np.real(butterworth(freqs, *params) - self.impedance),
                    np.imag(butterworth(freqs, *params) - self.impedance)
                ]),
                initial_guess
            )
            self.Z_fit = butterworth(freqs, *result.x)
            if hasattr(self, 'fit_btn'):
                self.fit_btn.setEnabled(True)
            
            # Store frequencies for later use
            self.freqs = freqs
            
            # Create proper timestamp
            current_time = pd.Timestamp.now()
            t_seconds = 0  # Initial measurement
            
            self.data = pd.DataFrame({
                "Timestamp": [current_time] * len(freqs),
                "Time": [t_seconds] * len(freqs),
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
            "Time": ["" for _ in range(10)],
            "Frequency(Hz)": ["" for _ in range(10)],
            "Resistance(Ω)": ["" for _ in range(10)],
            "Phase": ["" for _ in range(10)]
        })
        self.model = TableModel(self.data)
        self.model.dataChanged.connect(self.update_plot)
        self.table.setModel(self.model)


    def upload_button_clicked(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select File", "", "CSV (*.csv);;Excel (*.xlsx)")
        try:
            if not file:
                return
            self.data = pd.read_csv(file) if file.endswith(".csv") else pd.read_excel(file)
            for col in ["Timestamp", "Time", "Frequency(Hz)", "Resistance(Ω)", "Phase"]:
                if col not in self.data.columns:
                    self.data[col] = ""
            self.model = TableModel(self.data)
            self.model.dataChanged.connect(self.update_plot)
            self.table.setModel(self.model)

            self.update_plot()
        except Exception as e:
            QMessageBox.warning(self, "File Error", str(e))

    def crystallizationdynamicskinetics(self):
     try:
        main_layout = QVBoxLayout()
        self.crystallization_widget = QWidget()
        self.crystallization_widget.setLayout(main_layout)
        self.crystallization_widget.setWindowTitle("Crystallization Dynamics & Kinetics")
        self.crystallization_widget.resize(1000, 800)
        self.crystallization_widget.show()

        # Section: Plots
        plot_layout = QVBoxLayout()

        self.plot_resistance = pg.PlotWidget()
        self.plot_resistance.setTitle("Motional Resistance vs Time")
        self.plot_resistance.setLabel("left", "Resistance (Ω)")
        self.plot_resistance.setLabel("bottom", "Time (s)")
        plot_layout.addWidget(self.plot_resistance)

        self.plot_frequency = pg.PlotWidget()
        self.plot_frequency.setTitle("Resonance Frequency vs Time")
        self.plot_frequency.setLabel("left", "Frequency (Hz)")
        self.plot_frequency.setLabel("bottom", "Time (s)")
        plot_layout.addWidget(self.plot_frequency)

        self.plot_phase = pg.PlotWidget()
        self.plot_phase.setTitle("Phase vs Time")
        self.plot_phase.setLabel("left", "Phase (degrees)")
        self.plot_phase.setLabel("bottom", "Time (s)")
        plot_layout.addWidget(self.plot_phase)

        # Section: Parameter Inputs
        form_section = QHBoxLayout()

        param_form1 = QFormLayout()
        param_form1.addRow("Resonance Frequency (f):", QLineEdit())
        param_form1.addRow("Motional Resistance (Rm):", QLineEdit())
        param_form1.addRow("Motional Inductance (Lm):", QLineEdit())
        param_form1.addRow("Motional Capacitance (Cm):", QLineEdit())
        param_form1.addRow("Static Capacitance (C0):", QLineEdit())

        param_form2 = QFormLayout()
        param_form2.addRow("Initial Frequency (f₀):", QLineEdit())
        param_form2.addRow("Final Frequency (f_inf):", QLineEdit())
        param_form2.addRow("Time (t):", QLineEdit())
        param_form2.addRow("Crystallization Rate (k):", QLineEdit())
        param_form2.addRow("Avrami Exponent (n):", QLineEdit())

        form_section.addLayout(param_form1)
        form_section.addLayout(param_form2)

        # Fit Button
        self.fit_btn = QPushButton("Fit BVD Model")
        self.fit_btn.clicked.connect(self.fit_data)
        self.fit_btn.setEnabled(False)

        # Add to main layout
        main_layout.addLayout(plot_layout)
        main_layout.addLayout(form_section)
        main_layout.addWidget(self.fit_btn)

        # Plot data from main table
        if not self.data.empty:
            try:
                timestamps = pd.to_datetime(self.data["Timestamp"], errors="coerce")
                t_seconds = (timestamps - timestamps.min()).dt.total_seconds()

                R = pd.to_numeric(self.data["Resistance(Ω)"], errors="coerce")
                F = pd.to_numeric(self.data["Frequency(Hz)"], errors="coerce")
                P = pd.to_numeric(self.data["Phase"], errors="coerce")

                mask_r = timestamps.notnull() & R.notnull()
                self.plot_resistance.plot(t_seconds[mask_r], R[mask_r], pen='r', symbol='o', symbolBrush='b')

                mask_f = timestamps.notnull() & F.notnull()
                self.plot_frequency.plot(t_seconds[mask_f], F[mask_f], pen='g', symbol='x', symbolBrush='b')

                mask_p = timestamps.notnull() & P.notnull()
                self.plot_phase.plot(t_seconds[mask_p], P[mask_p], pen='c', symbol='t', symbolBrush='b')

            except Exception as e:
                QMessageBox.warning(self, "Plotting Error", str(e))

     except Exception as e:
        QMessageBox.warning(self, "Window Error", str(e))


    def fit_data(self):
        if hasattr(self, "fit_plot_window") and self.fit_plot_window:
            self.fit_plot_window.close()
        self.fit_plot_window = pg.GraphicsLayoutWidget(show=True, title="Butterworth Model Fit")
        self.fit_plot_window.resize(800, 600)

        plot_widget = self.fit_plot_window.addPlot(title="Measured and Fitted Impedance vs Frequency")
        plot_widget.setLabel("left", "Impedance (Ohms)")
        plot_widget.setLabel("bottom", "Frequency (Hz)")
        plot_widget.plot(self.freqs, np.abs(self.impedance), pen='b', name='Measured Impedance')
        plot_widget.plot(self.freqs, np.abs(self.Z_fit), pen='r', name='Fitted Impedance')

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
