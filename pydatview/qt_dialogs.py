"""Dialogs and table models used by the Qt main window."""

import os

import numpy as np

from pydatview.qt_compat import QtCore, QtGui, QtWidgets
from pydatview.qt_io import _format_specs, _parse_bladed_suffixes
from pydatview.qt_math import _MATH_FUNCTIONS, _TABLE_TRANSFORMS

class DataFrameModel(QtCore.QAbstractTableModel):
    def __init__(self, dataframe=None, max_rows=200):
        super().__init__()
        self.max_rows = max_rows
        self.dataframe = dataframe

    def set_dataframe(self, dataframe):
        self.beginResetModel()
        self.dataframe = dataframe
        self.endResetModel()

    def rowCount(self, parent=None):
        if self.dataframe is None:
            return 0
        return min(len(self.dataframe), self.max_rows)

    def columnCount(self, parent=None):
        if self.dataframe is None:
            return 0
        return len(self.dataframe.columns)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole or self.dataframe is None or not index.isValid():
            return None
        metadata = getattr(self.dataframe, 'attrs', {}).get('pydatview', {})
        if (
            metadata.get('lazy_values')
            and index.column() >= metadata.get('lazy_column_offset', 2)
        ):
            return '<loaded when plotted>'
        value = self.dataframe.iat[index.row(), index.column()]
        return "" if value is None else str(value)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole or self.dataframe is None:
            return None
        if orientation == QtCore.Qt.Horizontal:
            return str(self.dataframe.columns[section])
        return str(section)


class CalculationDialog(QtWidgets.QDialog):
    def __init__(
            self,
            columns,
            selected_columns=None,
            parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mathematical operation")
        self.resize(680, 470)
        self._columns = [str(column) for column in columns if str(column) != "Index"]

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        content = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(content, 1)

        column_panel = QtWidgets.QWidget()
        column_layout = QtWidgets.QVBoxLayout(column_panel)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.addWidget(QtWidgets.QLabel("VARIABLES"))
        self.column_filter = QtWidgets.QLineEdit()
        self.column_filter.setPlaceholderText("Filter variables")
        self.column_filter.setClearButtonEnabled(True)
        column_layout.addWidget(self.column_filter)
        self.column_list = QtWidgets.QListWidget()
        column_layout.addWidget(self.column_list, 1)
        content.addWidget(column_panel)

        expression_panel = QtWidgets.QWidget()
        expression_layout = QtWidgets.QFormLayout(expression_panel)
        expression_layout.setContentsMargins(8, 0, 0, 0)
        expression_layout.setSpacing(8)
        self.operation_mode = QtWidgets.QComboBox()
        self.operation_mode.addItem("Derived variable", "column")
        self.operation_mode.addItem("Transform entire file", "table")
        expression_layout.addRow("Mode", self.operation_mode)
        self.result_name = QtWidgets.QLineEdit()
        self.result_name.setText("Calculated")
        self.result_name_label = QtWidgets.QLabel("Result name")
        expression_layout.addRow(self.result_name_label, self.result_name)
        self.expression = QtWidgets.QPlainTextEdit()
        self.expression.setMaximumHeight(110)
        self.expression_label = QtWidgets.QLabel("Expression")
        expression_layout.addRow(self.expression_label, self.expression)
        self.function_combo = QtWidgets.QComboBox()
        expression_layout.addRow("Function", self.function_combo)
        content.addWidget(expression_panel)
        content.setSizes([260, 400])

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        self.add_button = buttons.addButton("Add and plot", QtWidgets.QDialogButtonBox.AcceptRole)
        self.add_button.setObjectName("primaryButton")
        root.addWidget(buttons)

        selected_columns = [str(column) for column in (selected_columns or [])]
        if len(selected_columns) >= 2:
            self.expression.setPlainText(
                "{{{}}} - {{{}}}".format(selected_columns[0], selected_columns[1])
            )
            self.result_name.setText("Difference")
        elif selected_columns:
            self.expression.setPlainText("{{{}}}".format(selected_columns[0]))

        transform_script = "trim(start=0, stop=1)"
        self._mode_values = {
            "column": (
                self.result_name.text(),
                self.expression.toPlainText(),
            ),
            "table": ("_trimmed", transform_script),
        }
        self._active_mode = "column"

        self.column_filter.textChanged.connect(lambda _text: self.populate_columns())
        self.column_list.itemDoubleClicked.connect(self.insert_column)
        self.function_combo.activated.connect(self.insert_function)
        self.operation_mode.currentIndexChanged.connect(self.on_mode_changed)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.update_function_combo()
        self.populate_columns()

    def mode(self):
        return self.operation_mode.currentData()

    def update_function_combo(self):
        self.function_combo.blockSignals(True)
        self.function_combo.clear()
        self.function_combo.addItem("Insert function")
        functions = (
            _TABLE_TRANSFORMS if self.mode() == "table" else _MATH_FUNCTIONS
        )
        self.function_combo.addItems(list(functions))
        self.function_combo.blockSignals(False)

    def on_mode_changed(self, _index=None):
        self._mode_values[self._active_mode] = (
            self.result_name.text(),
            self.expression.toPlainText(),
        )
        self._active_mode = self.mode()
        result_name, expression = self._mode_values[self._active_mode]
        self.result_name.setText(result_name)
        self.expression.setPlainText(expression)
        table_mode = self._active_mode == "table"
        self.result_name_label.setText(
            "Table suffix" if table_mode else "Result name"
        )
        self.expression_label.setText("Script" if table_mode else "Expression")
        self.expression.setToolTip(
            "One safe table transform per line; trim bounds are inclusive"
            if table_mode
            else "Expression used to calculate the new variable"
        )
        self.add_button.setText(
            "Transform file" if table_mode else "Add and plot"
        )
        self.update_function_combo()

    def populate_columns(self):
        text_filter = self.column_filter.text().strip().lower()
        self.column_list.clear()
        for column in self._columns:
            if text_filter and text_filter not in column.lower():
                continue
            item = QtWidgets.QListWidgetItem(column)
            item.setData(QtCore.Qt.UserRole, column)
            self.column_list.addItem(item)

    def insert_column(self, item):
        self.expression.insertPlainText("{{{}}}".format(item.data(QtCore.Qt.UserRole)))
        self.expression.setFocus()

    def insert_function(self, index):
        if index <= 0:
            return
        name = self.function_combo.itemText(index)
        self.expression.insertPlainText("{}()".format(name))
        cursor = self.expression.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.Left)
        self.expression.setTextCursor(cursor)
        self.expression.setFocus()
        self.function_combo.setCurrentIndex(0)

    def accept(self):
        if not self.result_name.text().strip():
            self.result_name.setFocus()
            return
        if not self.expression.toPlainText().strip():
            self.expression.setFocus()
            return
        super().accept()

    def values(self):
        return self.result_name.text().strip(), self.expression.toPlainText().strip()


class AxisLimitsDialog(QtWidgets.QDialog):
    def __init__(self, limits=None, logx=False, logy=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Axis limits")
        self.setMinimumWidth(390)
        self.logx = bool(logx)
        self.logy = bool(logy)
        self._limits = dict(limits or {})

        root = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)
        root.addLayout(form)

        form.addWidget(QtWidgets.QLabel("Axis"), 0, 0)
        form.addWidget(QtWidgets.QLabel("Minimum"), 0, 1)
        form.addWidget(QtWidgets.QLabel("Maximum"), 0, 2)
        self.edits = {}
        for row, (axis, minimum_key, maximum_key) in enumerate(
            (("X", "xmin", "xmax"), ("Y", "ymin", "ymax")),
            start=1,
        ):
            form.addWidget(QtWidgets.QLabel(axis), row, 0)
            for column, key in ((1, minimum_key), (2, maximum_key)):
                edit = QtWidgets.QLineEdit()
                edit.setPlaceholderText("Auto")
                value = self._limits.get(key)
                if value is not None:
                    edit.setText("{:.12g}".format(value))
                form.addWidget(edit, row, column)
                self.edits[key] = edit

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.auto_button = buttons.addButton("Reset to auto", QtWidgets.QDialogButtonBox.ResetRole)
        root.addWidget(buttons)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.auto_button.clicked.connect(self.reset_to_auto)

    @staticmethod
    def _parse_value(text):
        text = text.strip()
        if not text:
            return None
        value = float(text.replace(",", "."))
        if not np.isfinite(value):
            raise ValueError("Axis limits must be finite numbers")
        return value

    def accept(self):
        try:
            limits = {
                key: self._parse_value(edit.text())
                for key, edit in self.edits.items()
            }
            for axis, minimum_key, maximum_key, logarithmic in (
                ("X", "xmin", "xmax", self.logx),
                ("Y", "ymin", "ymax", self.logy),
            ):
                minimum = limits[minimum_key]
                maximum = limits[maximum_key]
                if minimum is not None and maximum is not None and minimum >= maximum:
                    raise ValueError("{} minimum must be less than {} maximum".format(axis, axis))
                if logarithmic and any(
                    value is not None and value <= 0 for value in (minimum, maximum)
                ):
                    raise ValueError("{} limits must be positive in logarithmic mode".format(axis))
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Axis limits", str(exc))
            return
        self._limits = limits
        super().accept()

    def reset_to_auto(self):
        self._limits = {key: None for key in ("xmin", "xmax", "ymin", "ymax")}
        for edit in self.edits.values():
            edit.clear()
        super().accept()

    def values(self):
        return dict(self._limits)


class FatigueDelDialog(QtWidgets.QDialog):
    def __init__(self, columns, selected_signal=None, selected_time=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fatigue / DEL analysis")
        self.setMinimumWidth(430)
        self._columns = [str(column) for column in columns]

        root = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        root.addLayout(form)

        self.signal_combo = QtWidgets.QComboBox()
        self.time_combo = QtWidgets.QComboBox()
        for column in self._columns:
            self.signal_combo.addItem(column)
            self.time_combo.addItem(column)
        if selected_signal in self._columns:
            self.signal_combo.setCurrentIndex(self._columns.index(selected_signal))
        else:
            for row, column in enumerate(self._columns):
                if not column.lower().startswith("index"):
                    self.signal_combo.setCurrentIndex(row)
                    break
        if selected_time in self._columns:
            self.time_combo.setCurrentIndex(self._columns.index(selected_time))
        else:
            for row, column in enumerate(self._columns):
                lower = column.lower()
                if lower.startswith("time") or lower == "t" or lower.startswith("t_["):
                    self.time_combo.setCurrentIndex(row)
                    break
        form.addRow("Signal", self.signal_combo)
        form.addRow("Time", self.time_combo)

        self.slope_spin = QtWidgets.QDoubleSpinBox()
        self.slope_spin.setRange(0.1, 50.0)
        self.slope_spin.setDecimals(3)
        self.slope_spin.setValue(4.0)
        form.addRow("S-N slope m", self.slope_spin)

        self.frequency_spin = QtWidgets.QDoubleSpinBox()
        self.frequency_spin.setRange(1e-9, 1e9)
        self.frequency_spin.setDecimals(6)
        self.frequency_spin.setValue(1.0)
        self.frequency_spin.setSuffix(" Hz")
        form.addRow("Equivalent frequency", self.frequency_spin)

        self.lifetime_spin = QtWidgets.QDoubleSpinBox()
        self.lifetime_spin.setRange(0.0, 1e12)
        self.lifetime_spin.setDecimals(3)
        self.lifetime_spin.setValue(20.0)
        self.lifetime_spin.setSuffix(" years")
        form.addRow("Lifetime", self.lifetime_spin)

        self.bins_spin = QtWidgets.QSpinBox()
        self.bins_spin.setRange(4, 4096)
        self.bins_spin.setValue(100)
        form.addRow("Rainflow bins", self.bins_spin)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        self.calculate_button = buttons.addButton(
            "Calculate", QtWidgets.QDialogButtonBox.AcceptRole
        )
        self.calculate_button.setObjectName("primaryButton")
        root.addWidget(buttons)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    def values(self):
        return {
            "signal": self.signal_combo.currentText(),
            "time": self.time_combo.currentText(),
            "m": self.slope_spin.value(),
            "frequency": self.frequency_spin.value(),
            "lifetime_years": self.lifetime_spin.value(),
            "bins": self.bins_spin.value(),
        }


class ExtremeLoadDialog(QtWidgets.QDialog):
    def __init__(self, columns, selected_signal=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ULS / Extreme-load comparison")
        self.setMinimumWidth(390)
        self._columns = [str(column) for column in columns]

        root = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        root.addLayout(form)

        self.signal_combo = QtWidgets.QComboBox()
        for column in self._columns:
            self.signal_combo.addItem(column)
        if selected_signal in self._columns:
            self.signal_combo.setCurrentIndex(self._columns.index(selected_signal))
        else:
            for row, column in enumerate(self._columns):
                if not column.lower().startswith("index"):
                    self.signal_combo.setCurrentIndex(row)
                    break
        form.addRow("Signal", self.signal_combo)

        self.top_n_spin = QtWidgets.QSpinBox()
        self.top_n_spin.setRange(1, 1000000)
        self.top_n_spin.setValue(10)
        form.addRow("Top N", self.top_n_spin)

        self.safety_factor_spin = QtWidgets.QDoubleSpinBox()
        self.safety_factor_spin.setRange(0.0, 1000.0)
        self.safety_factor_spin.setDecimals(4)
        self.safety_factor_spin.setValue(1.35)
        form.addRow("Safety factor", self.safety_factor_spin)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        self.compare_button = buttons.addButton(
            "Compare", QtWidgets.QDialogButtonBox.AcceptRole
        )
        self.compare_button.setObjectName("primaryButton")
        root.addWidget(buttons)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    def values(self):
        return {
            "signal": self.signal_combo.currentText(),
            "top_n": self.top_n_spin.value(),
            "safety_factor": self.safety_factor_spin.value(),
        }


class AnalysisResultsDialog(QtWidgets.QDialog):
    def __init__(self, title, tables, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(860, 520)
        self._models = []

        root = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs, 1)

        for label, dataframe in tables:
            view = QtWidgets.QTableView()
            model = DataFrameModel(dataframe, max_rows=max(200, len(dataframe)))
            self._models.append(model)
            view.setModel(model)
            view.setAlternatingRowColors(True)
            view.setSortingEnabled(False)
            view.resizeColumnsToContents()
            self.tabs.addTab(view, label)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.copy_button = buttons.addButton(
            "Copy current tab", QtWidgets.QDialogButtonBox.ActionRole
        )
        root.addWidget(buttons)
        buttons.rejected.connect(self.close)
        self.copy_button.clicked.connect(self.copy_current_tab)

    def copy_current_tab(self):
        index = self.tabs.currentIndex()
        if index < 0 or index >= len(self._models):
            return
        dataframe = self._models[index].dataframe
        if dataframe is None:
            return
        QtWidgets.QApplication.clipboard().setText(dataframe.to_csv(index=False))


class OrderTrackingDialog(QtWidgets.QDialog):
    def __init__(self, options=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Order tracking overlays")
        self.setMinimumWidth(390)
        options = dict(options or {})

        root = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        root.addLayout(form)

        self.rotor_speed_spin = QtWidgets.QDoubleSpinBox()
        self.rotor_speed_spin.setRange(1e-9, 1e9)
        self.rotor_speed_spin.setDecimals(6)
        self.rotor_speed_spin.setValue(float(options.get("rotor_speed", 12.1)))
        form.addRow("Rated rotor speed", self.rotor_speed_spin)

        self.speed_unit_combo = QtWidgets.QComboBox()
        self.speed_unit_combo.addItem("rpm", "rpm")
        self.speed_unit_combo.addItem("Hz", "hz")
        self.speed_unit_combo.addItem("rad/s", "rad/s")
        unit = str(options.get("speed_unit", "rpm"))
        unit_index = self.speed_unit_combo.findData(unit)
        self.speed_unit_combo.setCurrentIndex(max(0, unit_index))
        form.addRow("Speed unit", self.speed_unit_combo)

        self.orders_edit = QtWidgets.QLineEdit()
        self.orders_edit.setText(str(options.get("orders", "1, 3, 6")))
        form.addRow("Orders", self.orders_edit)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        root.addWidget(buttons)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    @staticmethod
    def _parse_orders(text):
        values = []
        for part in text.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            value = float(part)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("Orders must be positive finite numbers")
            if value not in values:
                values.append(value)
        if not values:
            raise ValueError("Enter at least one order")
        return values

    def accept(self):
        try:
            self._parse_orders(self.orders_edit.text())
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Order tracking overlays", str(exc))
            self.orders_edit.setFocus()
            return
        super().accept()

    def values(self):
        orders = self._parse_orders(self.orders_edit.text())
        return {
            "rotor_speed": float(self.rotor_speed_spin.value()),
            "speed_unit": self.speed_unit_combo.currentData(),
            "orders": ", ".join("{:g}".format(order) for order in orders),
        }


class StandardizeUnitsDialog(QtWidgets.QDialog):
    def __init__(self, initial_flavor="WE", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Standardize units")
        self.setMinimumWidth(390)

        root = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.addItem("Wind Energy / OpenFAST", "WE")
        self.target_combo.addItem("SI", "SI")
        target_index = self.target_combo.findData(initial_flavor)
        self.target_combo.setCurrentIndex(max(0, target_index))
        form.addRow("Target units", self.target_combo)
        root.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Cancel)
        self.apply_button = buttons.addButton(
            "Apply", QtWidgets.QDialogButtonBox.AcceptRole
        )
        self.apply_button.setObjectName("primaryButton")
        root.addWidget(buttons)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    def target_flavor(self):
        return self.target_combo.currentData()


class ScanDialog(QtWidgets.QDialog):
    def __init__(self, file_formats, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("Scan folder")
        self.resize(620, 560)
        self.file_formats = list(file_formats)
        self.settings = settings or QtCore.QSettings("NREL", "pyDatView")
        self.check_states = {}
        saved_formats = self.settings.value("scan/formats", [])
        if isinstance(saved_formats, str):
            saved_formats = [saved_formats]
        saved_formats = set(saved_formats or [])
        for i_fmt, fmt in enumerate(self.file_formats):
            if fmt.name in saved_formats:
                self.check_states[i_fmt] = QtCore.Qt.Checked

        root = QtWidgets.QVBoxLayout(self)

        folder_row = QtWidgets.QHBoxLayout()
        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_edit.setPlaceholderText("Folder containing simulation files")
        self.folder_edit.setText(str(self.settings.value("scan/folder", "") or ""))
        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self.browse_folder)
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(browse_button)
        root.addLayout(folder_row)

        self.recursive_check = QtWidgets.QCheckBox("Include subfolders")
        self.recursive_check.setChecked(self.settings.value("scan/recursive", True, type=bool))
        root.addWidget(self.recursive_check)

        self.keep_existing_check = QtWidgets.QCheckBox("Keep files from previous scans")
        self.keep_existing_check.setChecked(
            self.settings.value("scan/keep_existing", False, type=bool)
        )
        self.keep_existing_check.setToolTip(
            "Append new matches to the current scan index without unloading or removing existing files"
        )
        root.addWidget(self.keep_existing_check)

        bladed_row = QtWidgets.QHBoxLayout()
        bladed_row.addWidget(QtWidgets.QLabel("Bladed suffixes"))
        self.bladed_suffix_edit = QtWidgets.QLineEdit()
        self.bladed_suffix_edit.setPlaceholderText("Blank = .$PJ only; or 04, 05, 298")
        self.bladed_suffix_edit.setText(str(self.settings.value("scan/bladed_suffixes", "") or ""))
        self.bladed_suffix_edit.setToolTip(
            "Leave blank to scan only Bladed .$PJ projects. Enter suffixes such as 04 or 298 "
            "to scan those .$ output files instead."
        )
        bladed_row.addWidget(self.bladed_suffix_edit, 1)
        root.addLayout(bladed_row)

        filter_row = QtWidgets.QHBoxLayout()
        self.format_filter = QtWidgets.QLineEdit()
        self.format_filter.setPlaceholderText("Filter file types")
        self.select_all_button = QtWidgets.QPushButton("All")
        self.clear_button = QtWidgets.QPushButton("None")
        filter_row.addWidget(self.format_filter, 1)
        filter_row.addWidget(self.select_all_button)
        filter_row.addWidget(self.clear_button)
        root.addLayout(filter_row)

        self.format_list = QtWidgets.QListWidget()
        self.format_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        root.addWidget(self.format_list, 1)

        self.summary_label = QtWidgets.QLabel("Select one or more file types to scan.")
        root.addWidget(self.summary_label)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.format_filter.textChanged.connect(self.populate_formats)
        self.select_all_button.clicked.connect(lambda: self.set_visible_checked(True))
        self.clear_button.clicked.connect(lambda: self.set_visible_checked(False))
        self.populate_formats()
        geometry = self.settings.value("scan/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    def browse_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)

    def populate_formats(self):
        self.remember_checks()

        self.format_list.clear()
        text_filter = self.format_filter.text().strip().lower()
        for i_fmt, fmt in enumerate(self.file_formats):
            extensions = ", ".join(getattr(fmt, "extensions", []))
            label = "{}  ({})".format(fmt.name, extensions)
            if text_filter and text_filter not in label.lower():
                continue
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(self.check_states.get(i_fmt, QtCore.Qt.Unchecked))
            item.setData(QtCore.Qt.UserRole, i_fmt)
            item.setData(QtCore.Qt.UserRole + 1, _format_specs(fmt))
            self.format_list.addItem(item)

    def set_visible_checked(self, checked):
        state = QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked
        for row in range(self.format_list.count()):
            item = self.format_list.item(row)
            item.setCheckState(state)
            self.check_states[item.data(QtCore.Qt.UserRole)] = state

    def remember_checks(self):
        for row in range(self.format_list.count()):
            item = self.format_list.item(row)
            self.check_states[item.data(QtCore.Qt.UserRole)] = item.checkState()

    def selected_specs(self):
        self.remember_checks()
        specs = []
        for i_fmt, state in self.check_states.items():
            if state == QtCore.Qt.Checked:
                specs.extend(_format_specs(self.file_formats[i_fmt]))
        return specs

    def selected_format_entries(self):
        self.remember_checks()
        entries = []
        for i_fmt, state in self.check_states.items():
            if state == QtCore.Qt.Checked:
                fmt = self.file_formats[i_fmt]
                entries.append((fmt, _format_specs(fmt)))
        return entries

    def selected_folder(self):
        return self.folder_edit.text().strip()

    def recursive(self):
        return self.recursive_check.isChecked()

    def bladed_suffixes(self):
        return _parse_bladed_suffixes(self.bladed_suffix_edit.text())

    def keep_existing(self):
        return self.keep_existing_check.isChecked()

    def accept(self):
        if not os.path.isdir(self.selected_folder()):
            QtWidgets.QMessageBox.warning(self, "Scan folder", "Select a valid folder.")
            return
        if not self.selected_specs():
            QtWidgets.QMessageBox.warning(self, "Scan folder", "Select at least one file type.")
            return
        selected_formats = [
            self.file_formats[i_fmt].name
            for i_fmt, state in self.check_states.items()
            if state == QtCore.Qt.Checked
        ]
        self.settings.setValue("scan/folder", self.selected_folder())
        self.settings.setValue("scan/recursive", self.recursive())
        self.settings.setValue("scan/keep_existing", self.keep_existing())
        self.settings.setValue("scan/bladed_suffixes", self.bladed_suffix_edit.text().strip())
        self.settings.setValue("scan/formats", selected_formats)
        self.settings.setValue("scan/geometry", self.saveGeometry())
        self.settings.sync()
        super().accept()
