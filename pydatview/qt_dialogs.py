"""Dialogs and table models used by the Qt main window."""

import os

import numpy as np

from pydatview.qt_compat import QtCore, QtWidgets
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
