"""Native Qt controls for selecting plottable xarray slices."""

from pydatview.qt_compat import QtCore, QtWidgets


class NetCDFSliceDialog(QtWidgets.QDialog):
    def __init__(self, netcdf_file, parent=None):
        super().__init__(parent)
        self.netcdf_file = netcdf_file
        self.setWindowTitle('NetCDF slice')
        self.resize(620, 480)

        root = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        root.addLayout(form)

        self.variable_combo = QtWidgets.QComboBox()
        self.variable_combo.addItems(sorted(netcdf_file.plottable_variables()))
        form.addRow('Variable', self.variable_combo)
        self.shape_label = QtWidgets.QLabel()
        form.addRow('Dimensions', self.shape_label)

        self.x_combo = QtWidgets.QComboBox()
        form.addRow('X dimension', self.x_combo)
        self.series_combo = QtWidgets.QComboBox()
        form.addRow('Series dimension', self.series_combo)

        range_row = QtWidgets.QHBoxLayout()
        self.range_enabled = QtWidgets.QCheckBox('Limit X range')
        self.range_start = QtWidgets.QComboBox()
        self.range_stop = QtWidgets.QComboBox()
        range_row.addWidget(self.range_enabled)
        range_row.addWidget(self.range_start, 1)
        range_row.addWidget(QtWidgets.QLabel('to'))
        range_row.addWidget(self.range_stop, 1)
        form.addRow('Range', range_row)

        root.addWidget(QtWidgets.QLabel('FIXED DIMENSIONS'))
        self.fixed_form = QtWidgets.QFormLayout()
        root.addLayout(self.fixed_form)
        root.addStretch(1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok
        )
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText('Add slice')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.fixed_combos = {}
        self.variable_combo.currentTextChanged.connect(self._variable_changed)
        self.x_combo.currentTextChanged.connect(self._dimensions_changed)
        self.series_combo.currentIndexChanged.connect(self._dimensions_changed)
        self.range_enabled.toggled.connect(self._range_state_changed)
        self._variable_changed(self.variable_combo.currentText())

    def _variable_changed(self, name):
        if not name:
            return
        info = self.netcdf_file.plottable_variables()[name]
        dimensions = list(info['dimensions'])
        self.shape_label.setText(', '.join(
            '{} ({})'.format(dim, size)
            for dim, size in zip(dimensions, info['shape'])
        ))
        self.x_combo.blockSignals(True)
        self.x_combo.clear()
        self.x_combo.addItems(dimensions)
        self.x_combo.blockSignals(False)
        self._dimensions_changed()

    def _clear_fixed(self):
        while self.fixed_form.rowCount():
            self.fixed_form.removeRow(0)
        self.fixed_combos = {}

    def _dimensions_changed(self, *_args):
        name = self.variable_combo.currentText()
        x_dimension = self.x_combo.currentText()
        if not name or not x_dimension:
            return
        dimensions = list(
            self.netcdf_file.plottable_variables()[name]['dimensions']
        )
        old_series = self.series_combo.currentData()
        self.series_combo.blockSignals(True)
        self.series_combo.clear()
        self.series_combo.addItem('None (single curve)', None)
        for dimension in dimensions:
            if dimension != x_dimension:
                self.series_combo.addItem(dimension, dimension)
        old_index = self.series_combo.findData(old_series)
        self.series_combo.setCurrentIndex(max(0, old_index))
        self.series_combo.blockSignals(False)

        labels = self.netcdf_file.dimension_labels(name, x_dimension)
        self.range_start.clear()
        self.range_stop.clear()
        for index, label in enumerate(labels):
            self.range_start.addItem(label, index)
            self.range_stop.addItem(label, index)
        if labels:
            self.range_stop.setCurrentIndex(len(labels) - 1)
        self._range_state_changed(self.range_enabled.isChecked())
        self._rebuild_fixed()

    def _rebuild_fixed(self):
        self._clear_fixed()
        name = self.variable_combo.currentText()
        retained = {self.x_combo.currentText(), self.series_combo.currentData()}
        for dimension in self.netcdf_file.plottable_variables()[name]['dimensions']:
            if dimension in retained:
                continue
            combo = QtWidgets.QComboBox()
            for index, label in enumerate(
                    self.netcdf_file.dimension_labels(name, dimension)):
                combo.addItem(label, index)
            self.fixed_combos[dimension] = combo
            self.fixed_form.addRow(dimension, combo)

    def _range_state_changed(self, enabled):
        self.range_start.setEnabled(enabled)
        self.range_stop.setEnabled(enabled)

    def selection(self):
        x_range = None
        if self.range_enabled.isChecked():
            x_range = (
                self.range_start.currentData(),
                self.range_stop.currentData(),
            )
        return {
            'variable_name': self.variable_combo.currentText(),
            'x_dimension': self.x_combo.currentText(),
            'series_dimension': self.series_combo.currentData(),
            'fixed_indices': {
                dimension: combo.currentData()
                for dimension, combo in self.fixed_combos.items()
            },
            'x_range': x_range,
        }

    def slice_name(self):
        selection = self.selection()
        labels = []
        for dimension, index in selection['fixed_indices'].items():
            value = self.netcdf_file.dimension_labels(
                selection['variable_name'], dimension,
            )[index]
            labels.append('{}={}'.format(dimension, value))
        suffix = ' [{}]'.format(', '.join(labels)) if labels else ''
        return '{}{} (slice)'.format(selection['variable_name'], suffix)

