"""PySide6/PyQtGraph pyDatView application.

The main module owns window composition and workflow orchestration. Reusable
Qt infrastructure, dialogs, plotting, I/O, and numerical helpers live in
focused sibling modules and are re-exported here for compatibility.
"""

import os
import sys
import time
import traceback
from collections import deque

import numpy as np

from pydatview.Tables import TableList
from pydatview.plotdata import PlotData
import pydatview.io as weio
from pydatview.qt_compat import QtCore, QtGui, QtWidgets, pg
from pydatview.qt_dialogs import (
    AxisLimitsDialog,
    CalculationDialog,
    DataFrameModel,
    ScanDialog,
    StandardizeUnitsDialog,
)
from pydatview.qt_io import (
    LazyFileEntry,
    LazyLoadWorker,
    SelectorPane,
    _default_lazy_workers,
    _format_columns,
    _format_specs,
    _indexed_format_entries,
    _match_indexed_format,
    _matches_bladed_suffix,
    _matches_specs,
    _parse_bladed_suffixes,
    _read_fast_ascii_columns,
    _read_fast_binary_columns,
    _resource_path,
    read_lazy_columns,
    scan_readable_file_matches,
    scan_readable_files,
)
from pydatview.qt_math import (
    _MATH_AST_NODES,
    _MATH_CONSTANTS,
    _MATH_FUNCTIONS,
    _TABLE_TRANSFORMS,
    _column_array,
    _resolve_expression_column,
    evaluate_math_expression,
    evaluate_table_script,
    transform_file_tables,
    trim_rows,
)
from pydatview.qt_plot import (
    NumericAxisItem,
    QtPlotCanvas,
    _PLOT_PALETTE,
    _curve_color,
    _curve_pen,
    _selected_curve_pen,
)
from pydatview.qt_stats import (
    _COMPARISON_METHODS,
    _DEFAULT_STATS_COLUMNS,
    _STATS_COLUMNS,
    _as_float_array,
    _comparison_axis_label,
    _comparison_error,
    _comparison_source,
    _equivalent_loads,
    box_plot_data,
    _finite_xy,
    _plot_ready_xy,
    _sample_spacing,
    _series_statistics,
    _trapezoidal_integral,
    compare_plot_data,
    swap_plot_axes,
)
from pydatview.qt_loading import QtLoadingMixin
from pydatview.qt_selection import QtSelectionPlotMixin
from pydatview.qt_tools import QtToolsStatsMixin
from pydatview.qt_theme import configure_application, windows_stylesheet

class MainWindow(
        QtLoadingMixin,
        QtSelectionPlotMixin,
        QtToolsStatsMixin,
        QtWidgets.QMainWindow):
    def __init__(self, filenames=None, dataframes=None, names=None):
        configure_application(QtWidgets.QApplication.instance())
        super().__init__()
        self.setWindowTitle("pyDatView Qt")
        ui_font = QtGui.QFont(self.font())
        self._ui_font_size = max(7, ui_font.pointSize() - 1)
        ui_font.setPointSize(self._ui_font_size)
        self.setFont(ui_font)
        self.resize(1280, 820)
        self.settings = QtCore.QSettings("NREL", "pyDatView")
        self.tab_list = TableList()
        self.file_formats, self.file_format_errors = self._load_file_formats()
        self.plot_data = []
        self.current_files = []
        self.lazy_entries = []
        self.lazy_load_queue = deque()
        self.lazy_loader_threads = {}
        self.lazy_loader_workers = {}
        self.lazy_memory_reservations = {}
        self._directory_file_sizes = {}
        self.lazy_generation = 0
        self.lazy_max_workers = _default_lazy_workers()
        self.bladed_worker_cap = 2 if sys.platform.startswith("win") else 4
        self.lazy_warning_backlog = []
        self.lazy_item_widgets = {}
        self.lazy_loaded_total = 0
        self.lazy_selected_batch = set()
        self.lazy_selection_refresh_pending = False
        self.lazy_last_ui_update = 0.0
        self.plot_after_lazy_load = False
        self.selector_panes = []
        self.lazy_batch_total = 0
        self.lazy_batch_done = 0
        self.unit_flavor = ""
        self.active_selector_pane = None
        self.axis_limits = {key: None for key in ("xmin", "xmax", "ymin", "ymax")}
        self._previous_plot_type = "Regular"
        self._regular_logy = False
        self.redraw_timer = QtCore.QTimer(self)
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.setInterval(40)
        self.redraw_timer.timeout.connect(self.redraw)

        self._build_ui()
        self._connect()
        self._show_file_format_errors()

        if dataframes is not None:
            self.load_dfs(dataframes, names=names)
        if filenames:
            self.load_files(filenames, add=False)

    def _load_file_formats(self):
        io_userpath = os.path.join(weio.defaultUserDataDir(), "pydatview_io")
        return weio.fileFormats(userpath=io_userpath, ignoreErrors=True, verbose=False)

    def _build_ui(self):
        self._build_actions()

        central = QtWidgets.QWidget()
        central.setObjectName("appBackground")
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        controls_panel = QtWidgets.QFrame()
        controls_panel.setObjectName("plotControls")
        top = QtWidgets.QGridLayout(controls_panel)
        top.setContentsMargins(10, 8, 10, 8)
        top.setHorizontalSpacing(8)
        top.setVerticalSpacing(7)
        root.addWidget(controls_panel)
        self.plot_type_combo = QtWidgets.QComboBox()
        self.plot_type_combo.addItems([
            "Regular",
            "FFT",
            "Cumulative PSD",
            "PDF",
            "MinMax",
            "Box Plot",
            "Compare",
        ])
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Overlay", "Subplots"])
        self.compare_combo = QtWidgets.QComboBox()
        self.compare_combo.addItems(["Auto", "2", "3"])
        self.live_plot = QtWidgets.QCheckBox("Live plot")
        self.live_plot.setChecked(True)
        self.swap_xy_check = QtWidgets.QCheckBox("Swap X-Y")
        self.swap_xy_check.setChecked(False)
        self.grid_check = QtWidgets.QCheckBox("Grid")
        self.grid_check.setChecked(False)
        self.logx_check = QtWidgets.QCheckBox("Log x")
        self.logy_check = QtWidgets.QCheckBox("Log y")
        self.legend_check = QtWidgets.QCheckBox("Legend")
        self.legend_check.setChecked(False)
        self.measurement_marker_check = QtWidgets.QCheckBox("X marker")
        self.measurement_marker_check.setToolTip(
            "Click a plot or its X axis to show every curve value at that X position"
        )
        self.line_width_spin = QtWidgets.QDoubleSpinBox()
        self.line_width_spin.setRange(0.25, 8.0)
        self.line_width_spin.setSingleStep(0.25)
        self.line_width_spin.setValue(1.5)
        self.marker_combo = QtWidgets.QComboBox()
        self.marker_combo.addItems(["None", "Circle", "Square", "Triangle", "Diamond"])
        self.axis_limits_button = QtWidgets.QPushButton("Limits")
        self.axis_limits_button.setToolTip("Set X and Y plot limits")
        self.zoom_area_button = QtWidgets.QPushButton("Zoom area")
        self.zoom_area_button.setObjectName("zoomAreaButton")
        self.zoom_area_button.setCheckable(True)
        self.zoom_area_button.setFixedWidth(105)
        self.zoom_area_button.setToolTip(
            "Drag a rectangle over a plot; use Auto range to restore the full view"
        )
        self.load_workers_combo = QtWidgets.QComboBox()
        self.load_workers_combo.addItems(["Auto", "1", "2", "4", "8", "16", "32", "64", "96"])
        self.load_workers_combo.setToolTip(
            "Maximum parallel file workers. Bladed projects use a separate safety cap "
            "of {} worker(s) to limit native-decoder memory pressure.".format(
                self.bladed_worker_cap
            )
        )
        self.loading_progress = QtWidgets.QProgressBar()
        self.loading_progress.setRange(0, 1)
        self.loading_progress.setValue(0)
        self.loading_progress.setFormat("Loading %v/%m")
        self.loading_progress.setMaximumWidth(180)
        self.loading_progress.setVisible(False)
        self.status_label = QtWidgets.QLabel("No files loaded")
        self.status_label.setObjectName("statusChip")

        top.addWidget(QtWidgets.QLabel("Plot"), 0, 0)
        top.addWidget(self.plot_type_combo, 0, 1)
        top.addWidget(QtWidgets.QLabel("Layout"), 0, 2)
        top.addWidget(self.mode_combo, 0, 3)
        top.addWidget(QtWidgets.QLabel("Compare"), 0, 4)
        top.addWidget(self.compare_combo, 0, 5)
        top.addWidget(self.live_plot, 0, 6)
        top.addWidget(self.swap_xy_check, 0, 7)
        top.addWidget(self.measurement_marker_check, 0, 8)
        top.setColumnStretch(9, 1)
        top.addWidget(self.status_label, 0, 10, QtCore.Qt.AlignRight)

        top.addWidget(self.grid_check, 1, 0)
        top.addWidget(self.logx_check, 1, 1)
        top.addWidget(self.logy_check, 1, 2)
        top.addWidget(self.legend_check, 1, 3)
        top.addWidget(QtWidgets.QLabel("Line width"), 1, 4)
        top.addWidget(self.line_width_spin, 1, 5)
        top.addWidget(QtWidgets.QLabel("Marker"), 1, 6)
        top.addWidget(self.marker_combo, 1, 7)
        top.addWidget(self.axis_limits_button, 1, 8)
        top.addWidget(self.zoom_area_button, 1, 9, QtCore.Qt.AlignLeft)
        load_controls = QtWidgets.QHBoxLayout()
        load_controls.setContentsMargins(0, 0, 0, 0)
        load_controls.setSpacing(6)
        load_controls.addWidget(QtWidgets.QLabel("Workers"))
        load_controls.addWidget(self.load_workers_combo)
        load_controls.addWidget(self.loading_progress)
        top.addLayout(load_controls, 1, 10)

        self.fft_options_panel = QtWidgets.QFrame()
        self.fft_options_panel.setObjectName("plotControls")
        fft_layout = QtWidgets.QGridLayout(self.fft_options_panel)
        fft_layout.setContentsMargins(10, 6, 10, 6)
        fft_layout.setHorizontalSpacing(8)
        fft_layout.setVerticalSpacing(5)
        self.fft_output_combo = QtWidgets.QComboBox()
        self.fft_output_combo.addItems(["PSD", "f x PSD", "Amplitude"])
        self.fft_averaging_combo = QtWidgets.QComboBox()
        self.fft_averaging_combo.addItems(["None", "Welch", "Binning"])
        self.fft_averaging_combo.setCurrentText("Welch")
        self.fft_window_combo = QtWidgets.QComboBox()
        self.fft_window_combo.addItems(["Hamming", "Hann", "Rectangular"])
        self.fft_x_combo = QtWidgets.QComboBox()
        self.fft_x_combo.addItem("Frequency [1/x]", "1/x")
        self.fft_x_combo.addItem("Cyclic frequency [2pi/x]", "2pi/x")
        self.fft_x_combo.addItem("Period [x]", "x")
        self.fft_detrend_check = QtWidgets.QCheckBox("Detrend")
        self.fft_detrend_check.setChecked(False)
        self.fft_nexp_spin = QtWidgets.QSpinBox()
        self.fft_nexp_spin.setRange(3, 30)
        self.fft_nexp_spin.setValue(11)
        self.fft_nexp_spin.setToolTip("Welch segment length as a power of two")
        self.fft_window_length_label = QtWidgets.QLabel("2048 samples")
        self.fft_bins_spin = QtWidgets.QSpinBox()
        self.fft_bins_spin.setRange(3, 200)
        self.fft_bins_spin.setValue(20)
        self.fft_bins_spin.setToolTip("Number of logarithmic frequency bins per decade")
        self.fft_output_combo.setCurrentText(
            str(self.settings.value("fft/output", "PSD"))
        )
        self.fft_averaging_combo.setCurrentText(
            str(self.settings.value("fft/averaging", "Welch"))
        )
        self.fft_window_combo.setCurrentText(
            str(self.settings.value("fft/window", "Hamming"))
        )
        saved_x_type = str(self.settings.value("fft/x_type", "1/x"))
        saved_x_index = self.fft_x_combo.findData(saved_x_type)
        self.fft_x_combo.setCurrentIndex(max(0, saved_x_index))
        self.fft_detrend_check.setChecked(
            self.settings.value("fft/detrend", False, type=bool)
        )
        self.fft_nexp_spin.setValue(
            self.settings.value("fft/n_exp", 11, type=int)
        )
        self.fft_bins_spin.setValue(
            self.settings.value("fft/bins_per_decade", 20, type=int)
        )

        fft_layout.addWidget(QtWidgets.QLabel("Spectrum"), 0, 0)
        fft_layout.addWidget(self.fft_output_combo, 0, 1)
        fft_layout.addWidget(QtWidgets.QLabel("Averaging"), 0, 2)
        fft_layout.addWidget(self.fft_averaging_combo, 0, 3)
        fft_layout.addWidget(QtWidgets.QLabel("Window"), 0, 4)
        fft_layout.addWidget(self.fft_window_combo, 0, 5)
        fft_layout.addWidget(QtWidgets.QLabel("X axis"), 0, 6)
        fft_layout.addWidget(self.fft_x_combo, 0, 7)
        fft_layout.addWidget(self.fft_detrend_check, 0, 8)
        fft_layout.addWidget(QtWidgets.QLabel("Welch 2^n"), 1, 0)
        fft_layout.addWidget(self.fft_nexp_spin, 1, 1)
        fft_layout.addWidget(self.fft_window_length_label, 1, 2, 1, 2)
        fft_layout.addWidget(QtWidgets.QLabel("Bins/decade"), 1, 4)
        fft_layout.addWidget(self.fft_bins_spin, 1, 5)
        fft_layout.setColumnStretch(9, 1)
        self.fft_options_panel.setVisible(False)
        root.addWidget(self.fft_options_panel)

        self.comparison_options_panel = QtWidgets.QFrame()
        self.comparison_options_panel.setObjectName("plotControls")
        comparison_layout = QtWidgets.QHBoxLayout(self.comparison_options_panel)
        comparison_layout.setContentsMargins(10, 6, 10, 6)
        comparison_layout.setSpacing(8)
        comparison_layout.addWidget(QtWidgets.QLabel("Comparison type"))
        self.comparison_method_combo = QtWidgets.QComboBox()
        self.comparison_method_combo.addItems(list(_COMPARISON_METHODS))
        saved_comparison_method = str(
            self.settings.value("compare/method", "Relative")
        )
        if saved_comparison_method in _COMPARISON_METHODS:
            self.comparison_method_combo.setCurrentText(saved_comparison_method)
        self.comparison_method_combo.setToolTip(
            "The first selected series in each group is the reference"
        )
        comparison_layout.addWidget(self.comparison_method_combo)
        comparison_layout.addStretch(1)
        self.comparison_options_panel.setVisible(False)
        root.addWidget(self.comparison_options_panel)

        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        root.addWidget(self.main_splitter, 1)

        side = QtWidgets.QWidget()
        side.setObjectName("selectorArea")
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)

        self.selector_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.selector_splitter.setChildrenCollapsible(False)
        side_layout.addWidget(self.selector_splitter, 1)
        self.set_compare_pane_count(1)

        button_row = QtWidgets.QGridLayout()
        button_row.setHorizontalSpacing(6)
        button_row.setVerticalSpacing(6)
        self.plot_button = QtWidgets.QPushButton("Plot")
        self.plot_button.setObjectName("primaryButton")
        self.plot_button.setIcon(QtGui.QIcon(_resource_path("icons", "chart.svg")))
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.select_all_y_button = QtWidgets.QPushButton("All Y")
        self.select_none_y_button = QtWidgets.QPushButton("None")
        self.load_selected_button = QtWidgets.QPushButton("Load full selected")
        self.load_selected_button.setToolTip(
            "Load every variable from the selected indexed files"
        )
        self.math_button = QtWidgets.QPushButton("Calculate")
        self.math_button.setToolTip("Create and plot a variable from a mathematical expression")
        button_row.addWidget(self.plot_button, 0, 0)
        button_row.addWidget(self.clear_button, 0, 1)
        button_row.addWidget(self.select_all_y_button, 0, 2)
        button_row.addWidget(self.select_none_y_button, 0, 3)
        button_row.addWidget(self.load_selected_button, 1, 0, 1, 2)
        button_row.addWidget(self.math_button, 1, 2, 1, 2)
        side_layout.addLayout(button_row)

        self.canvas = QtPlotCanvas()
        self.detail_tabs = QtWidgets.QTabWidget()
        self.table_model = DataFrameModel()
        self.table_view = QtWidgets.QTableView()
        self.table_view.setModel(self.table_model)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSortingEnabled(False)
        self.table_view.horizontalHeader().setStretchLastSection(False)
        self.info_text = QtWidgets.QPlainTextEdit()
        self.info_text.setReadOnly(True)
        self.stats_panel = QtWidgets.QWidget()
        stats_layout = QtWidgets.QVBoxLayout(self.stats_panel)
        stats_layout.setContentsMargins(6, 6, 6, 6)
        stats_layout.setSpacing(5)
        stats_controls = QtWidgets.QHBoxLayout()
        stats_controls.addWidget(QtWidgets.QLabel("Columns"))
        self.stats_columns_button = QtWidgets.QToolButton()
        self.stats_columns_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.stats_columns_menu = QtWidgets.QMenu(self.stats_columns_button)
        self.stats_column_actions = {}
        valid_stats_columns = {key for key, _label, _numeric in _STATS_COLUMNS}
        saved_stats_columns = self.settings.value(
            "stats/columns", list(_DEFAULT_STATS_COLUMNS)
        )
        if isinstance(saved_stats_columns, str):
            saved_stats_columns = [saved_stats_columns]
        elif not isinstance(saved_stats_columns, (list, tuple, set)):
            saved_stats_columns = [saved_stats_columns]
        selected_stats_columns = {
            str(key) for key in saved_stats_columns
            if str(key) in valid_stats_columns
        }
        if not selected_stats_columns:
            selected_stats_columns = set(_DEFAULT_STATS_COLUMNS)
        for key, label, _numeric in _STATS_COLUMNS:
            action = self.stats_columns_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(key in selected_stats_columns)
            action.toggled.connect(self.on_stats_columns_changed)
            self.stats_column_actions[key] = action
        self.stats_columns_menu.addSeparator()
        select_all_stats_action = self.stats_columns_menu.addAction("Select all")
        select_all_stats_action.triggered.connect(self.select_all_stats_columns)
        reset_stats_action = self.stats_columns_menu.addAction("Restore defaults")
        reset_stats_action.triggered.connect(self.reset_stats_columns)
        self.stats_columns_button.setMenu(self.stats_columns_menu)
        stats_controls.addWidget(self.stats_columns_button)
        stats_controls.addSpacing(12)
        stats_controls.addWidget(QtWidgets.QLabel("DEL slopes"))
        self.del_slopes_button = QtWidgets.QToolButton()
        self.del_slopes_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.del_slopes_menu = QtWidgets.QMenu(self.del_slopes_button)
        self.del_slope_actions = {}
        saved_slopes = self.settings.value("stats/del_slopes", [4])
        if isinstance(saved_slopes, str):
            saved_slopes = [saved_slopes]
        try:
            saved_slopes = {int(value) for value in saved_slopes}
        except (TypeError, ValueError):
            saved_slopes = {4}
        for slope in range(2, 14):
            action = self.del_slopes_menu.addAction("m = {}".format(slope))
            action.setCheckable(True)
            action.setChecked(slope in saved_slopes)
            action.toggled.connect(self.on_del_slopes_changed)
            self.del_slope_actions[slope] = action
        self.del_slopes_button.setMenu(self.del_slopes_menu)
        stats_controls.addWidget(self.del_slopes_button)
        stats_controls.addStretch(1)
        self.copy_stats_button = QtWidgets.QPushButton("Copy")
        self.copy_stats_button.setToolTip(
            "Copy selected statistics rows, or all rows when none are selected"
        )
        self.export_stats_button = QtWidgets.QPushButton("Export CSV")
        self.export_stats_button.setToolTip("Export all visible statistics")
        stats_controls.addWidget(self.copy_stats_button)
        stats_controls.addWidget(self.export_stats_button)
        stats_layout.addLayout(stats_controls)
        self.stats_table = QtWidgets.QTableWidget()
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.stats_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.stats_table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.stats_table.verticalHeader().setVisible(False)
        stats_layout.addWidget(self.stats_table, 1)
        self.update_stats_columns_button()
        self.update_del_slopes_button()
        self.stats_copy_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence.Copy, self.stats_table
        )
        self.stats_copy_shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.detail_tabs.addTab(self.table_view, "Data")
        self.detail_tabs.addTab(self.stats_panel, "Stats")
        self.detail_tabs.addTab(self.info_text, "File info")

        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_splitter.addWidget(self.canvas)
        right_splitter.addWidget(self.detail_tabs)
        right_splitter.setStretchFactor(0, 4)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([620, 180])

        self.main_splitter.addWidget(side)
        self.main_splitter.addWidget(right_splitter)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([340, 940])

        self.setStatusBar(QtWidgets.QStatusBar())
        self.coordinate_label = QtWidgets.QLabel("X: --   Y: --")
        self.coordinate_label.setObjectName("coordinateReadout")
        self.coordinate_label.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter
        )
        self.coordinate_label.setFixedWidth(300)
        coordinate_font = QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.FixedFont
        )
        coordinate_font.setPointSize(self._ui_font_size)
        self.coordinate_label.setFont(coordinate_font)
        self.statusBar().addPermanentWidget(self.coordinate_label)
        self._apply_light_borders()

    def create_selector_pane(self, index):
        frame = QtWidgets.QGroupBox("Set {}".format(index + 1))
        frame.setProperty("selectorPane", True)
        frame.setMinimumWidth(230)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(6)
        tables_label = QtWidgets.QLabel("TABLES")
        tables_label.setProperty("sectionLabel", True)
        layout.addWidget(tables_label)
        table_list_widget = QtWidgets.QListWidget()
        table_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        table_list_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        compact_font = QtGui.QFont(self.font())
        compact_font.setPointSize(max(7, self._ui_font_size - 1))
        table_list_widget.setFont(compact_font)
        layout.addWidget(table_list_widget, 2)
        bladed_dataset_label = QtWidgets.QLabel("BLADED VARIABLE GROUP")
        bladed_dataset_label.setProperty("sectionLabel", True)
        bladed_dataset_label.setVisible(False)
        layout.addWidget(bladed_dataset_label)
        bladed_dataset_combo = QtWidgets.QComboBox()
        bladed_dataset_combo.setToolTip("Variable group loaded from the selected Bladed .$PJ project")
        bladed_dataset_combo.setVisible(False)
        bladed_dataset_combo.setFont(compact_font)
        layout.addWidget(bladed_dataset_combo)
        x_label = QtWidgets.QLabel("X COLUMN")
        x_label.setProperty("sectionLabel", True)
        layout.addWidget(x_label)
        column_filter = QtWidgets.QLineEdit()
        column_filter.setPlaceholderText("Filter Y columns")
        column_filter.setClearButtonEnabled(True)
        layout.addWidget(column_filter)
        x_combo = QtWidgets.QComboBox()
        x_combo.setFont(compact_font)
        layout.addWidget(x_combo)
        y_label = QtWidgets.QLabel("Y COLUMNS")
        y_label.setProperty("sectionLabel", True)
        layout.addWidget(y_label)
        y_list_widget = QtWidgets.QListWidget()
        y_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        y_list_widget.setFont(compact_font)
        layout.addWidget(y_list_widget, 3)

        pane = SelectorPane(
            frame,
            table_list_widget,
            bladed_dataset_label,
            bladed_dataset_combo,
            column_filter,
            x_combo,
            y_list_widget,
        )
        table_list_widget.itemSelectionChanged.connect(
            lambda p=pane: self.on_table_selection_changed(p)
        )
        table_list_widget.customContextMenuRequested.connect(
            lambda position, p=pane: self.show_table_context_menu(p, position)
        )
        bladed_dataset_combo.currentIndexChanged.connect(
            lambda _index, p=pane: self.on_bladed_dataset_changed(p)
        )
        x_combo.currentIndexChanged.connect(
            lambda _index, p=pane: self.on_pane_selection_changed(p)
        )
        y_list_widget.itemSelectionChanged.connect(
            lambda p=pane: self.on_pane_selection_changed(p)
        )
        column_filter.textChanged.connect(lambda _text, p=pane: self.populate_columns(p))
        self.selector_splitter.addWidget(frame)
        self.selector_panes.append(pane)
        if self.active_selector_pane is None:
            self.active_selector_pane = pane
        if index == 0:
            self.table_list_widget = table_list_widget
            self.column_filter = column_filter
            self.x_combo = x_combo
            self.y_list_widget = y_list_widget
        return pane

    def compare_pane_count(self):
        text = self.compare_combo.currentText()
        if text == "2":
            return 2
        if text == "3":
            return 3
        return 1

    def set_compare_pane_count(self, count):
        while len(self.selector_panes) < count:
            self.create_selector_pane(len(self.selector_panes))
        for i, pane in enumerate(self.selector_panes):
            pane.frame.setTitle("Set {}".format(i + 1))
            pane.frame.setVisible(i < count)
        if self.selector_panes:
            self.selector_splitter.setSizes([290] * count)

    def resize_compare_region(self):
        count = self.compare_pane_count()
        sizes = self.main_splitter.sizes()
        available = sum(sizes) if sum(sizes) > 0 else max(self.width() - 16, 1000)
        minimum_plot_width = 420
        selector_width = max(340, count * 290)
        selector_width = min(selector_width, max(340, available - minimum_plot_width))
        self.main_splitter.setSizes([selector_width, max(minimum_plot_width, available - selector_width)])
        self.selector_splitter.setSizes([290] * count)

    def visible_selector_panes(self):
        return self.selector_panes[:self.compare_pane_count()]

    def _apply_light_borders(self):
        self.setStyleSheet("""
            QWidget {
                color: #17212b;
            }
            QMainWindow, QDialog, QWidget#appBackground {
                background: #dbe2e9;
            }
            QWidget#selectorArea {
                background: #dbe2e9;
            }
            QFrame#plotControls {
                background: #eef2f6;
                border: 1px solid #657585;
                border-radius: 6px;
            }
            QLabel[sectionLabel="true"] {
                color: #364656;
                background: #e1e7ed;
                border: 1px solid #c4ced8;
                border-radius: 3px;
                padding: 2px 4px;
                font-weight: 700;
            }
            QLabel#statusChip {
                background: #eaf2ff;
                color: #174ea6;
                border: 1px solid #9bbcf1;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QLabel#coordinateReadout {
                color: #17212b;
                background: #eef2f6;
                border-left: 1px solid #8794a2;
                padding: 2px 8px;
            }
            QMenuBar {
                background: #d6dee7;
                border-bottom: 1px solid #657585;
                spacing: 4px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 6px 10px;
                border: 1px solid transparent;
                border-radius: 3px;
            }
            QMenuBar::item:selected,
            QMenuBar::item:pressed {
                background: #e7f0ff;
                border: 1px solid #729bd3;
            }
            QMenu {
                background: #f7f9fb;
                border: 1px solid #657585;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 28px 6px 22px;
                border: 1px solid transparent;
            }
            QMenu::item:selected {
                background: #dbeafe;
                border: 1px solid #4f83cc;
            }
            QToolBar {
                background: #e1e7ed;
                border: 1px solid #657585;
                border-left: 0;
                border-right: 0;
                spacing: 5px;
                padding: 4px 7px;
            }
            QToolButton {
                background: #f7f9fb;
                border: 1px solid #758493;
                border-radius: 4px;
                padding: 5px;
            }
            QToolButton:hover {
                background: #eaf2ff;
                border-color: #3978c5;
            }
            QToolButton:pressed {
                background: #d8e8ff;
            }
            QSplitter::handle {
                background: #8796a5;
            }
            QSplitter::handle:hover {
                background: #4d89d6;
            }
            QGroupBox[selectorPane="true"] {
                background: #f7f9fb;
                border: 1px solid #657585;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 6px;
                font-weight: 600;
            }
            QGroupBox[selectorPane="true"]::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 9px;
                padding: 0 5px;
                color: #174ea6;
                background: #f7f9fb;
            }
            QListWidget, QTableView, QPlainTextEdit, QLineEdit, QComboBox, QDoubleSpinBox {
                background: #ffffff;
                border: 1px solid #758493;
                border-radius: 4px;
                alternate-background-color: #edf2f7;
                selection-background-color: #2563eb;
                selection-color: #ffffff;
            }
            QLineEdit, QComboBox, QDoubleSpinBox {
                min-height: 25px;
                padding: 1px 6px;
            }
            QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus,
            QListWidget:focus, QTableView:focus, QPlainTextEdit:focus {
                border: 2px solid #2f74c8;
            }
            QComboBox::drop-down {
                border: 0;
                width: 22px;
            }
            QListWidget::item {
                padding: 4px 6px;
                border: 1px solid transparent;
            }
            QListWidget::item:selected,
            QListWidget::item:selected:active,
            QListWidget::item:selected:!active {
                background: #2563eb;
                color: #ffffff;
                border: 1px solid #1d4ed8;
            }
            QListWidget::item:hover:!selected {
                background: #e5effa;
                border: 1px solid #9bbce8;
            }
            QTableView::item:selected,
            QTableView::item:selected:active,
            QTableView::item:selected:!active {
                background: #2563eb;
                color: #ffffff;
            }
            QPushButton {
                background: #e5ebf1;
                border: 1px solid #758493;
                border-radius: 4px;
                min-height: 25px;
                padding: 3px 10px;
            }
            QPushButton:hover {
                background: #eaf2ff;
                border-color: #3978c5;
            }
            QPushButton:pressed {
                background: #d8e8ff;
            }
            QPushButton#primaryButton {
                color: #ffffff;
                background: #1769c2;
                border-color: #0e559f;
                font-weight: 600;
            }
            QPushButton#primaryButton:hover {
                background: #0f5dad;
            }
            QPushButton#zoomAreaButton:checked {
                color: #ffffff;
                background: #1769c2;
                border-color: #0e559f;
                font-weight: 600;
            }
            QPushButton[limitsActive="true"] {
                color: #174ea6;
                background: #dbeafe;
                border: 2px solid #2f74c8;
                font-weight: 600;
            }
            QPushButton:disabled, QToolButton:disabled,
            QComboBox:disabled, QLineEdit:disabled, QDoubleSpinBox:disabled {
                color: #8793a0;
                background: #e8ebef;
                border-color: #b7c0c8;
            }
            QTabWidget::pane {
                border: 1px solid #657585;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #cfd8e2;
                border: 1px solid #758493;
                border-bottom: 0;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 14px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                border-bottom-color: #ffffff;
                color: #174ea6;
                font-weight: 600;
            }
            QHeaderView::section {
                background: #d6dee7;
                color: #263442;
                border: 0;
                border-right: 1px solid #aeb8c2;
                border-bottom: 1px solid #8794a2;
                padding: 5px;
                font-weight: 600;
            }
            QProgressBar {
                min-height: 19px;
                border: 1px solid #758493;
                border-radius: 4px;
                background: #f7f9fb;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #2f74c8;
                border-radius: 3px;
            }
            QScrollBar:vertical {
                background: #eef1f4;
                width: 12px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #9ba8b5;
                border-radius: 4px;
                min-height: 24px;
                margin: 2px;
            }
            QScrollBar:horizontal {
                background: #eef1f4;
                height: 12px;
                margin: 0;
            }
            QScrollBar::handle:horizontal {
                background: #9ba8b5;
                border-radius: 4px;
                min-width: 24px;
                margin: 2px;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                width: 0;
                height: 0;
            }
            QToolTip {
                color: #ffffff;
                background: #263442;
                border: 1px solid #101820;
                padding: 4px;
            }
            QStatusBar {
                background: #d6dee7;
                border-top: 1px solid #657585;
            }
        """ + windows_stylesheet())

    def _build_actions(self):
        file_menu = self.menuBar().addMenu("&File")
        self.open_action = file_menu.addAction("Open")
        self.add_action = file_menu.addAction("Add")
        self.reload_action = file_menu.addAction("Reload")
        self.scan_action = file_menu.addAction(QtGui.QIcon(_resource_path("icons", "scan.png")), "Scan folder")
        self.export_table_action = file_menu.addAction("Export selected table")
        self.export_plot_action = file_menu.addAction(
            QtGui.QIcon(_resource_path("icons", "filesave.svg")),
            "Export publication plot",
        )
        file_menu.addSeparator()
        quit_action = file_menu.addAction("Quit")
        self.open_action.triggered.connect(lambda: self.select_files(add=False))
        self.add_action.triggered.connect(lambda: self.select_files(add=True))
        self.reload_action.triggered.connect(self.reload_files)
        self.scan_action.triggered.connect(self.scan_folder)
        self.export_table_action.triggered.connect(self.export_selected_table)
        self.export_plot_action.triggered.connect(self.export_plot_image)
        quit_action.triggered.connect(self.close)

        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("main_toolbar")
        toolbar.setIconSize(QtCore.QSize(24, 24))
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.add_action)
        toolbar.addAction(self.reload_action)
        toolbar.addSeparator()
        toolbar.addAction(self.scan_action)
        toolbar.addAction(self.export_plot_action)

        view_menu = self.menuBar().addMenu("&View")
        self.autorange_action = view_menu.addAction("Auto range")
        self.autorange_action.triggered.connect(self.auto_range)
        self.zoom_area_action = view_menu.addAction("Zoom area")
        self.zoom_area_action.setCheckable(True)
        self.zoom_area_action.toggled.connect(self.on_zoom_area_toggled)
        self.axis_limits_action = view_menu.addAction("Axis limits")
        self.axis_limits_action.triggered.connect(self.open_axis_limits_dialog)
        view_menu.addSeparator()
        self.increase_font_action = view_menu.addAction("Increase font size")
        self.increase_font_action.setShortcuts([
            QtGui.QKeySequence("Ctrl++"),
            QtGui.QKeySequence("Ctrl+="),
        ])
        self.increase_font_action.triggered.connect(
            lambda: self.change_ui_font_size(1)
        )
        self.decrease_font_action = view_menu.addAction("Decrease font size")
        self.decrease_font_action.setShortcut(QtGui.QKeySequence("Ctrl+-"))
        self.decrease_font_action.triggered.connect(
            lambda: self.change_ui_font_size(-1)
        )
        view_menu.addSeparator()
        view_export_plot_action = view_menu.addAction(
            "Export publication plot"
        )
        view_export_plot_action.triggered.connect(self.export_plot_image)

        tools_menu = self.menuBar().addMenu("&Tools")
        self.standardize_units_action = tools_menu.addAction("Standardize units...")
        self.standardize_units_action.triggered.connect(
            self.open_standardize_units_dialog
        )
        tools_menu.addSeparator()
        self.math_action = tools_menu.addAction("Mathematical operation")
        self.math_action.triggered.connect(self.open_calculation_dialog)

    def _connect(self):
        self.plot_type_combo.currentIndexChanged.connect(self.on_plot_type_changed)
        self.mode_combo.currentIndexChanged.connect(self.on_selection_changed)
        self.compare_combo.currentIndexChanged.connect(self.on_compare_mode_changed)
        self.swap_xy_check.stateChanged.connect(self.on_swap_xy_changed)
        self.grid_check.stateChanged.connect(self.on_selection_changed)
        self.logx_check.stateChanged.connect(self.on_selection_changed)
        self.logy_check.stateChanged.connect(self.on_selection_changed)
        self.legend_check.stateChanged.connect(self.on_selection_changed)
        self.measurement_marker_check.toggled.connect(
            self.on_measurement_marker_toggled
        )
        self.line_width_spin.valueChanged.connect(self.on_selection_changed)
        self.marker_combo.currentIndexChanged.connect(self.on_selection_changed)
        self.axis_limits_button.clicked.connect(self.open_axis_limits_dialog)
        self.zoom_area_button.toggled.connect(self.zoom_area_action.setChecked)
        self.comparison_method_combo.currentIndexChanged.connect(
            self.on_comparison_options_changed
        )
        self.load_workers_combo.currentIndexChanged.connect(self.update_lazy_worker_limit)
        self.canvas.curveSelected.connect(self.on_curve_selected)
        self.canvas.hoverCoordinates.connect(self.on_plot_hover)
        self.plot_button.clicked.connect(self.redraw)
        self.clear_button.clicked.connect(self.clear)
        self.select_all_y_button.clicked.connect(self.select_all_y)
        self.select_none_y_button.clicked.connect(self.select_none_y)
        self.load_selected_button.clicked.connect(self.load_selected_lazy_files)
        self.math_button.clicked.connect(self.open_calculation_dialog)
        self.copy_stats_button.clicked.connect(self.copy_stats)
        self.export_stats_button.clicked.connect(self.export_stats_csv)
        self.stats_copy_shortcut.activated.connect(self.copy_stats)
        for combo in (
            self.fft_output_combo,
            self.fft_averaging_combo,
            self.fft_window_combo,
            self.fft_x_combo,
        ):
            combo.currentIndexChanged.connect(self.on_fft_options_changed)
        self.fft_detrend_check.stateChanged.connect(self.on_fft_options_changed)
        self.fft_nexp_spin.valueChanged.connect(self.on_fft_options_changed)
        self.fft_bins_spin.valueChanged.connect(self.on_fft_options_changed)

    def on_plot_type_changed(self):
        plot_type = self.plot_type_combo.currentText()
        is_fft = plot_type == "FFT"
        is_spectral = plot_type in ("FFT", "Cumulative PSD")
        if is_fft and self._previous_plot_type != "FFT":
            self._regular_logy = self.logy_check.isChecked()
            self.logy_check.blockSignals(True)
            self.logy_check.setChecked(True)
            self.logy_check.blockSignals(False)
        elif not is_fft and self._previous_plot_type == "FFT":
            self.logy_check.blockSignals(True)
            self.logy_check.setChecked(self._regular_logy)
            self.logy_check.blockSignals(False)
        self._previous_plot_type = plot_type
        self.fft_options_panel.setVisible(is_spectral)
        is_box_plot = plot_type == "Box Plot"
        self.swap_xy_check.setEnabled(not is_box_plot)
        self.logx_check.setEnabled(not is_box_plot)
        self.logy_check.setEnabled(not is_box_plot)
        self.comparison_options_panel.setVisible(plot_type == "Compare")
        self.update_fft_control_states()
        self.on_selection_changed()

    def on_comparison_options_changed(self, _value=None):
        self.settings.setValue(
            "compare/method", self.comparison_method_combo.currentText()
        )
        self.on_selection_changed()

    def on_swap_xy_changed(self, _value=None):
        self.on_selection_changed()

    def update_fft_control_states(self):
        averaging = self.fft_averaging_combo.currentText()
        cumulative = self.plot_type_combo.currentText() == "Cumulative PSD"
        self.fft_output_combo.setEnabled(not cumulative)
        self.fft_output_combo.setToolTip(
            "Cumulative PSD always integrates the power spectral density"
            if cumulative else "Select the FFT spectrum output"
        )
        self.fft_x_combo.setEnabled(not cumulative)
        self.fft_x_combo.setToolTip(
            "Cumulative PSD is plotted against frequency"
            if cumulative else "Select the spectral x-axis"
        )
        self.fft_window_combo.setEnabled(averaging == "Welch")
        self.fft_nexp_spin.setEnabled(averaging == "Welch")
        self.fft_bins_spin.setEnabled(averaging == "Binning")
        self.fft_window_length_label.setText(
            "{:,} samples".format(2 ** self.fft_nexp_spin.value())
        )

    def on_fft_options_changed(self, _value=None):
        self.update_fft_control_states()
        self.settings.setValue("fft/output", self.fft_output_combo.currentText())
        self.settings.setValue("fft/averaging", self.fft_averaging_combo.currentText())
        self.settings.setValue("fft/window", self.fft_window_combo.currentText())
        self.settings.setValue("fft/x_type", self.fft_x_combo.currentData())
        self.settings.setValue("fft/detrend", self.fft_detrend_check.isChecked())
        self.settings.setValue("fft/n_exp", self.fft_nexp_spin.value())
        self.settings.setValue("fft/bins_per_decade", self.fft_bins_spin.value())
        self.on_selection_changed()

    def on_compare_mode_changed(self):
        self.set_compare_pane_count(self.compare_pane_count())
        self.populate_tables()
        QtCore.QTimer.singleShot(0, self.resize_compare_region)
        self.on_selection_changed()

    def on_pane_selection_changed(self, pane):
        if pane in self.visible_selector_panes():
            self.active_selector_pane = pane
        self.on_selection_changed()

    def open_standardize_units_dialog(self):
        initial_flavor = str(self.settings.value("units/target", "WE"))
        dialog = StandardizeUnitsDialog(
            initial_flavor=initial_flavor,
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        flavor = dialog.target_flavor()
        try:
            if flavor == "SI":
                self.standardize_units_si()
            else:
                self.standardize_units_we()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Standardize units",
                "{}: {}".format(type(exc).__name__, exc),
            )
            return
        self.settings.setValue("units/target", flavor)

    def standardize_units_we(self):
        self.standardize_units("WE", "Wind Energy / OpenFAST")

    def standardize_units_si(self):
        self.standardize_units("SI", "SI")

def showApp(firstArg=None, dataframes=None, filenames=None, names=None):
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    if filenames is None:
        filenames = []
    if firstArg is not None:
        if isinstance(firstArg, list):
            if len(firstArg) > 0 and isinstance(firstArg[0], str):
                filenames = firstArg
            else:
                dataframes = firstArg
        elif isinstance(firstArg, str):
            filenames = [firstArg]
        else:
            dataframes = [firstArg]
    window = MainWindow(filenames=filenames, dataframes=dataframes, names=names)
    window.show()
    return app.exec()


def cmdline():
    filenames = sys.argv[1:] if len(sys.argv) > 1 else []
    return showApp(filenames=filenames)


if __name__ == "__main__":
    raise SystemExit(cmdline())
