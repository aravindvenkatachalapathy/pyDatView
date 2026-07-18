"""PySide6/PyQtGraph pyDatView application.

This is the migrated primary GUI path. It reuses pyDatView's existing IO,
TableList, and PlotData data model, while replacing the wx/matplotlib UI and
plotting surface with Qt widgets and PyQtGraph.
"""

import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field

import numpy as np

from pydatview.Tables import TableList
from pydatview.plotdata import PlotData, PDL_xlabel
import pydatview.io as weio


def _remove_user_site_for_conda_qt():
    if "conda" not in sys.version.lower() and "conda" not in sys.prefix.lower():
        return
    try:
        import site
        user_site = site.getusersitepackages()
    except Exception:
        return
    if not user_site:
        return
    user_site = os.path.abspath(user_site)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != user_site]


_remove_user_site_for_conda_qt()


def _require_qt():
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        import pyqtgraph as pg
    except ImportError as exc:
        raise SystemExit(
            "pyDatView Qt requires PySide6 and pyqtgraph.\n"
            "Install them with: pip install PySide6 pyqtgraph"
        ) from exc
    return QtCore, QtGui, QtWidgets, pg


QtCore, QtGui, QtWidgets, pg = _require_qt()


@dataclass
class LazyFileEntry:
    path: str
    file_format: object
    size: int = 0
    mtime: float = 0.0
    table_indices: list = field(default_factory=list)
    warning: str = ""
    attempted: bool = False
    loading: bool = False
    columns: list = field(default_factory=list)
    header_attempted: bool = False

    @property
    def loaded(self):
        return len(self.table_indices) > 0

    @property
    def basename(self):
        return os.path.basename(self.path)


class LazyLoadWorker(QtCore.QObject):
    finished = QtCore.Signal(int, int, object, str, float, str)

    def __init__(self, generation, lazy_index, path, file_format, options):
        super().__init__()
        self.generation = generation
        self.lazy_index = lazy_index
        self.path = path
        self.file_format = file_format
        self.options = dict(options)

    @QtCore.Slot()
    def run(self):
        t0 = time.perf_counter()
        try:
            loader = TableList(options=self.options)
            tabs, warning = loader._load_file_tabs(self.path, fileformat=self.file_format, bReload=False)
        except Exception as exc:
            tabs = []
            warning = "Error: Failed to open file:\n\n {}\n\n{}: {}\n".format(
                self.path, type(exc).__name__, exc
            )
        self.finished.emit(
            self.generation,
            self.lazy_index,
            tabs,
            warning or "",
            time.perf_counter() - t0,
            getattr(self.file_format, "name", "auto"),
        )


def _resource_path(*parts):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ressources", *parts))


def _format_columns(names, units):
    if units is None:
        return list(names)
    units = [re.sub(r'[()\[\]]', '', str(u)) for u in units]
    if len(names) != len(units):
        return list(names)
    return [str(n) + "_[" + str(u).replace("sec", "s") + "]" for n, u in zip(names, units)]


def _read_fast_ascii_columns(path):
    with open(path, encoding="ascii", errors="ignore") as f:
        for _ in range(35):
            line = f.readline()
            if not line:
                break
            first_word = (line + " dummy").lower().split()[0]
            if first_word in ("time", "alpha"):
                names = line.split()
                units = [unit[1:-1] for unit in f.readline().split()]
                return _format_columns(names, units)
    return []


def _read_fast_binary_columns(path):
    from pydatview.io.fast_output_file import (
        FileFmtID_ChanLen_In,
        FileFmtID_NoCompressWithoutTime,
        FileFmtID_WithTime,
        FileFmtID_WithoutTime,
    )

    def read(fmt, count=1):
        return np.fromfile(fid, dtype=fmt, count=count)

    with open(path, "rb") as fid:
        file_id = int(read(np.int16)[0])
        if file_id not in (
            FileFmtID_WithTime,
            FileFmtID_WithoutTime,
            FileFmtID_NoCompressWithoutTime,
            FileFmtID_ChanLen_In,
        ):
            return []
        len_name = int(read(np.int16)[0]) if file_id == FileFmtID_ChanLen_In else 10
        n_channels = int(read(np.int32)[0])
        read(np.int32)
        if file_id == FileFmtID_WithTime:
            read(np.float64, 2)
        else:
            read(np.float64, 2)
        if file_id != FileFmtID_NoCompressWithoutTime:
            read(np.float32, n_channels * 2)
        desc_len = int(read(np.int32)[0])
        read(np.uint8, desc_len)
        names = []
        units = []
        for _ in range(n_channels + 1):
            raw = read(np.uint8, len_name)
            names.append(bytes(raw).decode("ascii", errors="ignore").strip())
        for _ in range(n_channels + 1):
            raw = read(np.uint8, len_name)
            units.append(bytes(raw).decode("ascii", errors="ignore").strip()[1:-1])
    return _format_columns(names, units)


def read_lazy_columns(path, file_format):
    if getattr(file_format, "name", "") != "FAST output file":
        return []
    ext = os.path.splitext(path.lower())[1]
    if ext == ".outb":
        return _read_fast_binary_columns(path)
    if ext in (".out", ".elev", ".dbg", ".dbg2"):
        return _read_fast_ascii_columns(path)
    return []


def _format_specs(file_format):
    specs = []
    for ext in getattr(file_format, "extensions", []):
        ext = str(ext).strip()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        ext_l = ext.lower()
        if "*" in ext_l:
            specs.append(("prefix", ext_l.split("*", 1)[0]))
        elif "X" in ext:
            pat = "^" + "".join("[0-9]" if c == "X" else re.escape(c.lower()) for c in ext) + "$"
            specs.append(("regex", re.compile(pat, re.IGNORECASE)))
        else:
            specs.append(("suffix", ext_l))
    return specs


def _matches_specs(filename, specs):
    ext = os.path.splitext(filename)[1].lower()
    if not ext:
        return False
    for kind, value in specs:
        if kind == "suffix" and ext == value:
            return True
        if kind == "prefix" and ext.startswith(value):
            return True
        if kind == "regex" and value.match(ext):
            return True
    return False


def _parse_bladed_suffixes(text):
    suffixes = []
    for value in re.split(r"[,;\s]+", text.strip().lower()):
        value = value.strip().lstrip(".").lstrip("$").lstrip("%")
        if value:
            suffixes.append(value)
    return suffixes


def _matches_bladed_suffix(filename, suffixes):
    ext = os.path.splitext(filename)[1].lower()
    if not ext:
        return False
    suffix = ext.lstrip(".").lstrip("$").lstrip("%")
    return suffix in suffixes


def _indexed_format_entries(format_entries, bladed_suffixes=None):
    suffix_formats = {}
    prefix_entries = []
    regex_entries = []
    bladed_suffixes = set(bladed_suffixes or [])

    for fmt, specs in format_entries:
        if not specs:
            continue
        if getattr(fmt, "name", "") == "Bladed output file" and bladed_suffixes:
            for suffix in bladed_suffixes:
                for prefix in (".$", ".%", "."):
                    suffix_formats.setdefault(prefix + suffix, fmt)
            continue
        for kind, value in specs:
            if kind == "suffix":
                suffix_formats.setdefault(value, fmt)
            elif kind == "prefix":
                prefix_entries.append((value, fmt))
            elif kind == "regex":
                regex_entries.append((value, fmt))
    return suffix_formats, prefix_entries, regex_entries


def _match_indexed_format(filename, index):
    ext = os.path.splitext(filename)[1].lower()
    if not ext:
        return None
    suffix_formats, prefix_entries, regex_entries = index
    fmt = suffix_formats.get(ext)
    if fmt is not None:
        return fmt
    for prefix, fmt in prefix_entries:
        if ext.startswith(prefix):
            return fmt
    for regex, fmt in regex_entries:
        if regex.match(ext):
            return fmt
    return None


def scan_readable_files(folder, format_specs, recursive=True):
    matches = []
    if not folder or not os.path.isdir(folder):
        return matches
    stack = [folder]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if recursive:
                                stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False) and _matches_specs(entry.name, format_specs):
                            matches.append(entry.path)
                    except OSError:
                        continue
        except OSError:
            continue
    return sorted(matches)


def scan_readable_file_matches(folder, format_entries, recursive=True, bladed_suffixes=None):
    matches = []
    if not folder or not os.path.isdir(folder):
        return matches
    index = _indexed_format_entries(format_entries, bladed_suffixes=bladed_suffixes)
    stack = [folder]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as dir_entries:
                for entry in dir_entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if recursive:
                                stack.append(entry.path)
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        fmt = _match_indexed_format(entry.name, index)
                        if fmt is not None:
                            matches.append((entry.path, fmt))
                    except OSError:
                        continue
        except OSError:
            continue
    return sorted(matches, key=lambda item: item[0])


def _as_float_array(values):
    arr = np.asarray(values)
    if arr.dtype.kind == "M":
        return arr.astype("datetime64[ns]").astype(np.float64) / 1e9
    if arr.dtype.kind in "biuf":
        return arr.astype(np.float64, copy=False)
    return arr.astype(np.float64)


def _finite_xy(x, y):
    x = _as_float_array(x)
    y = _as_float_array(y)
    if x.shape != y.shape:
        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def _curve_pen(idx, width=1.25):
    return pg.mkPen(color=pg.intColor(idx, hues=12, values=1, maxValue=220), width=width)


class NumericAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        labels = []
        for value in values:
            v = value * scale
            if not np.isfinite(v):
                labels.append("")
            elif abs(v) >= 1e4 or (abs(v) > 0 and abs(v) < 1e-3):
                labels.append("{:.3g}".format(v))
            elif spacing >= 1:
                labels.append("{:.3f}".format(v).rstrip("0").rstrip("."))
            else:
                labels.append("{:.4f}".format(v).rstrip("0").rstrip("."))
        return labels


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
        value = self.dataframe.iat[index.row(), index.column()]
        return "" if value is None else str(value)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole or self.dataframe is None:
            return None
        if orientation == QtCore.Qt.Horizontal:
            return str(self.dataframe.columns[section])
        return str(section)


class ScanDialog(QtWidgets.QDialog):
    def __init__(self, file_formats, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scan folder")
        self.resize(620, 560)
        self.file_formats = list(file_formats)
        self.check_states = {}

        root = QtWidgets.QVBoxLayout(self)

        folder_row = QtWidgets.QHBoxLayout()
        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_edit.setPlaceholderText("Folder containing simulation files")
        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self.browse_folder)
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(browse_button)
        root.addLayout(folder_row)

        self.recursive_check = QtWidgets.QCheckBox("Include subfolders")
        self.recursive_check.setChecked(True)
        root.addWidget(self.recursive_check)

        bladed_row = QtWidgets.QHBoxLayout()
        bladed_row.addWidget(QtWidgets.QLabel("Bladed suffixes"))
        self.bladed_suffix_edit = QtWidgets.QLineEdit()
        self.bladed_suffix_edit.setPlaceholderText("04, 05, 298")
        self.bladed_suffix_edit.setToolTip(
            "Only for Bladed output scans. Example: 04 matches .$04, .%04, or .04."
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

    def accept(self):
        if not os.path.isdir(self.selected_folder()):
            QtWidgets.QMessageBox.warning(self, "Scan folder", "Select a valid folder.")
            return
        if not self.selected_specs():
            QtWidgets.QMessageBox.warning(self, "Scan folder", "Select at least one file type.")
            return
        super().accept()


class QtPlotCanvas(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        pg.setConfigOptions(useOpenGL=True, antialias=False, background="w", foreground="k")
        self.setBackground("w")
        self._plots = []

    def clear_plot(self):
        self.clear()
        self._plots = []

    def plot_data(self, plot_data, *, subplots=False, sharex=True, grid=True,
                  logx=False, logy=False, show_legend=True, line_width=1.25,
                  marker=None, step=False):
        self.clear_plot()
        if len(plot_data) == 0:
            return

        groups = self._group_plot_data(plot_data, subplots)
        previous_plot = None
        curve_idx = 0
        for i_group, group in enumerate(groups):
            plot = self.addPlot(
                row=i_group,
                col=0,
                axisItems={
                    "bottom": NumericAxisItem(orientation="bottom"),
                    "left": NumericAxisItem(orientation="left"),
                    "top": NumericAxisItem(orientation="top"),
                    "right": NumericAxisItem(orientation="right"),
                },
            )
            if previous_plot is not None and sharex:
                plot.setXLink(previous_plot)
            previous_plot = plot
            self._plots.append(plot)

            self._style_plot(plot)
            plot.showGrid(x=grid, y=grid, alpha=0.25)
            ylabel = " and ".join(sorted(set(pd.sy for pd in group)))
            if len(ylabel) < 120:
                plot.setLabel("left", ylabel)
            if i_group == len(groups) - 1:
                plot.setLabel("bottom", PDL_xlabel(plot_data))
            if show_legend:
                plot.addLegend(offset=(10, 10), labelTextColor="k", brush=(255, 255, 255, 210))

            for pd in group:
                try:
                    x, y = _finite_xy(pd.x, pd.y)
                except Exception as exc:
                    print("Skipping non-numeric curve {}: {}".format(pd.sy, exc))
                    continue
                if len(x) == 0:
                    continue
                item = plot.plot(
                    x,
                    y,
                    name=pd.syl or pd.sy,
                    pen=_curve_pen(curve_idx, width=line_width),
                    symbol=marker,
                    symbolSize=5 if marker else None,
                    symbolBrush=pg.intColor(curve_idx, hues=12, values=1, maxValue=220) if marker else None,
                    skipFiniteCheck=True,
                )
                item.setClipToView(True)
                item.setDownsampling(auto=True, method="peak")
                curve_idx += 1

            if logx or logy:
                plot.setLogMode(x=logx, y=logy)

    @staticmethod
    def _style_plot(plot):
        plot.showAxis("bottom", True)
        plot.showAxis("left", True)
        plot.showAxis("top", True)
        plot.showAxis("right", True)
        tick_font = QtWidgets.QApplication.font()
        tick_font.setPointSize(max(8, tick_font.pointSize()))
        for axis_name in ("bottom", "left", "top", "right"):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen("k"))
            axis.setTextPen(pg.mkPen("k"))
            axis.setTickFont(tick_font)
            axis.setStyle(showValues=True, tickLength=5, autoExpandTextSpace=False,
                          autoReduceTextSpace=False)
        plot.getAxis("bottom").setStyle(tickTextHeight=24)
        plot.getAxis("left").setStyle(tickTextWidth=70)
        plot.getAxis("bottom").showLabel(True)
        plot.getAxis("left").showLabel(True)
        plot.getAxis("top").setStyle(showValues=False)
        plot.getAxis("right").setStyle(showValues=False)
        plot.getViewBox().setBackgroundColor("w")
        plot.getViewBox().setBorder(pg.mkPen((180, 180, 180)))

    @staticmethod
    def _group_plot_data(plot_data, subplots):
        if not subplots:
            return [plot_data]
        labels = []
        for pd in plot_data:
            if pd.sy not in labels:
                labels.append(pd.sy)
        return [[pd for pd in plot_data if pd.sy == label] for label in labels]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, filenames=None, dataframes=None, names=None):
        super().__init__()
        self.setWindowTitle("pyDatView Qt")
        self.resize(1280, 820)
        self.tab_list = TableList()
        self.file_formats, self.file_format_errors = self._load_file_formats()
        self.plot_data = []
        self.current_files = []
        self.lazy_entries = []
        self.lazy_load_queue = []
        self.lazy_loader_threads = {}
        self.lazy_loader_workers = {}
        self.lazy_generation = 0
        self.lazy_max_workers = max(1, os.cpu_count() or 1)
        self.lazy_warning_backlog = []
        self.plot_after_lazy_load = False

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
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        top = QtWidgets.QHBoxLayout()
        root.addLayout(top)
        self.plot_type_combo = QtWidgets.QComboBox()
        self.plot_type_combo.addItems(["Regular", "PDF", "FFT", "MinMax"])
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Overlay", "Subplots"])
        self.live_plot = QtWidgets.QCheckBox("Live plot")
        self.live_plot.setChecked(True)
        self.grid_check = QtWidgets.QCheckBox("Grid")
        self.grid_check.setChecked(True)
        self.logx_check = QtWidgets.QCheckBox("Log x")
        self.logy_check = QtWidgets.QCheckBox("Log y")
        self.legend_check = QtWidgets.QCheckBox("Legend")
        self.legend_check.setChecked(True)
        self.line_width_spin = QtWidgets.QDoubleSpinBox()
        self.line_width_spin.setRange(0.25, 8.0)
        self.line_width_spin.setSingleStep(0.25)
        self.line_width_spin.setValue(1.25)
        self.marker_combo = QtWidgets.QComboBox()
        self.marker_combo.addItems(["None", "Circle", "Square", "Triangle", "Diamond"])
        self.status_label = QtWidgets.QLabel("No files loaded")
        top.addWidget(QtWidgets.QLabel("Plot:"))
        top.addWidget(self.plot_type_combo)
        top.addWidget(QtWidgets.QLabel("Mode:"))
        top.addWidget(self.mode_combo)
        top.addWidget(self.live_plot)
        top.addWidget(self.grid_check)
        top.addWidget(self.logx_check)
        top.addWidget(self.logy_check)
        top.addWidget(self.legend_check)
        top.addWidget(QtWidgets.QLabel("LW:"))
        top.addWidget(self.line_width_spin)
        top.addWidget(QtWidgets.QLabel("Marker:"))
        top.addWidget(self.marker_combo)
        top.addStretch(1)
        top.addWidget(self.status_label)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)

        side_layout.addWidget(QtWidgets.QLabel("Tables"))
        self.table_list_widget = QtWidgets.QListWidget()
        self.table_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        side_layout.addWidget(self.table_list_widget, 2)

        side_layout.addWidget(QtWidgets.QLabel("X column"))
        self.column_filter = QtWidgets.QLineEdit()
        self.column_filter.setPlaceholderText("Filter Y columns")
        side_layout.addWidget(self.column_filter)
        self.x_combo = QtWidgets.QComboBox()
        side_layout.addWidget(self.x_combo)

        side_layout.addWidget(QtWidgets.QLabel("Y columns"))
        self.y_list_widget = QtWidgets.QListWidget()
        self.y_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        side_layout.addWidget(self.y_list_widget, 3)

        button_row = QtWidgets.QHBoxLayout()
        self.plot_button = QtWidgets.QPushButton("Plot")
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.select_all_y_button = QtWidgets.QPushButton("All Y")
        self.select_none_y_button = QtWidgets.QPushButton("None")
        self.load_selected_button = QtWidgets.QPushButton("Load selected")
        button_row.addWidget(self.plot_button)
        button_row.addWidget(self.clear_button)
        button_row.addWidget(self.select_all_y_button)
        button_row.addWidget(self.select_none_y_button)
        button_row.addWidget(self.load_selected_button)
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
        self.stats_text = QtWidgets.QPlainTextEdit()
        self.stats_text.setReadOnly(True)
        self.detail_tabs.addTab(self.table_view, "Data")
        self.detail_tabs.addTab(self.stats_text, "Stats")
        self.detail_tabs.addTab(self.info_text, "File info")

        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_splitter.addWidget(self.canvas)
        right_splitter.addWidget(self.detail_tabs)
        right_splitter.setStretchFactor(0, 4)
        right_splitter.setStretchFactor(1, 1)
        right_splitter.setSizes([620, 180])

        splitter.addWidget(side)
        splitter.addWidget(right_splitter)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 960])

        self.setStatusBar(QtWidgets.QStatusBar())
        self._apply_light_borders()

    def _apply_light_borders(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #f6f7f9;
                color: #1f2933;
            }
            QMenuBar {
                background: #edf0f4;
                border-bottom: 2px solid #9aa7b4;
                spacing: 4px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 5px 10px;
                border: 1px solid transparent;
            }
            QMenuBar::item:selected,
            QMenuBar::item:pressed {
                background: #d9e2ec;
                border: 1px solid #778899;
            }
            QMenu {
                background: #ffffff;
                border: 2px solid #778899;
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
                background: #eef2f6;
                border: 1px solid #9aa7b4;
                border-left: 0;
                border-right: 0;
                spacing: 5px;
                padding: 3px;
            }
            QToolButton {
                background: #ffffff;
                border: 1px solid #9aa7b4;
                padding: 4px;
            }
            QToolButton:hover {
                background: #e5effa;
                border-color: #4f83cc;
            }
            QSplitter::handle {
                background: #9aa7b4;
            }
            QListWidget, QTableView, QPlainTextEdit, QLineEdit, QComboBox, QDoubleSpinBox {
                background: #ffffff;
                border: 1px solid #8b9aaa;
                selection-background-color: #2563eb;
                selection-color: #ffffff;
            }
            QListWidget::item {
                padding: 3px 5px;
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
                background: #ffffff;
                border: 1px solid #7f8fa3;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background: #e5effa;
                border-color: #4f83cc;
            }
            QTabWidget::pane {
                border: 1px solid #8b9aaa;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #e7ebf0;
                border: 1px solid #8b9aaa;
                padding: 5px 12px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                border-bottom-color: #ffffff;
            }
            QStatusBar {
                background: #edf0f4;
                border-top: 1px solid #9aa7b4;
            }
        """)

    def _build_actions(self):
        file_menu = self.menuBar().addMenu("&File")
        self.open_action = file_menu.addAction("Open")
        self.add_action = file_menu.addAction("Add")
        self.reload_action = file_menu.addAction("Reload")
        self.scan_action = file_menu.addAction(QtGui.QIcon(_resource_path("icons", "scan.png")), "Scan folder")
        export_action = file_menu.addAction("Export selected table")
        file_menu.addSeparator()
        quit_action = file_menu.addAction("Quit")
        self.open_action.triggered.connect(lambda: self.select_files(add=False))
        self.add_action.triggered.connect(lambda: self.select_files(add=True))
        self.reload_action.triggered.connect(self.reload_files)
        self.scan_action.triggered.connect(self.scan_folder)
        export_action.triggered.connect(self.export_selected_table)
        quit_action.triggered.connect(self.close)

        toolbar = self.addToolBar("Main")
        toolbar.setObjectName("main_toolbar")
        toolbar.setIconSize(QtCore.QSize(24, 24))
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.add_action)
        toolbar.addAction(self.reload_action)
        toolbar.addSeparator()
        toolbar.addAction(self.scan_action)

        view_menu = self.menuBar().addMenu("&View")
        self.autorange_action = view_menu.addAction("Auto range")
        self.autorange_action.triggered.connect(self.auto_range)
        export_plot_action = view_menu.addAction("Export plot image")
        export_plot_action.triggered.connect(self.export_plot_image)

    def _connect(self):
        self.table_list_widget.itemSelectionChanged.connect(self.on_table_selection_changed)
        self.plot_type_combo.currentIndexChanged.connect(self.on_selection_changed)
        self.x_combo.currentIndexChanged.connect(self.on_selection_changed)
        self.y_list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        self.mode_combo.currentIndexChanged.connect(self.on_selection_changed)
        self.grid_check.stateChanged.connect(self.on_selection_changed)
        self.logx_check.stateChanged.connect(self.on_selection_changed)
        self.logy_check.stateChanged.connect(self.on_selection_changed)
        self.legend_check.stateChanged.connect(self.on_selection_changed)
        self.line_width_spin.valueChanged.connect(self.on_selection_changed)
        self.marker_combo.currentIndexChanged.connect(self.on_selection_changed)
        self.column_filter.textChanged.connect(self.populate_columns)
        self.plot_button.clicked.connect(self.redraw)
        self.clear_button.clicked.connect(self.clear)
        self.select_all_y_button.clicked.connect(self.select_all_y)
        self.select_none_y_button.clicked.connect(self.select_none_y)
        self.load_selected_button.clicked.connect(self.load_selected_lazy_files)

    def _show_file_format_errors(self):
        for err in self.file_format_errors:
            self.statusBar().showMessage(str(err), 10000)

    def select_files(self, add=False):
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open files",
            "",
            "All supported files (*);;All files (*)",
        )
        if filenames:
            self.load_files(filenames, add=add)

    def scan_folder(self):
        dialog = ScanDialog(self.file_formats, self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return

        folder = dialog.selected_folder()
        format_entries = dialog.selected_format_entries()
        recursive = dialog.recursive()
        bladed_suffixes = dialog.bladed_suffixes()
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            t0 = time.perf_counter()
            self.statusBar().showMessage("Scanning {} ...".format(folder))
            QtWidgets.QApplication.processEvents()
            matches = scan_readable_file_matches(
                folder,
                format_entries,
                recursive=recursive,
                bladed_suffixes=bladed_suffixes,
            )
            scan_seconds = time.perf_counter() - t0
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        if not matches:
            QtWidgets.QMessageBox.information(
                self,
                "Scan folder",
                "No matching readable files were found in:\n{}".format(folder),
            )
            self.statusBar().showMessage("Scan found no files in {:.3f}s".format(scan_seconds), 8000)
            return

        self.set_lazy_file_index(matches)
        self.statusBar().showMessage(
            "Indexed {:,} files in {:.3f}s; loaded 0".format(len(matches), scan_seconds),
            12000,
        )

    def load_files(self, filenames, add=False, fileformats=None, status_prefix="Loading files"):
        t0 = time.perf_counter()
        try:
            if fileformats is None:
                pairs = [(f, None) for f in filenames if os.path.isfile(f)]
            else:
                pairs = [(f, ff) for f, ff in zip(filenames, fileformats) if os.path.isfile(f)]
            pairs = sorted(pairs, key=lambda item: item[0])
            filenames = [f for f, _ in pairs]
            fileformats = [ff for _, ff in pairs]
            if not filenames:
                return None
            if self.lazy_entries:
                self.lazy_generation += 1
                self.lazy_load_queue = []
                self.lazy_warning_backlog = []
                self.lazy_entries = []
            if not add:
                self.tab_list.clean()
                self.current_files = []

            last_status = {"t": 0.0}

            def status_function(i):
                now = time.perf_counter()
                if i == 0 or i == len(filenames) - 1 or now - last_status["t"] > 0.15:
                    last_status["t"] = now
                    self.status_label.setText("{} {}/{}".format(status_prefix, i + 1, len(filenames)))
                    self.statusBar().showMessage("{} {}/{}".format(status_prefix, i + 1, len(filenames)))
                    QtWidgets.QApplication.processEvents()

            new_tabs, warnings = self.tab_list.load_tables_from_files(
                filenames=filenames,
                fileformats=fileformats,
                bAdd=add,
                bReload=False,
                statusFunction=status_function,
            )
            self.current_files = self.tab_list.filenames
            warnings = [warning for warning in warnings if warning]
            if warnings:
                shown = "\n\n".join(warnings[:5])
                if len(warnings) > 5:
                    shown += "\n\n... {} more warnings".format(len(warnings) - 5)
                QtWidgets.QMessageBox.warning(self, "Load warnings", shown)
            if len(new_tabs) == 0 and len(self.tab_list) == 0:
                self.status_label.setText("No tables loaded")
                return time.perf_counter() - t0
            self.populate_tables()
            self.status_label.setText("{} tables loaded".format(len(self.tab_list)))
            self.redraw()
            return time.perf_counter() - t0
        except Exception as exc:
            self.show_exception("Failed to load files", exc)
            return None

    def set_lazy_file_index(self, matches):
        self.lazy_generation += 1
        self.lazy_load_queue = []
        self.lazy_warning_backlog = []
        self.tab_list.clean()
        self.current_files = [path for path, _ in matches]
        self.lazy_entries = []
        for path, fmt in matches:
            try:
                stat = os.stat(path)
                size = stat.st_size
                mtime = stat.st_mtime
            except OSError:
                size = 0
                mtime = 0.0
            self.lazy_entries.append(LazyFileEntry(path=path, file_format=fmt, size=size, mtime=mtime))
        self.populate_tables()
        self.clear()
        self.status_label.setText("{:,} files indexed, 0 loaded".format(len(self.lazy_entries)))

    def lazy_loaded_count(self):
        return sum(1 for entry in self.lazy_entries if entry.loaded)

    def lazy_item_text(self, entry):
        if entry.loaded:
            state = "loaded"
        elif entry.loading:
            state = "loading"
        elif entry.attempted:
            state = "failed"
        else:
            state = "indexed"
        size_mb = entry.size / (1024 * 1024) if entry.size else 0.0
        fmt_name = getattr(entry.file_format, "name", "auto")
        return "{}  [{} | {:.2f} MB | {}]".format(entry.basename, state, size_mb, fmt_name)

    def ensure_lazy_header(self, lazy_index):
        entry = self.lazy_entries[lazy_index]
        if entry.columns or entry.header_attempted or entry.loaded:
            return
        entry.header_attempted = True
        try:
            entry.columns = read_lazy_columns(entry.path, entry.file_format)
        except Exception as exc:
            entry.warning = "Header read failed: {}: {}".format(type(exc).__name__, exc)

    def ensure_lazy_loaded(self, lazy_index, show_warning=True):
        entry = self.lazy_entries[lazy_index]
        if entry.loaded:
            return entry.table_indices
        if entry.attempted:
            if entry.warning and show_warning:
                QtWidgets.QMessageBox.warning(self, "Load warning", entry.warning)
            return []
        self.queue_lazy_load(lazy_index)
        return []

    def queue_lazy_load(self, lazy_index):
        entry = self.lazy_entries[lazy_index]
        if entry.loaded or entry.loading or entry.attempted or lazy_index in self.lazy_load_queue:
            return
        entry.loading = True
        self.lazy_load_queue.append(lazy_index)
        self.status_label.setText("Loading {}".format(entry.basename))
        self.statusBar().showMessage("Queued {}".format(entry.path))
        self.update_lazy_item(lazy_index)
        self.start_next_lazy_load()

    def start_next_lazy_load(self):
        while len(self.lazy_loader_threads) < self.lazy_max_workers and self.lazy_load_queue:
            self.start_one_lazy_load()

    def start_one_lazy_load(self):
        if not self.lazy_load_queue:
            return
        lazy_index = self.lazy_load_queue.pop(0)
        if lazy_index >= len(self.lazy_entries):
            self.start_next_lazy_load()
            return
        entry = self.lazy_entries[lazy_index]
        self.status_label.setText("Loading {}".format(entry.basename))
        self.statusBar().showMessage("Loading {}".format(entry.path))

        generation = self.lazy_generation
        thread = QtCore.QThread(self)
        worker = LazyLoadWorker(generation, lazy_index, entry.path, entry.file_format, self.tab_list.options)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self.on_lazy_load_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda idx=lazy_index: self.on_lazy_thread_finished(idx))
        self.lazy_loader_threads[lazy_index] = thread
        self.lazy_loader_workers[lazy_index] = worker
        thread.start()

    def on_lazy_load_finished(self, generation, lazy_index, tabs, warning, elapsed, format_name):
        if generation != self.lazy_generation:
            return
        if lazy_index >= len(self.lazy_entries):
            return
        entry = self.lazy_entries[lazy_index]
        start = len(self.tab_list)
        if tabs:
            self.tab_list.append(tabs)
            entry.table_indices = list(range(start, start + len(tabs)))
        entry.warning = warning or ""
        entry.attempted = True
        entry.loading = False
        self.update_lazy_item(lazy_index)
        self.current_files = sorted(set(self.current_files + self.tab_list.filenames))
        self.status_label.setText(
            "{:,} files indexed, {:,} loaded, {:,} active".format(
                len(self.lazy_entries), self.lazy_loaded_count(), len(self.lazy_loader_threads)
            )
        )
        n_rows = sum(getattr(tab, "nRows", 0) for tab in tabs) if tabs else 0
        n_cols = sum(getattr(tab, "nCols", 0) for tab in tabs) if tabs else 0
        self.statusBar().showMessage(
            "Loaded {} in {:.3f}s ({}, {:,} rows, {:,} cols)".format(
                entry.basename, elapsed, format_name, n_rows, n_cols
            ),
            12000,
        )
        if entry.warning:
            self.lazy_warning_backlog.append(entry.warning)
        if self.is_lazy_selected(lazy_index):
            self.on_table_selection_changed()
        if self.plot_after_lazy_load and not self.has_unloaded_lazy_selection():
            self.plot_after_lazy_load = False
            self.redraw()

    def on_lazy_thread_finished(self, lazy_index):
        self.lazy_loader_threads.pop(lazy_index, None)
        self.lazy_loader_workers.pop(lazy_index, None)
        self.status_label.setText(
            "{:,} files indexed, {:,} loaded, {:,} active".format(
                len(self.lazy_entries), self.lazy_loaded_count(), len(self.lazy_loader_threads)
            )
        )
        self.start_next_lazy_load()

    def is_lazy_selected(self, lazy_index):
        for item in self.table_list_widget.selectedItems():
            data = item.data(QtCore.Qt.UserRole)
            if isinstance(data, tuple) and data == ("lazy", lazy_index):
                return True
        return False

    def update_lazy_item(self, lazy_index):
        for row in range(self.table_list_widget.count()):
            item = self.table_list_widget.item(row)
            data = item.data(QtCore.Qt.UserRole)
            if isinstance(data, tuple) and data == ("lazy", lazy_index):
                item.setText(self.lazy_item_text(self.lazy_entries[lazy_index]))
                return

    def load_selected_lazy_files(self):
        lazy_indices = self.selected_lazy_indices()
        if not lazy_indices:
            return
        for i, lazy_index in enumerate(lazy_indices):
            self.statusBar().showMessage("Queueing selected file {}/{}".format(i + 1, len(lazy_indices)))
            self.ensure_lazy_loaded(lazy_index, show_warning=False)
        self.on_table_selection_changed()

    def load_dfs(self, dataframes, names=None):
        if not isinstance(dataframes, list):
            dataframes = [dataframes]
        if names is None:
            names = ["df{}".format(i + 1) for i in range(len(dataframes))]
        if not isinstance(names, list):
            names = [names]
        self.lazy_generation += 1
        self.lazy_load_queue = []
        self.lazy_warning_backlog = []
        self.lazy_entries = []
        self.tab_list.from_dataframes(dataframes=dataframes, names=names, bAdd=False)
        self.populate_tables()
        self.redraw()

    def reload_files(self):
        if self.lazy_entries:
            self.lazy_generation += 1
            self.lazy_load_queue = []
            self.lazy_warning_backlog = []
            for entry in self.lazy_entries:
                entry.table_indices = []
                entry.warning = ""
                entry.attempted = False
                entry.loading = False
                entry.columns = []
                entry.header_attempted = False
            self.tab_list.clean()
            self.populate_tables()
            self.clear()
            self.status_label.setText("{:,} files indexed, 0 loaded".format(len(self.lazy_entries)))
            return
        filenames = sorted(set(f for f in self.current_files if f))
        if filenames:
            self.load_files(filenames, add=False)

    def populate_tables(self):
        self.table_list_widget.blockSignals(True)
        self.table_list_widget.clear()
        if self.lazy_entries:
            for i, entry in enumerate(self.lazy_entries):
                item = QtWidgets.QListWidgetItem(self.lazy_item_text(entry))
                item.setData(QtCore.Qt.UserRole, ("lazy", i))
                self.table_list_widget.addItem(item)
        else:
            names = self.tab_list.getDisplayTabNames()
            for i, tab in enumerate(self.tab_list):
                item = QtWidgets.QListWidgetItem("{}  ({})".format(names[i], tab.shapestring))
                item.setData(QtCore.Qt.UserRole, ("table", i))
                self.table_list_widget.addItem(item)
        if self.table_list_widget.count() > 0 and not self.lazy_entries:
            self.table_list_widget.item(0).setSelected(True)
        self.table_list_widget.blockSignals(False)
        self.on_table_selection_changed()

    def selected_lazy_indices(self):
        items = self.table_list_widget.selectedItems()
        indices = []
        for item in items:
            data = item.data(QtCore.Qt.UserRole)
            if isinstance(data, tuple) and data[0] == "lazy":
                indices.append(data[1])
        return indices

    def selected_table_indices(self, load=True, show_warning=False):
        items = self.table_list_widget.selectedItems()
        indices = []
        for item in items:
            data = item.data(QtCore.Qt.UserRole)
            if isinstance(data, tuple) and data[0] == "table":
                indices.append(data[1])
            elif isinstance(data, tuple) and data[0] == "lazy":
                entry = self.lazy_entries[data[1]]
                if entry.loaded:
                    indices.extend(entry.table_indices)
                elif load:
                    indices.extend(self.ensure_lazy_loaded(data[1], show_warning=show_warning))
        return indices

    def on_table_selection_changed(self):
        self.populate_columns()
        self.update_table_preview()
        self.update_file_info()
        self.on_selection_changed()

    def populate_columns(self):
        previous_x = self.x_combo.currentData()
        previous_y = set(self.selected_y_indices_original())
        lazy_indices = self.selected_lazy_indices()
        indices = []
        columns = []
        if lazy_indices:
            lazy_index = lazy_indices[0]
            entry = self.lazy_entries[lazy_index]
            if entry.loaded:
                indices = list(entry.table_indices)
            else:
                self.ensure_lazy_header(lazy_index)
                columns = list(entry.columns)
        if not lazy_indices:
            indices = self.selected_table_indices(load=False)
        if not indices and len(self.tab_list) > 0 and not self.lazy_entries:
            indices = [0]
        if indices and not columns:
            columns = list(self.tab_list[indices[0]].columns)
        all_columns = [(i, str(col)) for i, col in enumerate(columns)]
        text_filter = self.column_filter.text().strip().lower()
        visible_y = [(i, col) for i, col in all_columns
                     if not text_filter or text_filter in col.lower()]

        self.x_combo.blockSignals(True)
        self.y_list_widget.blockSignals(True)
        self.x_combo.clear()
        self.y_list_widget.clear()
        for original_i, col in all_columns:
            self.x_combo.addItem(col, original_i)
        for original_i, col in visible_y:
            item = QtWidgets.QListWidgetItem(col)
            item.setData(QtCore.Qt.UserRole, original_i)
            self.y_list_widget.addItem(item)

        if all_columns:
            all_indices = [i for i, _ in all_columns]
            if previous_x in all_indices:
                x_to_select = previous_x
            else:
                x_to_select = next((i for i, col in all_columns if col.lower().startswith("time")), all_columns[0][0])
            self.x_combo.setCurrentIndex(all_indices.index(x_to_select))
        if visible_y and not previous_y:
            x_current = self.x_combo.currentData()
            default_row = next((row for row, (i, _) in enumerate(visible_y) if i != x_current), 0)
            self.y_list_widget.item(default_row).setSelected(True)
        else:
            for row in range(self.y_list_widget.count()):
                item = self.y_list_widget.item(row)
                if item.data(QtCore.Qt.UserRole) in previous_y:
                    item.setSelected(True)
        self.x_combo.blockSignals(False)
        self.y_list_widget.blockSignals(False)

    def on_selection_changed(self):
        if self.live_plot.isChecked() and not self.has_unloaded_lazy_selection():
            self.redraw()

    def has_unloaded_lazy_selection(self):
        for lazy_index in self.selected_lazy_indices():
            if not self.lazy_entries[lazy_index].loaded:
                return True
        return False

    def select_all_y(self):
        self.y_list_widget.blockSignals(True)
        for row in range(self.y_list_widget.count()):
            self.y_list_widget.item(row).setSelected(True)
        self.y_list_widget.blockSignals(False)
        self.on_selection_changed()

    def select_none_y(self):
        self.y_list_widget.blockSignals(True)
        for row in range(self.y_list_widget.count()):
            self.y_list_widget.item(row).setSelected(False)
        self.y_list_widget.blockSignals(False)
        self.on_selection_changed()

    def selected_y_indices(self):
        return self.selected_y_indices_original()

    def selected_y_indices_original(self):
        return [item.data(QtCore.Qt.UserRole) for item in self.y_list_widget.selectedItems()]

    def build_plot_data(self):
        plot_data = []
        table_indices = self.selected_table_indices()
        y_indices = self.selected_y_indices()
        ix = self.x_combo.currentData()
        if ix is None or not y_indices or not table_indices:
            return plot_data

        same_col = len(table_indices) > 1
        for it in table_indices:
            tab = self.tab_list[it]
            for iy in y_indices:
                if iy >= len(tab.columns):
                    continue
                idx = (it, ix, iy, str(tab.columns[ix]), str(tab.columns[iy]), tab.active_name)
                pd = PlotData()
                pd.fromIDs(self.tab_list, len(plot_data), idx, same_col, pipeline=None)
                self.apply_plot_type(pd)
                if len(table_indices) == 1:
                    pd.syl = pd.sy
                else:
                    pd.syl = "{} - {}".format(pd.st, pd.sy)
                plot_data.append(pd)
        return plot_data

    def apply_plot_type(self, pd):
        plot_type = self.plot_type_combo.currentText()
        if plot_type == "PDF":
            pd.toPDF(nBins=101, smooth=False)
        elif plot_type == "FFT":
            pd.toFFT(yType="PSD", xType="1/x", avgMethod="Welch", avgWindow="Hamming",
                     bDetrend=True, nExp=11, nPerDecade=20)
        elif plot_type == "MinMax":
            pd.toMinMax(xScale=False, yScale=True, yCenter="None")

    def redraw(self):
        try:
            if self.has_unloaded_lazy_selection():
                self.plot_after_lazy_load = True
                self.selected_table_indices(load=True)
                self.statusBar().showMessage("Loading selected files before plotting ...", 8000)
                return
            self.plot_data = self.build_plot_data()
            self.canvas.plot_data(
                self.plot_data,
                subplots=self.mode_combo.currentText() == "Subplots",
                sharex=True,
                grid=self.grid_check.isChecked(),
                logx=self.logx_check.isChecked(),
                logy=self.logy_check.isChecked(),
                show_legend=self.legend_check.isChecked(),
                line_width=self.line_width_spin.value(),
                marker=self.marker_symbol(),
            )
            n_curves = len(self.plot_data)
            n_points = sum(len(pd.y) for pd in self.plot_data)
            self.update_stats()
            self.statusBar().showMessage("{} curves, {:,} points".format(n_curves, n_points))
        except Exception as exc:
            self.show_exception("Failed to plot data", exc)

    def clear(self):
        self.canvas.clear_plot()
        self.plot_data = []

    def auto_range(self):
        for plot in self.canvas._plots:
            plot.autoRange()

    def marker_symbol(self):
        return {
            "None": None,
            "Circle": "o",
            "Square": "s",
            "Triangle": "t",
            "Diamond": "d",
        }.get(self.marker_combo.currentText(), None)

    def update_table_preview(self):
        indices = self.selected_table_indices(load=False)
        if not indices:
            self.table_model.set_dataframe(None)
            return
        self.table_model.set_dataframe(self.tab_list[indices[0]].data)

    def update_file_info(self):
        lazy_indices = self.selected_lazy_indices()
        if lazy_indices:
            lines = []
            for lazy_index in lazy_indices:
                entry = self.lazy_entries[lazy_index]
                if entry.loaded:
                    status = "loaded"
                elif entry.loading:
                    status = "loading"
                elif entry.attempted:
                    status = "failed"
                else:
                    status = "indexed"
                lines.append("File: {}".format(entry.path))
                lines.append("Format: {}".format(getattr(entry.file_format, "name", "auto")))
                lines.append("Status: {}".format(status))
                lines.append("Size: {:.3f} MB".format(entry.size / (1024 * 1024) if entry.size else 0.0))
                if entry.mtime:
                    lines.append("Modified: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.mtime))))
                if entry.warning:
                    lines.append("Warning: {}".format(entry.warning.splitlines()[0]))
                lines.append("")
            self.info_text.setPlainText("\n".join(lines))
            return

        indices = self.selected_table_indices(load=False)
        if not indices:
            self.info_text.clear()
            return
        lines = []
        for it in indices:
            tab = self.tab_list[it]
            lines.append("Table: {}".format(tab.active_name))
            lines.append("File: {}".format(tab.filename))
            lines.append("Format: {}".format(tab.fileformat_name))
            lines.append("Shape: {}".format(tab.shapestring))
            lines.append("Columns: {}".format(", ".join(map(str, tab.columns[:40]))))
            if len(tab.columns) > 40:
                lines.append("...")
            lines.append("")
        self.info_text.setPlainText("\n".join(lines))

    def update_stats(self):
        if not self.plot_data:
            self.stats_text.clear()
            return
        lines = []
        for pd in self.plot_data:
            try:
                _, y = _finite_xy(pd.x, pd.y)
            except Exception:
                continue
            if len(y) == 0:
                continue
            lines.append(pd.syl or pd.sy)
            lines.append("  n    = {:,}".format(len(y)))
            lines.append("  min  = {:.6g}".format(np.nanmin(y)))
            lines.append("  mean = {:.6g}".format(np.nanmean(y)))
            lines.append("  max  = {:.6g}".format(np.nanmax(y)))
            lines.append("  std  = {:.6g}".format(np.nanstd(y)))
            lines.append("")
        self.stats_text.setPlainText("\n".join(lines))

    def export_plot_image(self):
        if not self.canvas._plots:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export plot image",
            "pydatview_plot.png",
            "PNG files (*.png);;SVG files (*.svg);;All files (*)",
        )
        if not path:
            return
        try:
            from pyqtgraph.exporters import ImageExporter, SVGExporter
            exporter_cls = SVGExporter if path.lower().endswith(".svg") else ImageExporter
            exporter = exporter_cls(self.canvas.scene())
            exporter.export(path)
        except Exception as exc:
            self.show_exception("Failed to export plot image", exc)

    def export_selected_table(self):
        indices = self.selected_table_indices()
        if not indices:
            return
        tab = self.tab_list[indices[0]]
        default = (tab.basename if tab.filename else tab.name) + ".csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export selected table",
            default,
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            try:
                tab.export(path=path, fformat="csv")
            except Exception as exc:
                self.show_exception("Failed to export table", exc)

    def show_exception(self, title, exc):
        traceback.print_exc()
        QtWidgets.QMessageBox.critical(self, title, "{}\n\n{}".format(exc, traceback.format_exc(limit=5)))


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
