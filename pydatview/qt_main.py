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


@dataclass
class SelectorPane:
    frame: object
    table_list_widget: object
    bladed_dataset_label: object
    bladed_dataset_combo: object
    column_filter: object
    x_combo: object
    y_list_widget: object


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
    if finite.all():
        return x, y
    return x[finite], y[finite]


_PLOT_PALETTE = (
    (0, 87, 184),     # blue
    (209, 73, 0),     # vermilion
    (0, 135, 90),     # green
    (180, 35, 24),    # red
    (111, 66, 193),   # purple
    (0, 124, 145),    # teal
    (194, 24, 91),    # magenta
    (138, 90, 0),     # ochre
    (29, 78, 216),    # royal blue
    (162, 59, 114),   # berry
    (46, 125, 50),    # dark green
    (109, 76, 65),    # brown
    (0, 96, 100),     # dark cyan
    (156, 39, 176),   # violet
    (230, 81, 0),     # burnt orange
    (55, 65, 81),     # charcoal
)


def _curve_color(idx):
    return _PLOT_PALETTE[idx % len(_PLOT_PALETTE)]


def _curve_pen(idx, width=1.25):
    return pg.mkPen(color=_curve_color(idx), width=width)


def _selected_curve_pen(width=1.25):
    return pg.mkPen(color=(17, 24, 39), width=max(width + 2.5, 3.5))


def _default_lazy_workers():
    cpu_count = max(1, os.cpu_count() or 1)
    env_value = os.environ.get("PYDATVIEW_MAX_WORKERS")
    if env_value:
        try:
            return max(1, min(cpu_count, int(env_value)))
        except ValueError:
            print("[pyDatView] Ignoring invalid PYDATVIEW_MAX_WORKERS={!r}".format(env_value))
    if sys.platform.startswith("win"):
        return min(cpu_count, 8)
    return min(cpu_count, 32)


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

        bladed_row = QtWidgets.QHBoxLayout()
        bladed_row.addWidget(QtWidgets.QLabel("Bladed suffixes"))
        self.bladed_suffix_edit = QtWidgets.QLineEdit()
        self.bladed_suffix_edit.setPlaceholderText("04, 05, 298")
        self.bladed_suffix_edit.setText(str(self.settings.value("scan/bladed_suffixes", "") or ""))
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
        self.settings.setValue("scan/bladed_suffixes", self.bladed_suffix_edit.text().strip())
        self.settings.setValue("scan/formats", selected_formats)
        self.settings.setValue("scan/geometry", self.saveGeometry())
        self.settings.sync()
        super().accept()


class QtPlotCanvas(pg.GraphicsLayoutWidget):
    curveSelected = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        pg.setConfigOptions(useOpenGL=True, antialias=False, background="w", foreground="k")
        self.setBackground("w")
        self._plots = []
        self._curve_items = []
        self._selected_curve = None

    def clear_plot(self):
        self.clear()
        self._plots = []
        self._curve_items = []
        self._selected_curve = None

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
                curve_color = _curve_color(curve_idx)
                item = plot.plot(
                    x,
                    y,
                    name=pd.syl or pd.sy,
                    pen=_curve_pen(curve_idx, width=line_width),
                    symbol=marker,
                    symbolSize=5 if marker else None,
                    symbolBrush=curve_color if marker else None,
                    symbolPen=pg.mkPen(curve_color) if marker else None,
                    skipFiniteCheck=True,
                )
                item.setClipToView(True)
                item.setDownsampling(auto=True, method="peak")
                item.setCurveClickable(True, width=8)
                base_pen = _curve_pen(curve_idx, width=line_width)
                meta = {
                    "label": pd.syl or pd.sy,
                    "file": getattr(pd, "st", ""),
                    "filename": getattr(pd, "filename", ""),
                    "table_index": getattr(pd, "it", None),
                    "pane_index": getattr(pd, "pane_index", 0),
                    "x": getattr(pd, "sx", ""),
                    "y": getattr(pd, "sy", ""),
                    "points": len(x),
                    "line_width": line_width,
                }
                item.sigClicked.connect(lambda clicked_item, _ev, meta=meta: self.select_curve(clicked_item, meta))
                self._curve_items.append((item, base_pen, meta))
                curve_idx += 1

            if logx or logy:
                plot.setLogMode(x=logx, y=logy)

    def select_curve(self, selected_item, meta):
        for item, base_pen, _ in self._curve_items:
            item.setPen(base_pen)
        selected_item.setPen(_selected_curve_pen(meta.get("line_width", 1.25)))
        selected_item.setZValue(10)
        for item, _, _ in self._curve_items:
            if item is not selected_item:
                item.setZValue(0)
        self._selected_curve = selected_item
        self.curveSelected.emit(meta)

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
        self.settings = QtCore.QSettings("NREL", "pyDatView")
        self.tab_list = TableList()
        self.file_formats, self.file_format_errors = self._load_file_formats()
        self.plot_data = []
        self.current_files = []
        self.lazy_entries = []
        self.lazy_load_queue = []
        self.lazy_loader_threads = {}
        self.lazy_loader_workers = {}
        self.lazy_generation = 0
        self.lazy_max_workers = _default_lazy_workers()
        self.lazy_warning_backlog = []
        self.plot_after_lazy_load = False
        self.selector_panes = []
        self.lazy_batch_total = 0
        self.lazy_batch_done = 0

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
        self.plot_type_combo.addItems(["Regular", "PDF", "FFT", "MinMax"])
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Overlay", "Subplots"])
        self.compare_combo = QtWidgets.QComboBox()
        self.compare_combo.addItems(["Auto", "2", "3"])
        self.live_plot = QtWidgets.QCheckBox("Live plot")
        self.live_plot.setChecked(True)
        self.grid_check = QtWidgets.QCheckBox("Grid")
        self.grid_check.setChecked(False)
        self.logx_check = QtWidgets.QCheckBox("Log x")
        self.logy_check = QtWidgets.QCheckBox("Log y")
        self.legend_check = QtWidgets.QCheckBox("Legend")
        self.legend_check.setChecked(False)
        self.line_width_spin = QtWidgets.QDoubleSpinBox()
        self.line_width_spin.setRange(0.25, 8.0)
        self.line_width_spin.setSingleStep(0.25)
        self.line_width_spin.setValue(1.25)
        self.marker_combo = QtWidgets.QComboBox()
        self.marker_combo.addItems(["None", "Circle", "Square", "Triangle", "Diamond"])
        self.load_workers_combo = QtWidgets.QComboBox()
        self.load_workers_combo.addItems(["Auto", "1", "2", "4", "8", "16", "32", "64", "96"])
        self.load_workers_combo.setToolTip("Maximum parallel file load workers. Auto is capped on Windows to reduce UI hangs.")
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
        load_controls = QtWidgets.QHBoxLayout()
        load_controls.setContentsMargins(0, 0, 0, 0)
        load_controls.setSpacing(6)
        load_controls.addWidget(QtWidgets.QLabel("Workers"))
        load_controls.addWidget(self.load_workers_combo)
        load_controls.addWidget(self.loading_progress)
        top.addLayout(load_controls, 1, 10)

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

        button_row = QtWidgets.QHBoxLayout()
        self.plot_button = QtWidgets.QPushButton("Plot")
        self.plot_button.setObjectName("primaryButton")
        self.plot_button.setIcon(QtGui.QIcon(_resource_path("icons", "chart.svg")))
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.select_all_y_button = QtWidgets.QPushButton("All Y")
        self.select_none_y_button = QtWidgets.QPushButton("None")
        self.load_selected_button = QtWidgets.QPushButton("Load selected")
        self.load_selected_button.setToolTip("Parse the selected indexed files and cache them in memory")
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

        self.main_splitter.addWidget(side)
        self.main_splitter.addWidget(right_splitter)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([340, 940])

        self.setStatusBar(QtWidgets.QStatusBar())
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
        layout.addWidget(table_list_widget, 2)
        bladed_dataset_label = QtWidgets.QLabel("BLADED VARIABLES")
        bladed_dataset_label.setProperty("sectionLabel", True)
        bladed_dataset_label.setVisible(False)
        layout.addWidget(bladed_dataset_label)
        bladed_dataset_combo = QtWidgets.QComboBox()
        bladed_dataset_combo.setToolTip("Variable group loaded from the selected Bladed .$PJ project")
        bladed_dataset_combo.setVisible(False)
        layout.addWidget(bladed_dataset_combo)
        x_label = QtWidgets.QLabel("X COLUMN")
        x_label.setProperty("sectionLabel", True)
        layout.addWidget(x_label)
        column_filter = QtWidgets.QLineEdit()
        column_filter.setPlaceholderText("Filter Y columns")
        column_filter.setClearButtonEnabled(True)
        layout.addWidget(column_filter)
        x_combo = QtWidgets.QComboBox()
        layout.addWidget(x_combo)
        y_label = QtWidgets.QLabel("Y COLUMNS")
        y_label.setProperty("sectionLabel", True)
        layout.addWidget(y_label)
        y_list_widget = QtWidgets.QListWidget()
        y_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
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
        table_list_widget.itemSelectionChanged.connect(self.on_table_selection_changed)
        bladed_dataset_combo.currentIndexChanged.connect(
            lambda _index, p=pane: self.on_bladed_dataset_changed(p)
        )
        x_combo.currentIndexChanged.connect(self.on_selection_changed)
        y_list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        column_filter.textChanged.connect(lambda _text, p=pane: self.populate_columns(p))
        self.selector_splitter.addWidget(frame)
        self.selector_panes.append(pane)
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
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#statusChip {
                background: #eaf2ff;
                color: #174ea6;
                border: 1px solid #9bbcf1;
                border-radius: 4px;
                padding: 4px 8px;
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
        """)

    def _build_actions(self):
        file_menu = self.menuBar().addMenu("&File")
        self.open_action = file_menu.addAction("Open")
        self.add_action = file_menu.addAction("Add")
        self.reload_action = file_menu.addAction("Reload")
        self.scan_action = file_menu.addAction(QtGui.QIcon(_resource_path("icons", "scan.png")), "Scan folder")
        self.export_table_action = file_menu.addAction("Export selected table")
        self.export_plot_action = file_menu.addAction(
            QtGui.QIcon(_resource_path("icons", "filesave.svg")), "Export plot"
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
        self.standardize_si_action = view_menu.addAction("Standardize units to SI")
        self.standardize_si_action.triggered.connect(self.standardize_units_si)
        view_export_plot_action = view_menu.addAction("Export plot")
        view_export_plot_action.triggered.connect(self.export_plot_image)

    def _connect(self):
        self.plot_type_combo.currentIndexChanged.connect(self.on_selection_changed)
        self.mode_combo.currentIndexChanged.connect(self.on_selection_changed)
        self.compare_combo.currentIndexChanged.connect(self.on_compare_mode_changed)
        self.grid_check.stateChanged.connect(self.on_selection_changed)
        self.logx_check.stateChanged.connect(self.on_selection_changed)
        self.logy_check.stateChanged.connect(self.on_selection_changed)
        self.legend_check.stateChanged.connect(self.on_selection_changed)
        self.line_width_spin.valueChanged.connect(self.on_selection_changed)
        self.marker_combo.currentIndexChanged.connect(self.on_selection_changed)
        self.load_workers_combo.currentIndexChanged.connect(self.update_lazy_worker_limit)
        self.canvas.curveSelected.connect(self.on_curve_selected)
        self.plot_button.clicked.connect(self.redraw)
        self.clear_button.clicked.connect(self.clear)
        self.select_all_y_button.clicked.connect(self.select_all_y)
        self.select_none_y_button.clicked.connect(self.select_none_y)
        self.load_selected_button.clicked.connect(self.load_selected_lazy_files)

    def on_compare_mode_changed(self):
        self.set_compare_pane_count(self.compare_pane_count())
        self.populate_tables()
        QtCore.QTimer.singleShot(0, self.resize_compare_region)
        self.on_selection_changed()

    def update_lazy_worker_limit(self):
        text = self.load_workers_combo.currentText()
        if text == "Auto":
            self.lazy_max_workers = _default_lazy_workers()
        else:
            self.lazy_max_workers = max(1, min(max(1, os.cpu_count() or 1), int(text)))
        self.statusBar().showMessage("Parallel file load workers: {}".format(self.lazy_max_workers), 8000)
        self.start_next_lazy_load()

    def set_loading_controls_enabled(self, enabled):
        for action in (
            self.open_action,
            self.add_action,
            self.reload_action,
            self.scan_action,
            self.autorange_action,
            self.standardize_si_action,
            self.export_table_action,
            self.export_plot_action,
        ):
            action.setEnabled(enabled)
        for widget in (
            self.plot_type_combo,
            self.mode_combo,
            self.compare_combo,
            self.live_plot,
            self.grid_check,
            self.logx_check,
            self.logy_check,
            self.legend_check,
            self.line_width_spin,
            self.marker_combo,
            self.load_workers_combo,
            self.plot_button,
            self.clear_button,
            self.select_all_y_button,
            self.select_none_y_button,
            self.load_selected_button,
        ):
            widget.setEnabled(enabled)
        for pane in self.selector_panes:
            pane.frame.setEnabled(enabled)

    def begin_lazy_load_batch(self, total):
        if total <= 0:
            return
        if self.lazy_batch_total == 0:
            self.lazy_batch_done = 0
            self.lazy_batch_total = total
        else:
            self.lazy_batch_total += total
        self.loading_progress.setRange(0, self.lazy_batch_total)
        self.loading_progress.setValue(self.lazy_batch_done)
        self.loading_progress.setFormat("Loading %v/%m")
        self.loading_progress.setVisible(True)
        self.set_loading_controls_enabled(False)

    def advance_lazy_load_progress(self):
        if self.lazy_batch_total <= 0:
            return
        self.lazy_batch_done = min(self.lazy_batch_done + 1, self.lazy_batch_total)
        self.loading_progress.setValue(self.lazy_batch_done)
        self.loading_progress.setFormat("Loading {}/{}".format(self.lazy_batch_done, self.lazy_batch_total))

    def finish_lazy_load_batch_if_done(self):
        if self.lazy_batch_total <= 0:
            return
        if self.lazy_load_queue or self.lazy_loader_threads:
            return
        self.loading_progress.setValue(self.lazy_batch_total)
        self.loading_progress.setFormat("Loaded {}/{}".format(self.lazy_batch_done, self.lazy_batch_total))
        self.loading_progress.setVisible(False)
        self.lazy_batch_total = 0
        self.lazy_batch_done = 0
        self.set_loading_controls_enabled(True)

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
        dialog = ScanDialog(self.file_formats, self, settings=self.settings)
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

    def pending_lazy_indices(self, lazy_indices):
        pending = []
        for lazy_index in lazy_indices:
            entry = self.lazy_entries[lazy_index]
            if entry.loaded or entry.loading or entry.attempted or lazy_index in self.lazy_load_queue:
                continue
            pending.append(lazy_index)
        return pending

    def queue_lazy_load(self, lazy_index):
        entry = self.lazy_entries[lazy_index]
        if entry.loaded or entry.loading or entry.attempted or lazy_index in self.lazy_load_queue:
            return
        if self.lazy_batch_total == 0:
            self.begin_lazy_load_batch(1)
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
        self.advance_lazy_load_progress()
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
        self.finish_lazy_load_batch_if_done()

    def on_lazy_thread_finished(self, lazy_index):
        self.lazy_loader_threads.pop(lazy_index, None)
        self.lazy_loader_workers.pop(lazy_index, None)
        self.status_label.setText(
            "{:,} files indexed, {:,} loaded, {:,} active".format(
                len(self.lazy_entries), self.lazy_loaded_count(), len(self.lazy_loader_threads)
            )
        )
        self.start_next_lazy_load()
        self.finish_lazy_load_batch_if_done()

    def is_lazy_selected(self, lazy_index):
        for pane in self.visible_selector_panes():
            for item in pane.table_list_widget.selectedItems():
                data = item.data(QtCore.Qt.UserRole)
                if isinstance(data, tuple) and data == ("lazy", lazy_index):
                    return True
        return False

    def update_lazy_item(self, lazy_index):
        for pane in self.selector_panes:
            for row in range(pane.table_list_widget.count()):
                item = pane.table_list_widget.item(row)
                data = item.data(QtCore.Qt.UserRole)
                if isinstance(data, tuple) and data == ("lazy", lazy_index):
                    item.setText(self.lazy_item_text(self.lazy_entries[lazy_index]))

    def load_selected_lazy_files(self):
        lazy_indices = self.selected_lazy_indices()
        if not lazy_indices:
            return
        self.begin_lazy_load_batch(len(self.pending_lazy_indices(lazy_indices)))
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
        self.status_label.setText("{} tables loaded".format(len(self.tab_list)))
        self.redraw()

    def reload_files(self):
        if self.lazy_entries:
            self.lazy_generation += 1
            self.lazy_load_queue = []
            self.lazy_warning_backlog = []
            self.lazy_batch_total = 0
            self.lazy_batch_done = 0
            self.loading_progress.setVisible(False)
            self.set_loading_controls_enabled(True)
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
        visible = self.visible_selector_panes()
        names = self.tab_list.getDisplayTabNames() if not self.lazy_entries else []
        for pane_index, pane in enumerate(visible):
            pane.table_list_widget.blockSignals(True)
            pane.table_list_widget.clear()
            if self.lazy_entries:
                for i, entry in enumerate(self.lazy_entries):
                    item = QtWidgets.QListWidgetItem(self.lazy_item_text(entry))
                    item.setData(QtCore.Qt.UserRole, ("lazy", i))
                    pane.table_list_widget.addItem(item)
            else:
                for i, tab in enumerate(self.tab_list):
                    item = QtWidgets.QListWidgetItem("{}  ({})".format(names[i], tab.shapestring))
                    item.setData(QtCore.Qt.UserRole, ("table", i))
                    pane.table_list_widget.addItem(item)
            if pane.table_list_widget.count() > 0:
                default_row = min(pane_index, pane.table_list_widget.count() - 1)
                pane.table_list_widget.item(default_row).setSelected(True)
            pane.table_list_widget.blockSignals(False)
        self.on_table_selection_changed()

    def selected_lazy_indices(self, pane=None):
        panes = [pane] if pane is not None else self.visible_selector_panes()
        indices = []
        for p in panes:
            for item in p.table_list_widget.selectedItems():
                data = item.data(QtCore.Qt.UserRole)
                if isinstance(data, tuple) and data[0] == "lazy" and data[1] not in indices:
                    indices.append(data[1])
        return indices

    def selected_table_indices(self, load=True, show_warning=False, pane=None):
        panes = [pane] if pane is not None else self.visible_selector_panes()
        indices = []
        for p in panes:
            if not p.bladed_dataset_combo.isHidden():
                table_index = p.bladed_dataset_combo.currentData()
                if isinstance(table_index, int) and 0 <= table_index < len(self.tab_list):
                    if table_index not in indices:
                        indices.append(table_index)
                    continue
            for item in p.table_list_widget.selectedItems():
                data = item.data(QtCore.Qt.UserRole)
                if isinstance(data, tuple) and data[0] == "table":
                    if data[1] not in indices:
                        indices.append(data[1])
                elif isinstance(data, tuple) and data[0] == "lazy":
                    entry = self.lazy_entries[data[1]]
                    if entry.loaded:
                        for table_index in entry.table_indices:
                            if table_index not in indices:
                                indices.append(table_index)
                    elif load:
                        for table_index in self.ensure_lazy_loaded(data[1], show_warning=show_warning):
                            if table_index not in indices:
                                indices.append(table_index)
        return indices

    def on_table_selection_changed(self):
        for pane in self.visible_selector_panes():
            self.populate_bladed_datasets(pane)
            self.populate_columns(pane)
        self.update_table_preview()
        self.update_file_info()
        self.on_selection_changed()

    def populate_bladed_datasets(self, pane):
        previous_table_index = pane.bladed_dataset_combo.currentData()
        table_indices = []
        selected_items = pane.table_list_widget.selectedItems()
        if len(selected_items) == 1:
            data = selected_items[0].data(QtCore.Qt.UserRole)
            if isinstance(data, tuple) and data[0] == "lazy":
                entry = self.lazy_entries[data[1]]
                is_bladed = getattr(entry.file_format, "name", "") == "Bladed output file"
                is_project = os.path.splitext(entry.path)[1].lower() == ".$pj"
                if is_bladed and is_project and entry.loaded:
                    table_indices = list(entry.table_indices)
            elif isinstance(data, tuple) and data[0] == "table":
                table_index = data[1]
                tab = self.tab_list[table_index]
                if os.path.splitext(tab.filename)[1].lower() == ".$pj":
                    table_indices = [
                        i for i, candidate in enumerate(self.tab_list)
                        if candidate.filename == tab.filename
                    ]

        pane.bladed_dataset_combo.blockSignals(True)
        pane.bladed_dataset_combo.clear()
        for table_index in table_indices:
            tab = self.tab_list[table_index]
            label = "{}  ({})".format(tab.nickname, tab.shapestring)
            pane.bladed_dataset_combo.addItem(label, table_index)
        if table_indices:
            selected_index = (
                previous_table_index
                if previous_table_index in table_indices
                else table_indices[0]
            )
            pane.bladed_dataset_combo.setCurrentIndex(table_indices.index(selected_index))
        visible = bool(table_indices)
        pane.bladed_dataset_label.setVisible(visible)
        pane.bladed_dataset_combo.setVisible(visible)
        pane.bladed_dataset_combo.blockSignals(False)

    def on_bladed_dataset_changed(self, pane):
        if pane.bladed_dataset_combo.isHidden():
            return
        pane.y_list_widget.clearSelection()
        self.populate_columns(pane)
        self.update_table_preview()
        self.update_file_info()
        self.on_selection_changed()

    def populate_columns(self, pane=None):
        pane = pane or self.selector_panes[0]
        previous_x = pane.x_combo.currentData()
        previous_y = set(self.selected_y_indices_original(pane))
        lazy_indices = self.selected_lazy_indices(pane)
        indices = []
        columns = []
        if lazy_indices:
            lazy_index = lazy_indices[0]
            entry = self.lazy_entries[lazy_index]
            if entry.loaded:
                indices = self.selected_table_indices(load=False, pane=pane)
            else:
                self.ensure_lazy_header(lazy_index)
                columns = list(entry.columns)
        if not lazy_indices:
            indices = self.selected_table_indices(load=False, pane=pane)
        if not indices and len(self.tab_list) > 0 and not self.lazy_entries:
            indices = [0]
        if indices and not columns:
            columns = list(self.tab_list[indices[0]].columns)
        all_columns = [(i, str(col)) for i, col in enumerate(columns)]
        text_filter = pane.column_filter.text().strip().lower()
        visible_y = [(i, col) for i, col in all_columns
                     if not text_filter or text_filter in col.lower()]

        pane.x_combo.blockSignals(True)
        pane.y_list_widget.blockSignals(True)
        pane.x_combo.clear()
        pane.y_list_widget.clear()
        for original_i, col in all_columns:
            pane.x_combo.addItem(col, original_i)
        for original_i, col in visible_y:
            item = QtWidgets.QListWidgetItem(col)
            item.setData(QtCore.Qt.UserRole, original_i)
            pane.y_list_widget.addItem(item)

        if all_columns:
            all_indices = [i for i, _ in all_columns]
            if previous_x in all_indices:
                x_to_select = previous_x
            else:
                x_to_select = next((i for i, col in all_columns if col.lower().startswith("time")), all_columns[0][0])
            pane.x_combo.setCurrentIndex(all_indices.index(x_to_select))
        if visible_y and not previous_y:
            x_current = pane.x_combo.currentData()
            default_row = next((row for row, (i, _) in enumerate(visible_y) if i != x_current), 0)
            pane.y_list_widget.item(default_row).setSelected(True)
        else:
            for row in range(pane.y_list_widget.count()):
                item = pane.y_list_widget.item(row)
                if item.data(QtCore.Qt.UserRole) in previous_y:
                    item.setSelected(True)
        pane.x_combo.blockSignals(False)
        pane.y_list_widget.blockSignals(False)

    def on_selection_changed(self):
        if self.live_plot.isChecked() and not self.has_unloaded_lazy_selection():
            self.redraw()

    def has_unloaded_lazy_selection(self):
        for lazy_index in self.selected_lazy_indices():
            if not self.lazy_entries[lazy_index].loaded:
                return True
        return False

    def select_all_y(self):
        for pane in self.visible_selector_panes():
            pane.y_list_widget.blockSignals(True)
            for row in range(pane.y_list_widget.count()):
                pane.y_list_widget.item(row).setSelected(True)
            pane.y_list_widget.blockSignals(False)
        self.on_selection_changed()

    def select_none_y(self):
        for pane in self.visible_selector_panes():
            pane.y_list_widget.blockSignals(True)
            for row in range(pane.y_list_widget.count()):
                pane.y_list_widget.item(row).setSelected(False)
            pane.y_list_widget.blockSignals(False)
        self.on_selection_changed()

    def selected_y_indices(self, pane=None):
        return self.selected_y_indices_original(pane)

    def selected_y_indices_original(self, pane=None):
        pane = pane or self.selector_panes[0]
        return [item.data(QtCore.Qt.UserRole) for item in pane.y_list_widget.selectedItems()]

    def build_plot_data(self):
        plot_data = []
        pane_payloads = []
        total_table_count = 0
        for pane_index, pane in enumerate(self.visible_selector_panes()):
            table_indices = self.selected_table_indices(pane=pane)
            y_indices = self.selected_y_indices(pane)
            ix = pane.x_combo.currentData()
            if ix is None or not y_indices or not table_indices:
                continue
            pane_payloads.append((pane_index, table_indices, y_indices, ix))
            total_table_count += len(table_indices)

        same_col = total_table_count > 1 or len(pane_payloads) > 1
        for pane_index, table_indices, y_indices, ix in pane_payloads:
            for it in table_indices:
                tab = self.tab_list[it]
                if ix >= len(tab.columns):
                    continue
                for iy in y_indices:
                    if iy >= len(tab.columns):
                        continue
                    idx = (it, ix, iy, str(tab.columns[ix]), str(tab.columns[iy]), tab.active_name)
                    pd = PlotData()
                    pd.fromIDs(self.tab_list, len(plot_data), idx, same_col, pipeline=None)
                    pd.pane_index = pane_index
                    self.apply_plot_type(pd)
                    if same_col:
                        pd.syl = "Set {}: {} - {}".format(pane_index + 1, pd.st, pd.sy)
                    else:
                        pd.syl = pd.sy
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
                self.begin_lazy_load_batch(len(self.pending_lazy_indices(self.selected_lazy_indices())))
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

    def on_curve_selected(self, meta):
        self.highlight_curve_table(meta)
        message = "Selected: {label} | file/table: {file} | y: {y} | x: {x} | {points:,} points".format(
            label=meta.get("label", ""),
            file=meta.get("file", ""),
            y=meta.get("y", ""),
            x=meta.get("x", ""),
            points=meta.get("points", 0),
        )
        self.statusBar().showMessage(message)

    def highlight_curve_table(self, meta):
        table_index = meta.get("table_index")
        if table_index is None:
            return
        pane_index = meta.get("pane_index", 0)
        panes = self.visible_selector_panes()
        if not panes:
            return
        pane = panes[pane_index] if isinstance(pane_index, int) and pane_index < len(panes) else panes[0]
        target_row = None
        for row in range(pane.table_list_widget.count()):
            item = pane.table_list_widget.item(row)
            data = item.data(QtCore.Qt.UserRole)
            if isinstance(data, tuple) and data == ("table", table_index):
                target_row = row
                break
            if isinstance(data, tuple) and data[0] == "lazy":
                entry = self.lazy_entries[data[1]]
                if table_index in entry.table_indices:
                    target_row = row
                    break
        if target_row is None:
            return
        pane.table_list_widget.blockSignals(True)
        pane.table_list_widget.clearSelection()
        item = pane.table_list_widget.item(target_row)
        item.setSelected(True)
        pane.table_list_widget.setCurrentItem(item)
        pane.table_list_widget.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
        pane.table_list_widget.blockSignals(False)
        self.update_table_preview()
        self.update_file_info()

    def standardize_units_si(self):
        indices = self.selected_table_indices(load=False)
        if not indices:
            indices = list(range(len(self.tab_list)))
        if not indices:
            self.statusBar().showMessage("No loaded tables to standardize", 8000)
            return

        changed = 0
        for it in indices:
            tab = self.tab_list[it]
            before = list(tab.data.columns)
            tab.changeUnits(data={"flavor": "SI"})
            after = list(tab.data.columns)
            if before != after:
                changed += 1
                print("[pyDatView] Standardized units to SI: {}".format(tab.active_name))

        self.populate_columns()
        self.update_table_preview()
        self.update_file_info()
        if self.live_plot.isChecked() and not self.has_unloaded_lazy_selection():
            self.redraw()
        self.statusBar().showMessage(
            "Standardized units to SI for {:,} loaded table(s), {:,} changed".format(len(indices), changed),
            12000,
        )

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
            self.statusBar().showMessage("Create a plot before exporting", 5000)
            return
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export plot",
            "pydatview_plot.png",
            "PNG files (*.png);;PDF files (*.pdf);;All files (*)",
        )
        if not path:
            return
        path_lower = path.lower()
        if "." not in os.path.basename(path):
            if selected_filter.startswith("PDF"):
                path += ".pdf"
                path_lower = path.lower()
            else:
                path += ".png"
                path_lower = path.lower()
        elif not path_lower.endswith((".png", ".pdf")):
            path += ".png"
            path_lower = path.lower()
        try:
            if path_lower.endswith(".pdf"):
                from PySide6 import QtPrintSupport
                try:
                    printer_mode = QtPrintSupport.QPrinter.PrinterMode.HighResolution
                    pdf_format = QtPrintSupport.QPrinter.OutputFormat.PdfFormat
                except AttributeError:
                    printer_mode = QtPrintSupport.QPrinter.HighResolution
                    pdf_format = QtPrintSupport.QPrinter.PdfFormat
                printer = QtPrintSupport.QPrinter(printer_mode)
                printer.setOutputFormat(pdf_format)
                printer.setOutputFileName(path)
                painter = QtGui.QPainter(printer)
                paint_rect = printer.pageLayout().paintRectPixels(printer.resolution())
                self.canvas.scene().render(
                    painter,
                    QtCore.QRectF(paint_rect),
                    self.canvas.scene().sceneRect(),
                    QtCore.Qt.KeepAspectRatio,
                )
                painter.end()
            else:
                from pyqtgraph.exporters import ImageExporter
                exporter = ImageExporter(self.canvas.scene())
                exporter.export(path)
            self.statusBar().showMessage("Plot exported to {}".format(path), 8000)
        except Exception as exc:
            self.show_exception("Failed to export plot", exc)

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
