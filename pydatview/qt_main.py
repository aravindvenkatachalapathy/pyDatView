"""PySide6/PyQtGraph pyDatView application.

This is the primary GUI path. It reuses pyDatView's existing IO, TableList,
and PlotData data model with Qt widgets and PyQtGraph.
"""

import ast
import os
import re
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from pydatview.Tables import TableList
from pydatview.common import no_unit
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
    loaded_column_indices: set = field(default_factory=set)
    full_loaded: bool = False
    estimated_load_bytes: int = 0

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
    display_columns: list = field(default_factory=list)
    bladed_project_mode: bool = False


class LazyLoadWorker(QtCore.QObject):
    finished = QtCore.Signal(int, int, object, str, float, str, object)

    def __init__(
            self,
            generation,
            lazy_index,
            path,
            file_format,
            options,
            channel_indices=None):
        super().__init__()
        self.generation = generation
        self.lazy_index = lazy_index
        self.path = path
        self.file_format = file_format
        self.options = dict(options)
        self.channel_indices = channel_indices

    @QtCore.Slot()
    def run(self):
        t0 = time.perf_counter()
        try:
            loader = TableList(options=self.options)
            selective = (
                self.channel_indices is not None
                and getattr(self.file_format, "name", "") == "FAST output file"
            )
            tabs, warning = loader._load_file_tabs(
                self.path,
                fileformat=self.file_format,
                bReload=False,
                channel_indices=self.channel_indices if selective else None,
            )
            loaded_column_indices = (
                list(self.channel_indices) if selective else None
            )
        except Exception as exc:
            tabs = []
            loaded_column_indices = self.channel_indices
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
            loaded_column_indices,
        )


def _resource_path(*parts):
    source_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "ressources", *parts)
    )
    if os.path.exists(source_path):
        return source_path
    return os.path.join(sys.prefix, "ressources", *parts)


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
        if getattr(fmt, "name", "") == "Bladed output file":
            if bladed_suffixes:
                for suffix in bladed_suffixes:
                    suffix_formats.setdefault(".$" + suffix, fmt)
            else:
                suffix_formats.setdefault(".$pj", fmt)
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


def _equivalent_loads(time_values, signal_values, slopes):
    """Return 1 Hz DEL values while performing rainflow counting only once."""
    from pydatview.tools.fatigue import find_range_count

    slopes = tuple(int(slope) for slope in slopes)
    if not slopes:
        return {}
    try:
        time_values, signal_values = _finite_xy(time_values, signal_values)
        if len(time_values) < 2:
            raise ValueError("Not enough finite samples")
        duration = float(time_values[-1] - time_values[0])
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Time duration must be positive")
        cycles, ranges, _ = find_range_count(
            signal_values,
            bins=100,
            method="rainflow_windap",
            meanBin=True,
            binStartAt0=False,
        )
        cycles = np.asarray(cycles, dtype=float)
        ranges = np.asarray(ranges, dtype=float)
        if cycles.ndim == 0 or ranges.ndim == 0:
            raise ValueError("Rainflow counting failed")
        values = {}
        with np.errstate(over="ignore", invalid="ignore"):
            for slope in slopes:
                damage = np.sum(np.power(ranges, slope) * cycles / duration)
                values[slope] = float(np.power(damage, 1.0 / slope))
        return values
    except Exception:
        return {slope: np.nan for slope in slopes}


def _plot_ready_xy(x, y, logx=False, logy=False):
    x, y = _finite_xy(x, y)
    if not logx and not logy:
        return x, y
    valid = np.ones(len(x), dtype=bool)
    if logx:
        valid &= x > 0
    if logy:
        valid &= y > 0
    if valid.all():
        return x, y
    return x[valid], y[valid]


_MATH_FUNCTIONS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
    "where": np.where,
    "gradient": np.gradient,
    "degrees": np.degrees,
    "radians": np.radians,
    "mean": np.mean,
    "std": np.std,
}
_MATH_CONSTANTS = {"pi": np.pi, "e": np.e}
_MATH_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Compare,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitAnd,
    ast.BitOr,
    ast.UAdd,
    ast.USub,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def _column_array(dataframe, column):
    series = dataframe[column]
    try:
        return series.to_numpy(copy=False)
    except TypeError:
        return series.to_numpy()
    except AttributeError:
        return np.asarray(series)


def _resolve_expression_column(dataframe, token):
    token = token.strip()
    columns = [str(column) for column in dataframe.columns]
    exact_matches = [column for column in columns if column == token]
    if len(exact_matches) == 1:
        return exact_matches[0]
    if len(exact_matches) > 1:
        raise ValueError("Column name is ambiguous: {}".format(token))
    matches = [column for column in columns if no_unit(column).strip() == token]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError("Column not found: {}".format(token))
    raise ValueError("Column name is ambiguous: {}".format(token))


def evaluate_math_expression(dataframe, expression):
    expression = expression.strip()
    if not expression:
        raise ValueError("Expression is empty")

    namespace = dict(_MATH_FUNCTIONS)
    namespace.update(_MATH_CONSTANTS)
    columns = [str(column) for column in dataframe.columns]

    identifier_columns = {}
    for column in columns:
        for candidate in (column, no_unit(column).strip()):
            if candidate.isidentifier() and candidate not in namespace:
                identifier_columns.setdefault(candidate, []).append(column)
    for identifier, matches in identifier_columns.items():
        unique_matches = list(dict.fromkeys(matches))
        if len(unique_matches) == 1:
            namespace[identifier] = _column_array(dataframe, unique_matches[0])

    token_index = 0

    def replace_column(match):
        nonlocal token_index
        column = _resolve_expression_column(dataframe, match.group(1))
        variable = "_column_{}".format(token_index)
        token_index += 1
        namespace[variable] = _column_array(dataframe, column)
        return variable

    prepared = re.sub(r"\{([^{}]+)\}", replace_column, expression)
    for name in tuple(_MATH_FUNCTIONS) + tuple(_MATH_CONSTANTS):
        prepared = re.sub(r"\bnp\.{}\b".format(re.escape(name)), name, prepared)

    try:
        tree = ast.parse(prepared, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid expression syntax: {}".format(exc.msg)) from exc

    for node in ast.walk(tree):
        if not isinstance(node, _MATH_AST_NODES):
            raise ValueError("Unsupported expression element: {}".format(type(node).__name__))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _MATH_FUNCTIONS:
                raise ValueError("Unsupported function")
            if node.keywords:
                raise ValueError("Function keyword arguments are not supported")
        if isinstance(node, ast.Name) and node.id not in namespace:
            raise ValueError("Unknown variable or function: {}".format(node.id))
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool)):
            raise ValueError("Only numeric constants are supported")

    with np.errstate(all="ignore"):
        result = eval(compile(tree, "<calculation>", "eval"), {"__builtins__": {}}, namespace)
    result = np.asarray(result)
    if result.ndim == 0:
        result = np.full(len(dataframe), result.item())
    if result.ndim != 1:
        raise ValueError("Result must be a one-dimensional variable")
    if len(result) != len(dataframe):
        raise ValueError(
            "Result has {:,} values; the table has {:,} rows".format(len(result), len(dataframe))
        )
    if result.dtype.kind not in "biuf":
        raise ValueError("Result must contain numeric values")
    return result


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
        if self.logMode:
            return self.logTickStrings(values, scale, spacing)
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

    def logTickStrings(self, values, scale, spacing):
        if not np.isfinite(scale) or scale <= 0:
            return [""] * len(values)
        scale_exponent = np.log10(scale)
        labels = []
        for value in values:
            exponent = float(value) + scale_exponent
            rounded = int(round(exponent))
            labels.append(
                "10^{}".format(rounded)
                if abs(exponent - rounded) < 1e-8
                else ""
            )
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


class CalculationDialog(QtWidgets.QDialog):
    def __init__(self, columns, selected_columns=None, parent=None):
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
        self.result_name = QtWidgets.QLineEdit()
        self.result_name.setText("Calculated")
        expression_layout.addRow("Result name", self.result_name)
        self.expression = QtWidgets.QPlainTextEdit()
        self.expression.setMaximumHeight(110)
        expression_layout.addRow("Expression", self.expression)
        self.function_combo = QtWidgets.QComboBox()
        self.function_combo.addItem("Insert function")
        self.function_combo.addItems(list(_MATH_FUNCTIONS))
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

        self.column_filter.textChanged.connect(lambda _text: self.populate_columns())
        self.column_list.itemDoubleClicked.connect(self.insert_column)
        self.function_combo.activated.connect(self.insert_function)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.populate_columns()

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
                  marker=None, step=False, axis_limits=None):
        # QGraphicsView's OpenGL viewport can crash on Windows when log transforms
        # discard points. Keep accelerated rendering for regular plots.
        self.useOpenGL(not (logx or logy))
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
                    x, y = _plot_ready_xy(pd.x, pd.y, logx=logx, logy=logy)
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
                    skipFiniteCheck=not (logx or logy),
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
            self._apply_axis_limits(
                plot,
                axis_limits or {},
                logx=logx,
                logy=logy,
            )

    @staticmethod
    def _limited_range(current_range, minimum, maximum, logarithmic, axis):
        def transform(value):
            if value is None:
                return None
            if logarithmic:
                if value <= 0:
                    raise ValueError(
                        "{} limits must be positive in logarithmic mode".format(axis)
                    )
                return float(np.log10(value))
            return float(value)

        minimum = transform(minimum)
        maximum = transform(maximum)
        if minimum is not None and maximum is not None and minimum >= maximum:
            raise ValueError("{} minimum must be less than {} maximum".format(axis, axis))

        lower = current_range[0] if minimum is None else minimum
        upper = current_range[1] if maximum is None else maximum
        if lower >= upper:
            if minimum is not None and maximum is None:
                upper = lower + max(abs(lower) * 0.05, 1.0)
            elif maximum is not None and minimum is None:
                lower = upper - max(abs(upper) * 0.05, 1.0)
        return lower, upper

    @classmethod
    def _apply_axis_limits(cls, plot, limits, logx=False, logy=False):
        x_values = (limits.get("xmin"), limits.get("xmax"))
        y_values = (limits.get("ymin"), limits.get("ymax"))
        if not any(value is not None for value in x_values + y_values):
            return
        plot.autoRange()
        x_current, y_current = plot.getViewBox().viewRange()
        if any(value is not None for value in x_values):
            x_range = cls._limited_range(x_current, *x_values, logx, "X")
            plot.setXRange(*x_range, padding=0)
        if any(value is not None for value in y_values):
            y_range = cls._limited_range(y_current, *y_values, logy, "Y")
            plot.setYRange(*y_range, padding=0)

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
        self.axis_limits_button = QtWidgets.QPushButton("Limits")
        self.axis_limits_button.setToolTip("Set X and Y plot limits")
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
        stats_layout.addLayout(stats_controls)
        self.stats_table = QtWidgets.QTableWidget()
        self.stats_table.setAlternatingRowColors(True)
        self.stats_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.stats_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.stats_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.stats_table.verticalHeader().setVisible(False)
        stats_layout.addWidget(self.stats_table, 1)
        self.update_del_slopes_button()
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
        bladed_dataset_label = QtWidgets.QLabel("BLADED VARIABLE GROUP")
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
        table_list_widget.itemSelectionChanged.connect(
            lambda p=pane: self.on_table_selection_changed(p)
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
        self.axis_limits_action = view_menu.addAction("Axis limits")
        self.axis_limits_action.triggered.connect(self.open_axis_limits_dialog)
        view_export_plot_action = view_menu.addAction("Export plot")
        view_export_plot_action.triggered.connect(self.export_plot_image)

        tools_menu = self.menuBar().addMenu("&Tools")
        units_menu = tools_menu.addMenu("Standardize units")
        self.standardize_we_action = units_menu.addAction(
            "Wind Energy / OpenFAST units"
        )
        self.standardize_we_action.triggered.connect(self.standardize_units_we)
        self.standardize_si_action = units_menu.addAction("SI units")
        self.standardize_si_action.triggered.connect(self.standardize_units_si)
        tools_menu.addSeparator()
        self.math_action = tools_menu.addAction("Mathematical operation")
        self.math_action.triggered.connect(self.open_calculation_dialog)

    def _connect(self):
        self.plot_type_combo.currentIndexChanged.connect(self.on_plot_type_changed)
        self.mode_combo.currentIndexChanged.connect(self.on_selection_changed)
        self.compare_combo.currentIndexChanged.connect(self.on_compare_mode_changed)
        self.grid_check.stateChanged.connect(self.on_selection_changed)
        self.logx_check.stateChanged.connect(self.on_selection_changed)
        self.logy_check.stateChanged.connect(self.on_selection_changed)
        self.legend_check.stateChanged.connect(self.on_selection_changed)
        self.line_width_spin.valueChanged.connect(self.on_selection_changed)
        self.marker_combo.currentIndexChanged.connect(self.on_selection_changed)
        self.axis_limits_button.clicked.connect(self.open_axis_limits_dialog)
        self.load_workers_combo.currentIndexChanged.connect(self.update_lazy_worker_limit)
        self.canvas.curveSelected.connect(self.on_curve_selected)
        self.plot_button.clicked.connect(self.redraw)
        self.clear_button.clicked.connect(self.clear)
        self.select_all_y_button.clicked.connect(self.select_all_y)
        self.select_none_y_button.clicked.connect(self.select_none_y)
        self.load_selected_button.clicked.connect(self.load_selected_lazy_files)
        self.math_button.clicked.connect(self.open_calculation_dialog)
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
        self.fft_options_panel.setVisible(is_fft)
        self.update_fft_control_states()
        self.on_selection_changed()

    def update_fft_control_states(self):
        averaging = self.fft_averaging_combo.currentText()
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

    def update_lazy_worker_limit(self):
        text = self.load_workers_combo.currentText()
        if text == "Auto":
            self.lazy_max_workers = _default_lazy_workers()
        else:
            self.lazy_max_workers = max(1, min(max(1, os.cpu_count() or 1), int(text)))
        self.statusBar().showMessage(
            "Parallel workers: {} overall, {} for Bladed".format(
                self.lazy_max_workers,
                min(self.lazy_max_workers, self.bladed_worker_cap),
            ),
            8000,
        )
        self.start_next_lazy_load()

    def set_loading_controls_enabled(self, enabled):
        for action in (
            self.open_action,
            self.add_action,
            self.reload_action,
            self.scan_action,
            self.autorange_action,
            self.axis_limits_action,
            self.standardize_we_action,
            self.standardize_si_action,
            self.export_table_action,
            self.export_plot_action,
            self.math_action,
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
            self.axis_limits_button,
            self.load_workers_combo,
            self.plot_button,
            self.clear_button,
            self.select_all_y_button,
            self.select_none_y_button,
            self.load_selected_button,
            self.math_button,
            self.fft_options_panel,
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
        self.lazy_last_ui_update = 0.0
        self.set_loading_controls_enabled(False)

    def advance_lazy_load_progress(self):
        if self.lazy_batch_total <= 0:
            return False
        self.lazy_batch_done = min(self.lazy_batch_done + 1, self.lazy_batch_total)
        now = time.perf_counter()
        refresh = (
            self.lazy_batch_done >= self.lazy_batch_total
            or now - self.lazy_last_ui_update >= 0.1
        )
        if refresh:
            self.lazy_last_ui_update = now
            self.loading_progress.setValue(self.lazy_batch_done)
            self.loading_progress.setFormat(
                "Loading {}/{}".format(self.lazy_batch_done, self.lazy_batch_total)
            )
        return refresh

    def finish_lazy_load_batch_if_done(self):
        if self.lazy_batch_total <= 0:
            return
        if self.lazy_load_queue or self.lazy_loader_threads:
            return
        self.loading_progress.setValue(self.lazy_batch_total)
        self.loading_progress.setFormat("Loaded {}/{}".format(self.lazy_batch_done, self.lazy_batch_total))
        self.loading_progress.setVisible(False)
        self.status_label.setText(
            "{:,} files indexed, {:,} loaded, 0 active".format(
                len(self.lazy_entries), self.lazy_loaded_count()
            )
        )
        self.lazy_batch_total = 0
        self.lazy_batch_done = 0
        self.lazy_last_ui_update = 0.0
        self.flush_lazy_selection_refresh()
        self.lazy_selected_batch = set()
        self.set_loading_controls_enabled(True)
        if self.lazy_warning_backlog:
            warning_count = len(self.lazy_warning_backlog)
            first_warning = self.lazy_warning_backlog[0].splitlines()[0]
            self.lazy_warning_backlog = []
            self.statusBar().showMessage(
                "{} load warning(s): {}".format(
                    warning_count,
                    first_warning,
                ),
                20000,
            )

    def flush_lazy_selection_refresh(self):
        needs_plot = self.plot_after_lazy_load
        self.plot_after_lazy_load = False
        if self.lazy_selection_refresh_pending:
            self.lazy_selection_refresh_pending = False
            self.on_table_selection_changed()
            if needs_plot and not self.live_plot.isChecked():
                self.redraw()
        elif needs_plot:
            self.redraw()

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

        added = self.set_lazy_file_index(
            matches,
            append=dialog.keep_existing(),
        )
        self.statusBar().showMessage(
            "Indexed {:,} new files in {:.3f}s; {:,} total, {:,} loaded".format(
                added,
                scan_seconds,
                len(self.lazy_entries),
                self.lazy_loaded_count(),
            ),
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
                self.lazy_load_queue = deque()
                self.lazy_warning_backlog = []
                self.lazy_memory_reservations = {}
                self.lazy_entries = []
                self.lazy_item_widgets = {}
                self.lazy_loaded_total = 0
                self.lazy_selected_batch = set()
                self.lazy_selection_refresh_pending = False
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

    @staticmethod
    def normalized_file_path(path):
        return os.path.normcase(os.path.abspath(path))

    def selected_lazy_paths_by_pane(self):
        selected = []
        for pane in self.visible_selector_panes():
            selected.append({
                self.normalized_file_path(
                    self.lazy_entries[data[1]].path
                )
                for item in pane.table_list_widget.selectedItems()
                for data in [item.data(QtCore.Qt.UserRole)]
                if isinstance(data, tuple) and data[0] == "lazy"
            })
        return selected

    def set_lazy_file_index(self, matches, append=False):
        if append and self.lazy_entries:
            selected_paths = self.selected_lazy_paths_by_pane()
            known_paths = {
                self.normalized_file_path(entry.path)
                for entry in self.lazy_entries
            }
            added = 0
            for path, fmt in matches:
                normalized = self.normalized_file_path(path)
                if normalized in known_paths:
                    continue
                try:
                    stat = os.stat(path)
                    size = stat.st_size
                    mtime = stat.st_mtime
                except OSError:
                    size = 0
                    mtime = 0.0
                self.lazy_entries.append(
                    LazyFileEntry(
                        path=path,
                        file_format=fmt,
                        size=size,
                        mtime=mtime,
                    )
                )
                known_paths.add(normalized)
                added += 1
            self.current_files = [entry.path for entry in self.lazy_entries]
            self.populate_tables(selected_lazy_paths=selected_paths)
            self.status_label.setText(
                "{:,} files indexed, {:,} loaded".format(
                    len(self.lazy_entries),
                    self.lazy_loaded_count(),
                )
            )
            return added

        self.lazy_generation += 1
        self.lazy_load_queue = deque()
        self.lazy_warning_backlog = []
        self.lazy_memory_reservations = {}
        self.lazy_item_widgets = {}
        self.lazy_loaded_total = 0
        self.lazy_selected_batch = set()
        self.lazy_selection_refresh_pending = False
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
        return len(self.lazy_entries)

    def lazy_loaded_count(self):
        return self.lazy_loaded_total

    def lazy_item_text(self, entry):
        if entry.full_loaded:
            state = "loaded"
        elif entry.loaded:
            total = len(entry.columns) if entry.columns else "?"
            state = "partial {}/{}".format(
                len(entry.loaded_column_indices),
                total,
            )
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
        if entry.columns or entry.header_attempted:
            return
        entry.header_attempted = True
        try:
            entry.columns = read_lazy_columns(entry.path, entry.file_format)
        except Exception as exc:
            entry.warning = "Header read failed: {}: {}".format(type(exc).__name__, exc)

    def lazy_request_satisfied(self, entry, channel_indices):
        if entry.full_loaded:
            return True
        if channel_indices is None:
            return False
        if not channel_indices:
            return True
        return entry.loaded and set(channel_indices).issubset(
            entry.loaded_column_indices
        )

    def is_lazy_queued(self, lazy_index):
        return any(item[0] == lazy_index for item in self.lazy_load_queue)

    @staticmethod
    def is_bladed_entry(entry):
        return getattr(entry.file_format, "name", "") == "Bladed output file"

    @staticmethod
    def available_memory_bytes():
        try:
            import psutil
            return int(psutil.virtual_memory().available)
        except Exception:
            return None

    def estimate_lazy_load_bytes(self, entry):
        if entry.estimated_load_bytes > 0:
            return entry.estimated_load_bytes
        source_bytes = max(0, int(entry.size))
        if self.is_bladed_entry(entry) and self.is_bladed_project_path(entry.path):
            directory = os.path.dirname(os.path.abspath(entry.path))
            directory_key = os.path.normcase(directory)
            if directory_key not in self._directory_file_sizes:
                files = []
                try:
                    with os.scandir(directory) as entries:
                        for candidate in entries:
                            try:
                                if candidate.is_file(follow_symlinks=False):
                                    files.append((candidate.name.lower(), candidate.stat().st_size))
                            except OSError:
                                continue
                except OSError:
                    pass
                self._directory_file_sizes[directory_key] = files
            project_root = os.path.splitext(os.path.basename(entry.path))[0].lower()
            binary_prefix = project_root + ".$"
            source_bytes = sum(
                size for name, size in self._directory_file_sizes[directory_key]
                if name.startswith(binary_prefix)
            ) or source_bytes
        # Dataframes, index columns, and decoder scratch space add overhead.
        entry.estimated_load_bytes = max(
            64 * 1024 * 1024,
            int(source_bytes * 2.0),
        )
        return entry.estimated_load_bytes

    def effective_lazy_worker_limit(self):
        has_bladed = any(
            self.is_bladed_entry(self.lazy_entries[index])
            for index in self.lazy_loader_threads
            if index < len(self.lazy_entries)
        ) or any(
            index < len(self.lazy_entries) and self.is_bladed_entry(self.lazy_entries[index])
            for index, _channels in self.lazy_load_queue
        )
        if has_bladed:
            return min(self.lazy_max_workers, self.bladed_worker_cap)
        return self.lazy_max_workers

    def lazy_memory_allows_start(self, entry):
        available = self.available_memory_bytes()
        if available is None:
            return True, ""
        required = self.estimate_lazy_load_bytes(entry)
        reserved = sum(self.lazy_memory_reservations.values())
        reserve_floor = max(1024 ** 3, int(available * 0.10))
        if available - reserved - required >= reserve_floor:
            return True, ""
        return False, (
            "Not enough available memory to load {} safely: estimated {:.2f} GB "
            "required with {:.2f} GB available. Reduce the selection, unload data, "
            "or lower the worker count."
        ).format(
            entry.basename,
            required / 1024 ** 3,
            max(0, available - reserved) / 1024 ** 3,
        )

    def reject_lazy_load(self, lazy_index, warning):
        entry = self.lazy_entries[lazy_index]
        entry.loading = False
        entry.attempted = not entry.loaded
        entry.warning = warning
        self.lazy_warning_backlog.append(warning)
        self.advance_lazy_load_progress()
        self.update_lazy_item(lazy_index)
        print("[pyDatView] {}".format(warning))

    def ensure_lazy_loaded(
            self,
            lazy_index,
            show_warning=True,
            channel_indices=None):
        entry = self.lazy_entries[lazy_index]
        if self.lazy_request_satisfied(entry, channel_indices):
            return entry.table_indices
        if entry.attempted and not entry.loaded:
            if entry.warning and show_warning:
                QtWidgets.QMessageBox.warning(self, "Load warning", entry.warning)
            return []
        self.queue_lazy_load(lazy_index, channel_indices=channel_indices)
        return []

    def pending_lazy_indices(self, lazy_indices, column_requests=None):
        pending = []
        for lazy_index in lazy_indices:
            entry = self.lazy_entries[lazy_index]
            request = (
                column_requests.get(lazy_index)
                if column_requests is not None
                else None
            )
            if self.lazy_request_satisfied(entry, request):
                continue
            if (
                entry.loading
                or (entry.attempted and not entry.loaded)
                or self.is_lazy_queued(lazy_index)
            ):
                continue
            pending.append(lazy_index)
        return pending

    def queue_lazy_load(self, lazy_index, channel_indices=None):
        entry = self.lazy_entries[lazy_index]
        if self.lazy_request_satisfied(entry, channel_indices):
            return
        if (
            entry.loading
            or (entry.attempted and not entry.loaded)
            or self.is_lazy_queued(lazy_index)
        ):
            return
        if channel_indices is not None:
            channel_indices = tuple(sorted(
                entry.loaded_column_indices.union(channel_indices)
            ))
        if self.lazy_batch_total == 0:
            self.begin_lazy_load_batch(1)
        entry.loading = True
        self.lazy_load_queue.append((lazy_index, channel_indices))
        if self.lazy_batch_total <= 1:
            self.status_label.setText("Loading {}".format(entry.basename))
            self.statusBar().showMessage("Queued {}".format(entry.path))
            self.update_lazy_item(lazy_index)
        self.start_next_lazy_load()

    def start_next_lazy_load(self):
        while self.lazy_load_queue:
            if len(self.lazy_loader_threads) >= self.effective_lazy_worker_limit():
                break
            lazy_index, _channel_indices = self.lazy_load_queue[0]
            if lazy_index >= len(self.lazy_entries):
                self.lazy_load_queue.popleft()
                continue
            allowed, warning = self.lazy_memory_allows_start(
                self.lazy_entries[lazy_index]
            )
            if not allowed:
                if self.lazy_loader_threads:
                    self.statusBar().showMessage(
                        "Waiting for memory before loading {}".format(
                            self.lazy_entries[lazy_index].basename
                        )
                    )
                    break
                self.lazy_load_queue.popleft()
                self.reject_lazy_load(lazy_index, warning)
                continue
            self.start_one_lazy_load()

    def start_one_lazy_load(self):
        if not self.lazy_load_queue:
            return
        lazy_index, channel_indices = self.lazy_load_queue.popleft()
        if lazy_index >= len(self.lazy_entries):
            self.start_next_lazy_load()
            return
        entry = self.lazy_entries[lazy_index]
        self.lazy_memory_reservations[lazy_index] = self.estimate_lazy_load_bytes(entry)
        if self.lazy_batch_total <= 1:
            self.status_label.setText("Loading {}".format(entry.basename))
            self.statusBar().showMessage("Loading {}".format(entry.path))

        generation = self.lazy_generation
        thread = QtCore.QThread(self)
        worker = LazyLoadWorker(
            generation,
            lazy_index,
            entry.path,
            entry.file_format,
            self.tab_list.options,
            channel_indices=channel_indices,
        )
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

    def on_lazy_load_finished(
            self,
            generation,
            lazy_index,
            tabs,
            warning,
            elapsed,
            format_name,
            loaded_column_indices):
        if generation != self.lazy_generation:
            return
        if lazy_index >= len(self.lazy_entries):
            return
        entry = self.lazy_entries[lazy_index]
        was_loaded = entry.loaded
        if tabs:
            if was_loaded and len(entry.table_indices) == len(tabs):
                for table_index, tab in zip(entry.table_indices, tabs):
                    self.tab_list._tabs[table_index] = tab
            else:
                start = len(self.tab_list)
                self.tab_list.append(tabs)
                entry.table_indices = list(range(start, start + len(tabs)))
                if not was_loaded:
                    self.lazy_loaded_total += 1
            if loaded_column_indices is None:
                entry.full_loaded = True
                entry.loaded_column_indices = set(range(len(entry.columns)))
            else:
                entry.loaded_column_indices = set(loaded_column_indices)
        entry.warning = warning or ""
        entry.attempted = not tabs and not entry.loaded
        entry.loading = False
        refresh_ui = self.advance_lazy_load_progress()
        self.update_lazy_item(lazy_index)
        if refresh_ui:
            self.status_label.setText(
                "{:,} files indexed, {:,} loaded, {:,} active".format(
                    len(self.lazy_entries), self.lazy_loaded_count(), len(self.lazy_loader_threads)
                )
            )
        n_rows = sum(getattr(tab, "nRows", 0) for tab in tabs) if tabs else 0
        n_cols = sum(getattr(tab, "nCols", 0) for tab in tabs) if tabs else 0
        if refresh_ui:
            self.statusBar().showMessage(
                "Loaded {} in {:.3f}s ({}, {:,} rows, {:,} cols)".format(
                    entry.basename, elapsed, format_name, n_rows, n_cols
                ),
                12000,
            )
        if entry.warning:
            self.lazy_warning_backlog.append(entry.warning)
        if self.is_lazy_selected(lazy_index):
            self.lazy_selection_refresh_pending = True
        if not self.has_unloaded_lazy_selection():
            self.flush_lazy_selection_refresh()
        self.finish_lazy_load_batch_if_done()

    def on_lazy_thread_finished(self, lazy_index):
        self.lazy_loader_threads.pop(lazy_index, None)
        self.lazy_loader_workers.pop(lazy_index, None)
        self.lazy_memory_reservations.pop(lazy_index, None)
        self.start_next_lazy_load()
        self.finish_lazy_load_batch_if_done()

    def is_lazy_selected(self, lazy_index):
        if self.lazy_selected_batch:
            return lazy_index in self.lazy_selected_batch
        for pane in self.visible_selector_panes():
            for item in pane.table_list_widget.selectedItems():
                data = item.data(QtCore.Qt.UserRole)
                if isinstance(data, tuple) and data == ("lazy", lazy_index):
                    return True
        return False

    def update_lazy_item(self, lazy_index):
        text = self.lazy_item_text(self.lazy_entries[lazy_index])
        for item in self.lazy_item_widgets.get(lazy_index, ()):
            item.setText(text)

    def load_selected_lazy_files(self):
        lazy_indices = self.selected_lazy_indices()
        if not lazy_indices:
            return
        pending = self.pending_lazy_indices(lazy_indices)
        self.lazy_selected_batch = set(lazy_indices)
        self.begin_lazy_load_batch(len(pending))
        if pending:
            self.statusBar().showMessage(
                "Queueing {:,} selected files".format(len(pending))
            )
        for lazy_index in lazy_indices:
            self.ensure_lazy_loaded(
                lazy_index,
                show_warning=False,
                channel_indices=None,
            )
        if not pending:
            self.lazy_selected_batch = set()
        self.finish_lazy_load_batch_if_done()
        self.on_table_selection_changed()

    def load_dfs(self, dataframes, names=None):
        if not isinstance(dataframes, list):
            dataframes = [dataframes]
        if names is None:
            names = ["df{}".format(i + 1) for i in range(len(dataframes))]
        if not isinstance(names, list):
            names = [names]
        self.lazy_generation += 1
        self.lazy_load_queue = deque()
        self.lazy_warning_backlog = []
        self.lazy_memory_reservations = {}
        self.lazy_entries = []
        self.lazy_item_widgets = {}
        self.lazy_loaded_total = 0
        self.lazy_selected_batch = set()
        self.lazy_selection_refresh_pending = False
        self.tab_list.from_dataframes(dataframes=dataframes, names=names, bAdd=False)
        self.populate_tables()
        self.status_label.setText("{} tables loaded".format(len(self.tab_list)))
        self.redraw()

    def reload_files(self):
        if self.lazy_entries:
            self.lazy_generation += 1
            self.lazy_load_queue = deque()
            self.lazy_warning_backlog = []
            self.lazy_memory_reservations = {}
            self.lazy_batch_total = 0
            self.lazy_batch_done = 0
            self.loading_progress.setVisible(False)
            self.set_loading_controls_enabled(True)
            self.lazy_loaded_total = 0
            self.lazy_selected_batch = set()
            self.lazy_selection_refresh_pending = False
            for entry in self.lazy_entries:
                entry.table_indices = []
                entry.warning = ""
                entry.attempted = False
                entry.loading = False
                entry.columns = []
                entry.header_attempted = False
                entry.loaded_column_indices = set()
                entry.full_loaded = False
            self.tab_list.clean()
            self.populate_tables()
            self.clear()
            self.status_label.setText("{:,} files indexed, 0 loaded".format(len(self.lazy_entries)))
            return
        filenames = sorted(set(f for f in self.current_files if f))
        if filenames:
            self.load_files(filenames, add=False)

    def populate_tables(self, selected_lazy_paths=None):
        visible = self.visible_selector_panes()
        names = self.tab_list.getDisplayTabNames() if not self.lazy_entries else []
        self.lazy_item_widgets = {}
        for pane_index, pane in enumerate(visible):
            pane.table_list_widget.blockSignals(True)
            pane.table_list_widget.clear()
            if self.lazy_entries:
                for i, entry in enumerate(self.lazy_entries):
                    item = QtWidgets.QListWidgetItem(self.lazy_item_text(entry))
                    item.setData(QtCore.Qt.UserRole, ("lazy", i))
                    pane.table_list_widget.addItem(item)
                    self.lazy_item_widgets.setdefault(i, []).append(item)
            else:
                displayed_projects = set()
                for i, tab in enumerate(self.tab_list):
                    if self.is_bladed_project_path(tab.filename):
                        project_path = os.path.abspath(tab.filename)
                        if project_path in displayed_projects:
                            continue
                        displayed_projects.add(project_path)
                        group_count = sum(
                            1 for candidate in self.tab_list
                            if os.path.abspath(candidate.filename) == project_path
                        )
                        item = QtWidgets.QListWidgetItem(os.path.basename(tab.filename))
                        item.setToolTip("{} Bladed variable groups".format(group_count))
                        item.setData(
                            QtCore.Qt.UserRole,
                            ("bladed_project", project_path),
                        )
                        pane.table_list_widget.addItem(item)
                        continue
                    item = QtWidgets.QListWidgetItem("{}  ({})".format(names[i], tab.shapestring))
                    item.setData(QtCore.Qt.UserRole, ("table", i))
                    pane.table_list_widget.addItem(item)
            restored_selection = False
            if self.lazy_entries and selected_lazy_paths is not None:
                paths = (
                    selected_lazy_paths[pane_index]
                    if pane_index < len(selected_lazy_paths)
                    else set()
                )
                for row in range(pane.table_list_widget.count()):
                    item = pane.table_list_widget.item(row)
                    data = item.data(QtCore.Qt.UserRole)
                    if (
                        isinstance(data, tuple)
                        and data[0] == "lazy"
                        and self.normalized_file_path(
                            self.lazy_entries[data[1]].path
                        ) in paths
                    ):
                        item.setSelected(True)
                        restored_selection = True
            if pane.table_list_widget.count() > 0 and not restored_selection:
                default_row = min(pane_index, pane.table_list_widget.count() - 1)
                pane.table_list_widget.item(default_row).setSelected(True)
            pane.table_list_widget.blockSignals(False)
        self.on_table_selection_changed()

    def selected_lazy_indices(self, pane=None):
        panes = [pane] if pane is not None else self.visible_selector_panes()
        indices = []
        seen = set()
        for p in panes:
            for item in p.table_list_widget.selectedItems():
                data = item.data(QtCore.Qt.UserRole)
                if isinstance(data, tuple) and data[0] == "lazy" and data[1] not in seen:
                    indices.append(data[1])
                    seen.add(data[1])
        return indices

    @staticmethod
    def is_bladed_project_path(path):
        return bool(path) and os.path.splitext(path)[1].lower() == ".$pj"

    def selected_bladed_project_paths(self, pane):
        paths = []
        seen = set()
        for item in pane.table_list_widget.selectedItems():
            data = item.data(QtCore.Qt.UserRole)
            path = None
            if isinstance(data, tuple) and data[0] == "bladed_project":
                path = data[1]
            elif isinstance(data, tuple) and data[0] == "lazy":
                entry = self.lazy_entries[data[1]]
                if entry.loaded and self.is_bladed_project_path(entry.path):
                    path = entry.path
            elif isinstance(data, tuple) and data[0] == "table":
                tab = self.tab_list[data[1]]
                if self.is_bladed_project_path(tab.filename):
                    path = tab.filename
            if path:
                normalized = os.path.abspath(path)
                if normalized not in seen:
                    paths.append(normalized)
                    seen.add(normalized)
        return paths

    def selected_bladed_group(self, pane):
        if pane.bladed_dataset_combo.isHidden():
            return "__all__"
        return pane.bladed_dataset_combo.currentData() or "__all__"

    def bladed_project_table_indices(self, pane, group=None):
        paths = set(self.selected_bladed_project_paths(pane))
        if not paths:
            return []
        group = self.selected_bladed_group(pane) if group is None else group
        return [
            i for i, tab in enumerate(self.tab_list)
            if os.path.abspath(tab.filename) in paths
            and (group == "__all__" or tab.nickname == group)
        ]

    def selected_table_indices(self, load=True, show_warning=False, pane=None):
        panes = [pane] if pane is not None else self.visible_selector_panes()
        indices = []
        seen = set()
        for p in panes:
            project_paths = set(self.selected_bladed_project_paths(p))
            for table_index in self.bladed_project_table_indices(p):
                if table_index not in seen:
                    indices.append(table_index)
                    seen.add(table_index)
            for item in p.table_list_widget.selectedItems():
                data = item.data(QtCore.Qt.UserRole)
                if isinstance(data, tuple) and data[0] == "table":
                    if data[1] not in seen:
                        indices.append(data[1])
                        seen.add(data[1])
                elif isinstance(data, tuple) and data[0] == "lazy":
                    entry = self.lazy_entries[data[1]]
                    if os.path.abspath(entry.path) in project_paths:
                        continue
                    if entry.loaded:
                        for table_index in entry.table_indices:
                            if table_index not in seen:
                                indices.append(table_index)
                                seen.add(table_index)
                    elif load:
                        for table_index in self.ensure_lazy_loaded(data[1], show_warning=show_warning):
                            if table_index not in seen:
                                indices.append(table_index)
                                seen.add(table_index)
        return indices

    def on_table_selection_changed(self, active_pane=None):
        if active_pane in self.visible_selector_panes():
            self.active_selector_pane = active_pane
        for pane in self.visible_selector_panes():
            self.populate_bladed_datasets(pane)
            self.populate_columns(pane)
        self.update_table_preview()
        self.update_file_info()
        self.on_selection_changed()

    def populate_bladed_datasets(self, pane):
        previous_group = pane.bladed_dataset_combo.currentData()
        table_indices = self.bladed_project_table_indices(pane, group="__all__")
        groups = []
        for table_index in table_indices:
            group = self.tab_list[table_index].nickname
            if group not in groups:
                groups.append(group)

        pane.bladed_dataset_combo.blockSignals(True)
        pane.bladed_dataset_combo.clear()
        if groups:
            pane.bladed_dataset_combo.addItem("All variable groups", "__all__")
            for group in groups:
                pane.bladed_dataset_combo.addItem(group, group)
            selected_group = previous_group if previous_group in groups else "__all__"
            pane.bladed_dataset_combo.setCurrentIndex(
                pane.bladed_dataset_combo.findData(selected_group)
            )
        visible = bool(groups)
        pane.bladed_dataset_label.setVisible(visible)
        pane.bladed_dataset_combo.setVisible(visible)
        pane.bladed_dataset_combo.blockSignals(False)

    def on_bladed_dataset_changed(self, pane):
        if pane.bladed_dataset_combo.isHidden():
            return
        self.active_selector_pane = pane
        pane.y_list_widget.clearSelection()
        self.populate_columns(pane)
        self.update_table_preview()
        self.update_file_info()
        self.on_selection_changed()

    def populate_columns(self, pane=None):
        pane = pane or self.selector_panes[0]
        previous_x_name = pane.x_combo.currentText()
        previous_y_names = {
            item.text() for item in pane.y_list_widget.selectedItems()
        }
        lazy_indices = self.selected_lazy_indices(pane)
        indices = []
        columns = []
        project_indices = self.bladed_project_table_indices(pane)
        pane.bladed_project_mode = bool(project_indices)
        if project_indices:
            for table_index in project_indices:
                for column in self.tab_list[table_index].columns:
                    column = str(column)
                    if column not in columns:
                        columns.append(column)
        if lazy_indices:
            lazy_index = lazy_indices[0]
            entry = self.lazy_entries[lazy_index]
            self.ensure_lazy_header(lazy_index)
            if entry.columns:
                columns = list(entry.columns)
            elif entry.loaded and not project_indices:
                indices = self.selected_table_indices(load=False, pane=pane)
        if not lazy_indices and not project_indices:
            indices = self.selected_table_indices(load=False, pane=pane)
        if not indices and len(self.tab_list) > 0 and not self.lazy_entries:
            indices = [0]
        if indices and not columns:
            columns = list(self.tab_list[indices[0]].columns)
        pane.display_columns = list(columns)
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
            all_names = [col for _, col in all_columns]
            if previous_x_name in all_names:
                x_to_select = all_columns[all_names.index(previous_x_name)][0]
            else:
                x_to_select = next((i for i, col in all_columns if col.lower().startswith("time")), all_columns[0][0])
            pane.x_combo.setCurrentIndex(
                next(
                    row for row in range(pane.x_combo.count())
                    if pane.x_combo.itemData(row) == x_to_select
                )
            )
        if visible_y and not previous_y_names:
            x_current = pane.x_combo.currentData()
            default_row = next((row for row, (i, _) in enumerate(visible_y) if i != x_current), 0)
            pane.y_list_widget.item(default_row).setSelected(True)
        else:
            for row in range(pane.y_list_widget.count()):
                item = pane.y_list_widget.item(row)
                if item.text() in previous_y_names:
                    item.setSelected(True)
        pane.x_combo.blockSignals(False)
        pane.y_list_widget.blockSignals(False)

    def on_selection_changed(self):
        if self.live_plot.isChecked():
            self.redraw_timer.start()

    def has_unloaded_lazy_selection(self):
        for lazy_index, request in self.lazy_plot_column_requests().items():
            entry = self.lazy_entries[lazy_index]
            if (
                not self.lazy_request_satisfied(entry, request)
                and not (entry.attempted and not entry.loaded)
            ):
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

    def lazy_plot_column_requests(self):
        requests = {}
        for pane in self.visible_selector_panes():
            lazy_indices = self.selected_lazy_indices(pane)
            if not lazy_indices:
                continue
            ix = pane.x_combo.currentData()
            y_indices = self.selected_y_indices_original(pane)
            if ix is None or not y_indices:
                continue

            reference = self.lazy_entries[lazy_indices[0]]
            self.ensure_lazy_header(lazy_indices[0])
            if not reference.columns:
                for lazy_index in lazy_indices:
                    requests[lazy_index] = None
                continue

            requested_names = []
            for column_index in [ix] + y_indices:
                if 0 <= column_index < len(reference.columns):
                    name = reference.columns[column_index]
                    if name not in requested_names:
                        requested_names.append(name)

            for lazy_index in lazy_indices:
                if requests.get(lazy_index) is None and lazy_index in requests:
                    continue
                entry = self.lazy_entries[lazy_index]
                self.ensure_lazy_header(lazy_index)
                if not entry.columns:
                    requests[lazy_index] = None
                    continue
                mapped = []
                for name in requested_names:
                    try:
                        mapped.append(entry.columns.index(name))
                    except ValueError:
                        mapped = []
                        break
                previous = set(requests.get(lazy_index, ()))
                requests[lazy_index] = tuple(sorted(previous.union(mapped)))
        return requests

    def build_plot_data(self):
        plot_data = []
        pane_payloads = []
        total_table_count = 0
        for pane_index, pane in enumerate(self.visible_selector_panes()):
            table_sources = []
            lazy_indices = self.selected_lazy_indices(pane)
            project_indices = self.bladed_project_table_indices(pane)
            if project_indices:
                table_sources = [(table_index, None) for table_index in project_indices]
            elif lazy_indices:
                for lazy_index in lazy_indices:
                    entry = self.lazy_entries[lazy_index]
                    for table_index in entry.table_indices:
                        table_sources.append((table_index, entry))
            else:
                table_sources = [
                    (table_index, None)
                    for table_index in self.selected_table_indices(
                        load=False,
                        pane=pane,
                    )
                ]
            y_indices = self.selected_y_indices(pane)
            ix = pane.x_combo.currentData()
            if ix is None or not y_indices or not table_sources:
                continue
            pane_payloads.append((
                pane_index,
                table_sources,
                y_indices,
                ix,
                pane.bladed_project_mode,
                list(pane.display_columns),
            ))
            if pane.bladed_project_mode:
                total_table_count += len({
                    os.path.abspath(self.tab_list[table_index].filename)
                    for table_index, _entry in table_sources
                })
            else:
                total_table_count += len(table_sources)

        same_col = total_table_count > 1 or len(pane_payloads) > 1
        for pane_index, table_sources, y_indices, ix, project_mode, display_columns in pane_payloads:
            seen_project_curves = set()
            for it, entry in table_sources:
                tab = self.tab_list[it]
                tab_columns = [str(column) for column in tab.columns]
                if project_mode:
                    if ix >= len(display_columns):
                        continue
                    x_name = display_columns[ix]
                    try:
                        actual_ix = tab_columns.index(x_name)
                    except ValueError:
                        continue
                elif entry is not None and entry.columns:
                    if ix >= len(entry.columns):
                        continue
                    x_name = entry.columns[ix]
                    try:
                        actual_ix = tab_columns.index(x_name)
                    except ValueError:
                        continue
                else:
                    actual_ix = ix
                if actual_ix >= len(tab.columns):
                    continue
                for iy in y_indices:
                    if project_mode:
                        if iy >= len(display_columns):
                            continue
                        y_name = display_columns[iy]
                        try:
                            actual_iy = tab_columns.index(y_name)
                        except ValueError:
                            continue
                        curve_key = (os.path.abspath(tab.filename), y_name)
                        if curve_key in seen_project_curves:
                            continue
                        seen_project_curves.add(curve_key)
                    elif entry is not None and entry.columns:
                        if iy >= len(entry.columns):
                            continue
                        y_name = entry.columns[iy]
                        try:
                            actual_iy = tab_columns.index(y_name)
                        except ValueError:
                            continue
                    else:
                        actual_iy = iy
                    if actual_iy >= len(tab.columns):
                        continue
                    idx = (
                        it,
                        actual_ix,
                        actual_iy,
                        str(tab.columns[actual_ix]),
                        str(tab.columns[actual_iy]),
                        tab.active_name,
                    )
                    pd = PlotData()
                    pd.fromIDs(self.tab_list, len(plot_data), idx, same_col, pipeline=None)
                    pd.pane_index = pane_index
                    if project_mode:
                        pd.st = os.path.basename(tab.filename)
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
            pd.toFFT(
                yType=self.fft_output_combo.currentText(),
                xType=self.fft_x_combo.currentData(),
                avgMethod=self.fft_averaging_combo.currentText(),
                avgWindow=self.fft_window_combo.currentText(),
                bDetrend=self.fft_detrend_check.isChecked(),
                nExp=self.fft_nexp_spin.value(),
                nPerDecade=self.fft_bins_spin.value(),
            )
        elif plot_type == "MinMax":
            pd.toMinMax(xScale=False, yScale=True, yCenter="None")

    def redraw(self):
        try:
            if self.redraw_timer.isActive():
                self.redraw_timer.stop()
            column_requests = self.lazy_plot_column_requests()
            missing = [
                lazy_index
                for lazy_index, request in column_requests.items()
                if not self.lazy_request_satisfied(
                    self.lazy_entries[lazy_index],
                    request,
                )
                and not (
                    self.lazy_entries[lazy_index].attempted
                    and not self.lazy_entries[lazy_index].loaded
                )
            ]
            if missing:
                pending = self.pending_lazy_indices(
                    missing,
                    column_requests=column_requests,
                )
                self.lazy_selected_batch = set(missing)
                self.plot_after_lazy_load = True
                self.begin_lazy_load_batch(len(pending))
                for lazy_index in pending:
                    self.ensure_lazy_loaded(
                        lazy_index,
                        show_warning=False,
                        channel_indices=column_requests[lazy_index],
                    )
                self.statusBar().showMessage(
                    "Loading selected X/Y variables from {:,} files ...".format(
                        len(missing)
                    ),
                    8000,
                )
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
                axis_limits=self.axis_limits,
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
            if (
                isinstance(data, tuple)
                and data[0] == "bladed_project"
                and os.path.abspath(self.tab_list[table_index].filename) == data[1]
            ):
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

    def open_calculation_dialog(self):
        panes = self.visible_selector_panes()
        if not panes:
            return
        pane = self.active_selector_pane if self.active_selector_pane in panes else panes[0]
        unloaded = [
            lazy_index for lazy_index in self.selected_lazy_indices(pane)
            if not self.lazy_entries[lazy_index].full_loaded
        ]
        if unloaded:
            self.statusBar().showMessage(
                "Use Load full selected before creating a calculated variable",
                10000,
            )
            return

        table_indices = self.selected_table_indices(load=False, pane=pane)
        if len(table_indices) != 1:
            QtWidgets.QMessageBox.information(
                self,
                "Mathematical operation",
                "Select one loaded table or one Bladed variable group.",
            )
            return

        table_index = table_indices[0]
        tab = self.tab_list[table_index]
        selected_columns = [
            str(tab.columns[index])
            for index in self.selected_y_indices_original(pane)
            if isinstance(index, int) and 0 <= index < len(tab.columns)
        ]
        dialog = CalculationDialog(
            tab.columns,
            selected_columns=selected_columns,
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return

        result_name, expression = dialog.values()
        if result_name in [str(column) for column in tab.data.columns]:
            QtWidgets.QMessageBox.warning(
                self,
                "Mathematical operation",
                "A variable named '{}' already exists.".format(result_name),
            )
            return

        try:
            result = evaluate_math_expression(tab.data, expression)
            tab.addColumn(
                result_name,
                result,
                i=len(tab.data.columns) - 1,
                sFormula=expression,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Mathematical operation",
                "{}: {}".format(type(exc).__name__, exc),
            )
            return

        new_column_index = len(tab.data.columns) - 1
        pane.column_filter.blockSignals(True)
        pane.column_filter.clear()
        pane.column_filter.blockSignals(False)
        self.populate_columns(pane)
        pane.y_list_widget.blockSignals(True)
        pane.y_list_widget.clearSelection()
        for row in range(pane.y_list_widget.count()):
            item = pane.y_list_widget.item(row)
            if item.data(QtCore.Qt.UserRole) == new_column_index:
                item.setSelected(True)
                pane.y_list_widget.setCurrentItem(item)
                pane.y_list_widget.scrollToItem(item)
                break
        pane.y_list_widget.blockSignals(False)
        self.update_table_preview()
        self.detail_tabs.setCurrentWidget(self.table_view)
        self.redraw()
        self.statusBar().showMessage(
            "Added calculated variable '{}' to {}".format(result_name, tab.nickname),
            10000,
        )

    def standardize_units(self, flavor, label):
        partial = [
            lazy_index for lazy_index in self.selected_lazy_indices()
            if not self.lazy_entries[lazy_index].full_loaded
        ]
        if partial:
            self.statusBar().showMessage(
                "Use Load full selected before standardizing table units",
                10000,
            )
            return
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
            tab.changeUnits(data={"flavor": flavor})
            after = list(tab.data.columns)
            if before != after:
                changed += 1
                print(
                    "[pyDatView] Standardized units to {}: {}".format(
                        label, tab.active_name
                    )
                )

        for pane in self.visible_selector_panes():
            self.populate_columns(pane)
        self.update_table_preview()
        self.update_file_info()
        if self.live_plot.isChecked() and not self.has_unloaded_lazy_selection():
            self.redraw()
        self.statusBar().showMessage(
            "Standardized units to {} for {:,} loaded table(s), {:,} changed".format(
                label, len(indices), changed
            ),
            12000,
        )

    def standardize_units_we(self):
        self.standardize_units("WE", "Wind Energy / OpenFAST")

    def standardize_units_si(self):
        self.standardize_units("SI", "SI")

    def clear(self):
        self.canvas.clear_plot()
        self.plot_data = []

    def open_axis_limits_dialog(self):
        dialog = AxisLimitsDialog(
            self.axis_limits,
            logx=self.logx_check.isChecked(),
            logy=self.logy_check.isChecked(),
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        self.axis_limits = dialog.values()
        self.update_axis_limits_button()
        self.redraw()

    def update_axis_limits_button(self):
        active = any(value is not None for value in self.axis_limits.values())
        self.axis_limits_button.setProperty("limitsActive", active)
        values = []
        for label, minimum_key, maximum_key in (
            ("X", "xmin", "xmax"),
            ("Y", "ymin", "ymax"),
        ):
            minimum = self.axis_limits.get(minimum_key)
            maximum = self.axis_limits.get(maximum_key)
            if minimum is not None or maximum is not None:
                values.append(
                    "{} [{}, {}]".format(
                        label,
                        "auto" if minimum is None else "{:.6g}".format(minimum),
                        "auto" if maximum is None else "{:.6g}".format(maximum),
                    )
                )
        self.axis_limits_button.setToolTip(
            "Set X and Y plot limits" if not values else "; ".join(values)
        )
        self.axis_limits_button.style().unpolish(self.axis_limits_button)
        self.axis_limits_button.style().polish(self.axis_limits_button)

    def auto_range(self):
        self.axis_limits = {key: None for key in ("xmin", "xmax", "ymin", "ymax")}
        self.update_axis_limits_button()
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
                if entry.full_loaded:
                    status = "loaded"
                elif entry.loaded:
                    status = "partial ({}/{} variables)".format(
                        len(entry.loaded_column_indices),
                        len(entry.columns) if entry.columns else "?",
                    )
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

    def selected_del_slopes(self):
        return [
            slope for slope, action in self.del_slope_actions.items()
            if action.isChecked()
        ]

    def update_del_slopes_button(self):
        slopes = self.selected_del_slopes()
        self.del_slopes_button.setText(
            "m = {}".format(", ".join(map(str, slopes))) if slopes else "None"
        )
        self.del_slopes_button.setToolTip(
            "Select one or more Wöhler slopes for 1 Hz damage-equivalent loads"
        )

    def on_del_slopes_changed(self, _checked=False):
        slopes = self.selected_del_slopes()
        self.settings.setValue("stats/del_slopes", [str(slope) for slope in slopes])
        self.update_del_slopes_button()
        self.update_stats()

    @staticmethod
    def _stats_table_item(value, numeric=False):
        if numeric:
            if isinstance(value, (int, np.integer)):
                text = "{:,}".format(int(value))
            else:
                try:
                    value = float(value)
                    text = "{:.6g}".format(value) if np.isfinite(value) else "N/A"
                except (TypeError, ValueError):
                    text = "N/A"
        else:
            text = str(value)
        item = QtWidgets.QTableWidgetItem(text)
        if numeric:
            item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        return item

    def update_stats(self):
        slopes = self.selected_del_slopes()
        headers = ["Series", "File", "n", "Min", "Mean", "Max", "Std"]
        headers.extend("DEL m={} (1 Hz)".format(slope) for slope in slopes)
        self.stats_table.setSortingEnabled(False)
        self.stats_table.clear()
        self.stats_table.setColumnCount(len(headers))
        self.stats_table.setHorizontalHeaderLabels(headers)
        if not self.plot_data:
            self.stats_table.setRowCount(0)
            return

        rows = []
        for pd in self.plot_data:
            try:
                x_raw, y_raw = _finite_xy(pd.x0, pd.y0)
            except Exception:
                continue
            if len(y_raw) == 0:
                continue
            del_values = _equivalent_loads(x_raw, y_raw, slopes)
            rows.append([
                pd.syl or pd.sy,
                os.path.basename(getattr(pd, "filename", "") or getattr(pd, "st", "")),
                len(y_raw),
                np.min(y_raw),
                np.mean(y_raw),
                np.max(y_raw),
                np.std(y_raw),
                *[del_values[slope] for slope in slopes],
            ])

        self.stats_table.setRowCount(len(rows))
        for row_index, row_values in enumerate(rows):
            for column_index, value in enumerate(row_values):
                item = self._stats_table_item(value, numeric=column_index >= 2)
                if column_index == 1:
                    item.setToolTip(str(value))
                self.stats_table.setItem(row_index, column_index, item)
        header = self.stats_table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        if headers:
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        if len(headers) > 1:
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.Interactive)
            header.resizeSection(1, 150)
        self.stats_table.resizeRowsToContents()

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
        partial = [
            lazy_index for lazy_index in self.selected_lazy_indices()
            if not self.lazy_entries[lazy_index].full_loaded
        ]
        if partial:
            self.statusBar().showMessage(
                "Use Load full selected before exporting a complete table",
                10000,
            )
            return
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
