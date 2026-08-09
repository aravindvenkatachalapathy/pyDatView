"""File indexing and background loading primitives for the Qt GUI."""

import os
import re
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from pydatview.Tables import TableList
from pydatview.qt_compat import QtCore

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
    dataset_mode: str = None


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
