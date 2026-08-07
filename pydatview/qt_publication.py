"""Publication-quality Matplotlib export for plots displayed with PyQtGraph."""

import os
import shutil
from dataclasses import dataclass

import numpy as np

from pydatview.plotdata import PDL_xlabel
from pydatview.qt_compat import QtCore, QtWidgets
from pydatview.qt_plot import _PLOT_PALETTE
from pydatview.qt_stats import _plot_ready_xy


_EXPORT_FORMATS = (
    ("PDF - vector", "pdf"),
    ("SVG - vector", "svg"),
    ("PGF - LaTeX", "pgf"),
    ("PNG - raster", "png"),
    ("TIFF - raster", "tiff"),
)
_RASTER_FORMATS = {"png", "tif", "tiff"}
_MARKERS = {None: None, "o": "o", "s": "s", "t": "^", "d": "D"}


@dataclass(frozen=True)
class PublicationExportOptions:
    path: str
    width: float = 7.2
    height: float = 4.5
    dpi: int = 600
    font_family: str = "DejaVu Serif"
    font_size: float = 9.0
    line_width: float = 1.25
    max_points: int = 50000
    grid: bool = False
    legend: bool = False
    transparent: bool = False
    use_tex: bool = False
    tex_system: str = "pdflatex"
    x_label: str = ""
    y_label: str = ""
    legend_labels: tuple = ()

    @property
    def extension(self):
        return os.path.splitext(self.path)[1].lower().lstrip(".")


def _minmax_downsample(x, y, max_points):
    """Reduce a curve while retaining each block's local minimum and maximum."""
    x = np.asarray(x)
    y = np.asarray(y)
    max_points = max(6, int(max_points))
    if len(x) <= max_points:
        return x, y

    interior_count = len(x) - 2
    bucket_count = max(1, (max_points - 4) // 2)
    block_size = max(1, int(np.ceil(interior_count / bucket_count)))
    full_count = (interior_count // block_size) * block_size
    selected = []

    if full_count:
        blocks = y[1:1 + full_count].reshape(-1, block_size)
        starts = 1 + np.arange(len(blocks)) * block_size
        minima = starts + np.argmin(blocks, axis=1)
        maxima = starts + np.argmax(blocks, axis=1)
        pairs = np.column_stack((minima, maxima))
        pairs.sort(axis=1)
        selected.append(pairs.ravel())

    remainder_start = 1 + full_count
    if remainder_start < len(x) - 1:
        remainder = y[remainder_start:-1]
        pair = np.array([
            remainder_start + int(np.argmin(remainder)),
            remainder_start + int(np.argmax(remainder)),
        ])
        selected.append(np.sort(pair))

    middle = np.concatenate(selected) if selected else np.array([], dtype=int)
    indices = np.concatenate((
        np.array([0], dtype=int),
        middle.astype(int, copy=False),
        np.array([len(x) - 1], dtype=int),
    ))
    indices = np.unique(indices)
    return x[indices], y[indices]


def _plot_groups(plot_data, subplots):
    if not subplots:
        return [plot_data]
    labels = []
    for pd in plot_data:
        if pd.sy not in labels:
            labels.append(pd.sy)
    return [[pd for pd in plot_data if pd.sy == label] for label in labels]


def _tex_escape(text):
    text = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }

    def escape_plain(value):
        return "".join(replacements.get(char, char) for char in value)

    parts = text.split("$")
    return "$".join(
        part if index % 2 else escape_plain(part)
        for index, part in enumerate(parts)
    )


def _apply_limits(axis, limits):
    xmin = limits.get("xmin")
    xmax = limits.get("xmax")
    ymin = limits.get("ymin")
    ymax = limits.get("ymax")
    if xmin is not None or xmax is not None:
        axis.set_xlim(left=xmin, right=xmax)
    if ymin is not None or ymax is not None:
        axis.set_ylim(bottom=ymin, top=ymax)


def export_publication_plot(
        plot_data,
        options,
        *,
        subplots=False,
        sharex=True,
        logx=False,
        logy=False,
        marker=None,
        step=False,
        axis_limits=None):
    """Render transformed PlotData objects with an isolated Matplotlib figure."""
    if not plot_data:
        raise ValueError("There is no plot data to export")
    extension = options.extension
    supported = {fmt for _label, fmt in _EXPORT_FORMATS} | {"tif"}
    if extension not in supported:
        raise ValueError(
            "Unsupported publication format: .{}".format(extension or "")
        )
    if options.use_tex or extension == "pgf":
        required_tex = options.tex_system if extension == "pgf" else "latex"
        if shutil.which(required_tex) is None:
            raise RuntimeError(
                "{} is required for {} export. Install a LaTeX distribution "
                "or export PDF/SVG without LaTeX text.".format(
                    required_tex,
                    "PGF" if extension == "pgf" else "LaTeX text",
                )
            )

    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rc = {
        "font.family": "serif" if options.use_tex else options.font_family,
        "font.size": options.font_size,
        "axes.labelsize": options.font_size,
        "axes.linewidth": 0.8,
        "axes.titlesize": options.font_size,
        "xtick.labelsize": options.font_size - 1,
        "ytick.labelsize": options.font_size - 1,
        "legend.fontsize": options.font_size - 1,
        "lines.linewidth": options.line_width,
        "savefig.dpi": options.dpi,
        "savefig.transparent": options.transparent,
        "text.usetex": options.use_tex,
        "pgf.texsystem": options.tex_system,
        "pgf.rcfonts": False,
        "path.simplify": True,
        "path.simplify_threshold": 0.1,
        "agg.path.chunksize": 10000,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    groups = _plot_groups(plot_data, subplots)
    axis_limits = dict(axis_limits or {})
    marker = _MARKERS.get(marker, marker)

    with mpl.rc_context(rc):
        figure = Figure(
            figsize=(options.width, options.height),
            dpi=options.dpi,
            facecolor="none" if options.transparent else "white",
        )
        FigureCanvasAgg(figure)
        axes = figure.subplots(
            nrows=len(groups),
            ncols=1,
            sharex=sharex and len(groups) > 1,
            squeeze=False,
        )[:, 0]
        curve_index = 0
        for axis, group in zip(axes, groups):
            for pd in group:
                x, y = _plot_ready_xy(pd.x, pd.y, logx=logx, logy=logy)
                if len(x) == 0:
                    continue
                x, y = _minmax_downsample(x, y, options.max_points)
                color = np.asarray(
                    _PLOT_PALETTE[curve_index % len(_PLOT_PALETTE)],
                    dtype=float,
                ) / 255.0
                label = "Set {}".format(curve_index + 1)
                if curve_index < len(options.legend_labels):
                    custom_label = str(
                        options.legend_labels[curve_index]
                    ).strip()
                    if custom_label:
                        label = custom_label
                if options.use_tex:
                    label = _tex_escape(label)
                markevery = max(1, len(x) // 2000) if marker else None
                axis.plot(
                    x,
                    y,
                    label=label,
                    color=color,
                    linewidth=options.line_width,
                    marker=marker,
                    markersize=3.0 if marker else 0.0,
                    markeredgewidth=0.5,
                    markevery=markevery,
                    drawstyle="steps-mid" if step else "default",
                )
                curve_index += 1

            ylabel = options.y_label.strip() or " and ".join(
                sorted(set(pd.sy for pd in group))
            )
            axis.set_ylabel(_tex_escape(ylabel) if options.use_tex else ylabel)
            axis.set_xscale("log" if logx else "linear")
            axis.set_yscale("log" if logy else "linear")
            if options.grid:
                axis.grid(
                    True,
                    which="both",
                    color="#b8c0c8",
                    linewidth=0.5,
                    alpha=0.65,
                )
            else:
                axis.grid(False)
            axis.tick_params(
                axis="both",
                which="both",
                direction="in",
                top=True,
                right=True,
                width=0.8,
            )
            for spine in axis.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("black")
            if options.legend and axis.lines:
                axis.legend(frameon=False, handlelength=2.4)
            _apply_limits(axis, axis_limits)

        xlabel = options.x_label.strip() or PDL_xlabel(plot_data)
        axes[-1].set_xlabel(
            _tex_escape(xlabel) if options.use_tex else xlabel
        )
        figure.align_ylabels(axes)
        figure.tight_layout(pad=0.6)

        save_kwargs = {
            "format": extension,
            "transparent": options.transparent,
        }
        if extension in _RASTER_FORMATS:
            save_kwargs["dpi"] = options.dpi
        if extension in {"pdf", "png", "svg"}:
            save_kwargs["metadata"] = {
                "Creator": "pyDatView publication exporter"
            }
        figure.savefig(options.path, **save_kwargs)
        figure.clear()
    return options.path


class PublicationExportDialog(QtWidgets.QDialog):
    def __init__(self, initial=None, settings=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Publication export")
        self.setMinimumWidth(540)
        self.settings = settings or QtCore.QSettings("NREL", "pyDatView")
        initial = dict(initial or {})

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        output_form = QtWidgets.QFormLayout()
        path_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setText(self._default_path())
        self.browse_button = QtWidgets.QToolButton()
        self.browse_button.setText("...")
        self.browse_button.setToolTip("Choose output file")
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(self.browse_button)
        output_form.addRow("Output file", path_row)

        self.format_combo = QtWidgets.QComboBox()
        for label, extension in _EXPORT_FORMATS:
            self.format_combo.addItem(label, extension)
        saved_format = str(self.settings.value("publication/format", "pdf"))
        self.format_combo.setCurrentIndex(
            max(0, self.format_combo.findData(saved_format))
        )
        output_form.addRow("Format", self.format_combo)
        root.addLayout(output_form)

        figure_form = QtWidgets.QFormLayout()
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("Single column", (3.5, 2.6))
        self.preset_combo.addItem("Double column", (7.2, 4.5))
        self.preset_combo.addItem("Presentation", (10.0, 6.0))
        self.preset_combo.addItem("Custom", None)
        figure_form.addRow("Size preset", self.preset_combo)

        size_row = QtWidgets.QHBoxLayout()
        self.width_spin = QtWidgets.QDoubleSpinBox()
        self.height_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.width_spin, self.height_spin):
            spin.setRange(1.0, 30.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            spin.setSuffix(" in")
        self.width_spin.setValue(
            float(self.settings.value("publication/width", 7.2))
        )
        self.height_spin.setValue(
            float(self.settings.value("publication/height", 4.5))
        )
        saved_size = (
            round(self.width_spin.value(), 2),
            round(self.height_spin.value(), 2),
        )
        preset_index = self.preset_combo.findText("Custom")
        for index in range(self.preset_combo.count() - 1):
            if tuple(self.preset_combo.itemData(index)) == saved_size:
                preset_index = index
                break
        self.preset_combo.setCurrentIndex(preset_index)
        size_row.addWidget(self.width_spin)
        size_row.addWidget(QtWidgets.QLabel("x"))
        size_row.addWidget(self.height_spin)
        figure_form.addRow("Dimensions", size_row)

        self.dpi_spin = QtWidgets.QSpinBox()
        self.dpi_spin.setRange(72, 1200)
        self.dpi_spin.setSingleStep(50)
        self.dpi_spin.setValue(
            int(self.settings.value("publication/dpi", 600))
        )
        figure_form.addRow("Raster resolution", self.dpi_spin)

        self.max_points_spin = QtWidgets.QSpinBox()
        self.max_points_spin.setRange(1000, 1000000)
        self.max_points_spin.setSingleStep(10000)
        self.max_points_spin.setValue(
            int(self.settings.value("publication/max_points", 50000))
        )
        self.max_points_spin.setSuffix(" / curve")
        figure_form.addRow("Vector points", self.max_points_spin)
        root.addLayout(figure_form)

        type_form = QtWidgets.QFormLayout()
        self.font_combo = QtWidgets.QComboBox()
        self.font_combo.setEditable(True)
        self.font_combo.addItems([
            "DejaVu Serif",
            "DejaVu Sans",
            "STIXGeneral",
            "Times New Roman",
            "Computer Modern Roman",
        ])
        self.font_combo.setCurrentText(
            str(self.settings.value(
                "publication/font_family", "DejaVu Serif"
            ))
        )
        type_form.addRow("Font family", self.font_combo)

        self.font_size_spin = QtWidgets.QDoubleSpinBox()
        self.font_size_spin.setRange(5.0, 30.0)
        self.font_size_spin.setSingleStep(0.5)
        self.font_size_spin.setSuffix(" pt")
        self.font_size_spin.setValue(
            float(self.settings.value("publication/font_size", 9.0))
        )
        type_form.addRow("Font size", self.font_size_spin)

        self.line_width_spin = QtWidgets.QDoubleSpinBox()
        self.line_width_spin.setRange(0.25, 8.0)
        self.line_width_spin.setSingleStep(0.25)
        self.line_width_spin.setValue(float(initial.get("line_width", 1.25)))
        type_form.addRow("Line width", self.line_width_spin)

        self.x_label_edit = QtWidgets.QLineEdit()
        self.x_label_edit.setText(str(initial.get("x_label", "")))
        self.x_label_edit.setPlaceholderText("Use current X-axis label")
        type_form.addRow("X-axis label", self.x_label_edit)

        self.y_label_edit = QtWidgets.QLineEdit()
        self.y_label_edit.setText(str(initial.get("y_label", "")))
        self.y_label_edit.setPlaceholderText(
            "Use current Y-axis or subplot labels"
        )
        type_form.addRow("Y-axis label", self.y_label_edit)

        self.use_tex_check = QtWidgets.QCheckBox("Use LaTeX text")
        self.use_tex_check.setChecked(
            self.settings.value("publication/use_tex", False, type=bool)
        )
        type_form.addRow("", self.use_tex_check)

        self.tex_system_combo = QtWidgets.QComboBox()
        self.tex_system_combo.addItems(["pdflatex", "xelatex", "lualatex"])
        saved_tex = str(
            self.settings.value("publication/tex_system", "pdflatex")
        )
        self.tex_system_combo.setCurrentIndex(
            max(0, self.tex_system_combo.findText(saved_tex))
        )
        type_form.addRow("LaTeX engine", self.tex_system_combo)
        root.addLayout(type_form)

        options_row = QtWidgets.QHBoxLayout()
        self.grid_check = QtWidgets.QCheckBox("Grid")
        self.legend_check = QtWidgets.QCheckBox("Legend")
        self.transparent_check = QtWidgets.QCheckBox("Transparent background")
        self.grid_check.setChecked(bool(initial.get("grid", False)))
        self.legend_check.setChecked(bool(initial.get("legend", False)))
        self.transparent_check.setChecked(
            self.settings.value(
                "publication/transparent", False, type=bool
            )
        )
        options_row.addWidget(self.grid_check)
        options_row.addWidget(self.legend_check)
        options_row.addWidget(self.transparent_check)
        options_row.addStretch(1)
        root.addLayout(options_row)

        legend_sources = list(initial.get("legend_sources", []))
        legend_labels = list(initial.get("legend_labels", []))
        legend_count = max(len(legend_sources), len(legend_labels))
        self.legend_table = QtWidgets.QTableWidget(legend_count, 2)
        self.legend_table.setHorizontalHeaderLabels([
            "Curve", "Legend label"
        ])
        self.legend_table.verticalHeader().setVisible(False)
        self.legend_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )
        self.legend_table.setAlternatingRowColors(True)
        self.legend_table.setMinimumHeight(90)
        self.legend_table.setMaximumHeight(190)
        header = self.legend_table.horizontalHeader()
        header.setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        for index in range(legend_count):
            source = (
                str(legend_sources[index])
                if index < len(legend_sources) else ""
            )
            label = (
                str(legend_labels[index])
                if index < len(legend_labels)
                else "Set {}".format(index + 1)
            )
            curve_item = QtWidgets.QTableWidgetItem(
                "Set {}".format(index + 1)
            )
            curve_item.setFlags(
                curve_item.flags() & ~QtCore.Qt.ItemIsEditable
            )
            curve_item.setToolTip(source)
            label_item = QtWidgets.QTableWidgetItem(label)
            label_item.setToolTip(source)
            self.legend_table.setItem(index, 0, curve_item)
            self.legend_table.setItem(index, 1, label_item)
        root.addWidget(QtWidgets.QLabel("LEGEND LABELS"))
        root.addWidget(self.legend_table)
        self.legend_table.setEnabled(self.legend_check.isChecked())

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel
        )
        self.export_button = buttons.addButton(
            "Export", QtWidgets.QDialogButtonBox.AcceptRole
        )
        self.export_button.setObjectName("primaryButton")
        root.addWidget(buttons)

        self.browse_button.clicked.connect(self.browse)
        self.format_combo.currentIndexChanged.connect(self.on_format_changed)
        self.preset_combo.currentIndexChanged.connect(self.apply_preset)
        self.width_spin.valueChanged.connect(self.set_custom_preset)
        self.height_spin.valueChanged.connect(self.set_custom_preset)
        self.use_tex_check.toggled.connect(self.update_tex_controls)
        self.legend_check.toggled.connect(self.legend_table.setEnabled)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.on_format_changed()

    def _default_path(self):
        directory = str(
            self.settings.value("publication/directory", os.getcwd())
        )
        return os.path.join(directory, "pydatview_plot.pdf")

    def current_extension(self):
        return str(self.format_combo.currentData())

    def on_format_changed(self, _index=None):
        extension = self.current_extension()
        path = self.path_edit.text().strip()
        root, current = os.path.splitext(path)
        known = {"." + fmt for _label, fmt in _EXPORT_FORMATS}
        if current.lower() in known:
            self.path_edit.setText(root + "." + extension)
        self.dpi_spin.setEnabled(extension in _RASTER_FORMATS)
        if extension == "pgf":
            self.use_tex_check.setChecked(True)
        self.update_tex_controls()

    def update_tex_controls(self, _checked=None):
        self.tex_system_combo.setEnabled(
            self.use_tex_check.isChecked()
            or self.current_extension() == "pgf"
        )

    def apply_preset(self, index):
        size = self.preset_combo.itemData(index)
        if size is None:
            return
        self.width_spin.blockSignals(True)
        self.height_spin.blockSignals(True)
        self.width_spin.setValue(size[0])
        self.height_spin.setValue(size[1])
        self.width_spin.blockSignals(False)
        self.height_spin.blockSignals(False)

    def set_custom_preset(self, _value):
        custom = self.preset_combo.findText("Custom")
        self.preset_combo.blockSignals(True)
        self.preset_combo.setCurrentIndex(custom)
        self.preset_combo.blockSignals(False)

    def browse(self):
        extension = self.current_extension()
        label = self.format_combo.currentText()
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export publication plot",
            self.path_edit.text(),
            "{} (*.{})".format(label, extension),
        )
        if path:
            if not os.path.splitext(path)[1]:
                path += "." + extension
            self.path_edit.setText(path)

    def options(self):
        path = self.path_edit.text().strip()
        extension = self.current_extension()
        if os.path.splitext(path)[1].lower() != "." + extension:
            path = os.path.splitext(path)[0] + "." + extension
        legend_labels = tuple(
            self.legend_table.item(row, 1).text()
            for row in range(self.legend_table.rowCount())
        )
        return PublicationExportOptions(
            path=path,
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            dpi=self.dpi_spin.value(),
            font_family=self.font_combo.currentText().strip(),
            font_size=self.font_size_spin.value(),
            line_width=self.line_width_spin.value(),
            max_points=self.max_points_spin.value(),
            grid=self.grid_check.isChecked(),
            legend=self.legend_check.isChecked(),
            transparent=self.transparent_check.isChecked(),
            use_tex=self.use_tex_check.isChecked(),
            tex_system=self.tex_system_combo.currentText(),
            x_label=self.x_label_edit.text(),
            y_label=self.y_label_edit.text(),
            legend_labels=legend_labels,
        )

    def accept(self):
        options = self.options()
        if not options.path:
            QtWidgets.QMessageBox.warning(
                self, "Publication export", "Select an output file."
            )
            return
        directory = os.path.dirname(os.path.abspath(options.path))
        if not os.path.isdir(directory):
            QtWidgets.QMessageBox.warning(
                self, "Publication export", "The output folder does not exist."
            )
            return
        self.settings.setValue("publication/directory", directory)
        self.settings.setValue("publication/format", options.extension)
        self.settings.setValue("publication/width", options.width)
        self.settings.setValue("publication/height", options.height)
        self.settings.setValue("publication/dpi", options.dpi)
        self.settings.setValue("publication/max_points", options.max_points)
        self.settings.setValue(
            "publication/font_family", options.font_family
        )
        self.settings.setValue("publication/font_size", options.font_size)
        self.settings.setValue("publication/use_tex", options.use_tex)
        self.settings.setValue(
            "publication/tex_system", options.tex_system
        )
        self.settings.setValue(
            "publication/transparent", options.transparent
        )
        self.settings.sync()
        super().accept()
