"""PyQtGraph canvas and plot styling for the Qt GUI."""

import html

import numpy as np

from pydatview.plotdata import PDL_xlabel
from pydatview.qt_compat import QtCore, QtGui, QtWidgets, pg
from pydatview.qt_stats import _plot_ready_xy

_PLOT_PALETTE = (
    # Matplotlib's Tableau palette first, followed by distinct extensions.
    (31, 119, 180),   # tab:blue
    (255, 127, 14),   # tab:orange
    (44, 160, 44),    # tab:green
    (214, 39, 40),    # tab:red
    (148, 103, 189),  # tab:purple
    (140, 86, 75),    # tab:brown
    (227, 119, 194),  # tab:pink
    (127, 127, 127),  # tab:gray
    (188, 189, 34),   # tab:olive
    (23, 190, 207),   # tab:cyan
    (57, 59, 121),
    (230, 85, 13),
    (0, 107, 164),
    (102, 166, 30),
    (123, 65, 115),
    (166, 54, 3),
)


def _curve_color(idx):
    return _PLOT_PALETTE[idx % len(_PLOT_PALETTE)]


def _curve_pen(idx, width=1.5):
    return pg.mkPen(color=_curve_color(idx), width=width)


def _selected_curve_pen(width=1.5):
    return pg.mkPen(color=(17, 24, 39), width=max(width + 2.5, 3.5))

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

class QtPlotCanvas(pg.GraphicsLayoutWidget):
    curveSelected = QtCore.Signal(object)
    hoverCoordinates = QtCore.Signal(object)
    measurementMarkerChanged = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        pg.setConfigOptions(useOpenGL=True, antialias=True, background="w", foreground="k")
        self.setBackground("w")
        self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.setMouseTracking(True)
        self.viewport().setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.viewport().setMouseTracking(True)
        self._plots = []
        self._curve_items = []
        self._curve_records = []
        self._selected_curve = None
        self._zoom_mode = False
        self._logx = False
        self._logy = False
        self._measurement_marker_enabled = False
        self._measurement_marker_x = None
        self._measurement_items = []
        self.measurement_values = []
        self._mouse_proxy = pg.SignalProxy(
            self.scene().sigMouseMoved,
            rateLimit=60,
            slot=self._on_mouse_moved,
        )
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    def clear_plot(self):
        self.clear()
        self._plots = []
        self._curve_items = []
        self._curve_records = []
        self._measurement_items = []
        self.measurement_values = []
        self._selected_curve = None
        self.hoverCoordinates.emit(None)

    def set_zoom_mode(self, enabled):
        self._zoom_mode = bool(enabled)
        mouse_mode = pg.ViewBox.RectMode if enabled else pg.ViewBox.PanMode
        for plot in self._plots:
            plot.getViewBox().setMouseMode(mouse_mode)

    def set_measurement_marker_enabled(self, enabled):
        self._measurement_marker_enabled = bool(enabled)
        if not enabled:
            self.clear_measurement_marker()

    def clear_measurement_marker(self):
        for plot, item in self._measurement_items:
            try:
                plot.removeItem(item)
            except (RuntimeError, ValueError):
                pass
        self._measurement_items = []
        self._measurement_marker_x = None
        self.measurement_values = []
        self.measurementMarkerChanged.emit([])

    @staticmethod
    def _display_axis_value(value, logarithmic):
        value = float(value)
        if not logarithmic:
            return value
        with np.errstate(over="ignore", invalid="ignore"):
            return float(np.power(10.0, value))

    def coordinates_at(self, scene_position):
        for plot_index, plot in enumerate(self._plots):
            view_box = plot.getViewBox()
            if not view_box.sceneBoundingRect().contains(scene_position):
                continue
            point = view_box.mapSceneToView(scene_position)
            return {
                "plot_index": plot_index,
                "plot_count": len(self._plots),
                "x": self._display_axis_value(point.x(), self._logx),
                "y": self._display_axis_value(point.y(), self._logy),
            }
        return None

    def _on_mouse_moved(self, event):
        scene_position = event[0] if isinstance(event, (tuple, list)) else event
        self.hoverCoordinates.emit(self.coordinates_at(scene_position))

    def _on_mouse_clicked(self, event):
        if not self._measurement_marker_enabled or self._zoom_mode:
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        scene_position = event.scenePos()
        for plot in self._plots:
            # Include the bottom axis as well as the plotting rectangle.
            if not plot.sceneBoundingRect().contains(scene_position):
                continue
            view_point = plot.getViewBox().mapSceneToView(scene_position)
            x = self._display_axis_value(view_point.x(), self._logx)
            if np.isfinite(x):
                self.set_measurement_marker(x)
            return

    def leaveEvent(self, event):
        self.hoverCoordinates.emit(None)
        super().leaveEvent(event)

    def plot_data(self, plot_data, *, subplots=False, sharex=True, grid=True,
                  logx=False, logy=False, show_legend=True, line_width=1.5,
                  marker=None, step=False, axis_limits=None,
                  order_overlays=None):
        # QGraphicsView's OpenGL viewport can crash on Windows when log transforms
        # discard points. Keep accelerated rendering for regular plots.
        self.useOpenGL(not (logx or logy))
        self.clear_plot()
        self._logx = bool(logx)
        self._logy = bool(logy)
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
            plot.getViewBox().setMouseMode(
                pg.ViewBox.RectMode if self._zoom_mode else pg.ViewBox.PanMode
            )
            plot.showGrid(x=grid, y=grid, alpha=0.25)
            label_style = {
                "color": "#262626",
                "font-family": "DejaVu Sans",
                "font-size": "10pt",
            }
            ylabel = " and ".join(sorted(set(pd.sy for pd in group)))
            if len(ylabel) < 120:
                plot.setLabel("left", ylabel, **label_style)
            if i_group == len(groups) - 1:
                plot.setLabel("bottom", PDL_xlabel(plot_data), **label_style)
            if show_legend:
                plot.addLegend(offset=(10, 10), labelTextColor="k", brush=(255, 255, 255, 210))

            categorical_ticks = []
            for pd in group:
                try:
                    x, y = _plot_ready_xy(pd.x, pd.y, logx=logx, logy=logy)
                except Exception as exc:
                    print("Skipping non-numeric curve {}: {}".format(pd.sy, exc))
                    continue
                if len(x) == 0:
                    continue
                color_index = getattr(pd, "color_index", curve_idx)
                curve_color = _curve_color(color_index)
                box_stats = getattr(pd, "boxplot_stats", None)
                if box_stats is not None:
                    center = float(x[0])
                    box_width = 0.62
                    outline = pg.mkPen(curve_color, width=max(1.25, line_width))
                    fill = pg.mkBrush(*curve_color, 105)
                    box = pg.BarGraphItem(
                        x=[center],
                        y0=[box_stats["q1"]],
                        height=[box_stats["q3"] - box_stats["q1"]],
                        width=box_width,
                        pen=outline,
                        brush=fill,
                    )
                    plot.addItem(box)
                    cap_half_width = box_width * 0.28
                    segment_x = np.asarray([
                        center, center, np.nan,
                        center, center, np.nan,
                        center - cap_half_width, center + cap_half_width, np.nan,
                        center - cap_half_width, center + cap_half_width,
                    ])
                    segment_y = np.asarray([
                        box_stats["minimum"], box_stats["q1"], np.nan,
                        box_stats["q3"], box_stats["maximum"], np.nan,
                        box_stats["minimum"], box_stats["minimum"], np.nan,
                        box_stats["maximum"], box_stats["maximum"],
                    ])
                    plot.plot(segment_x, segment_y, pen=outline, connect="finite")
                    plot.plot(
                        [center - box_width / 2, center + box_width / 2],
                        [box_stats["median"], box_stats["median"]],
                        pen=pg.mkPen((30, 30, 30), width=max(1.5, line_width)),
                    )
                    item = plot.plot(
                        [center],
                        [box_stats["mean"]],
                        name=pd.syl or pd.sy,
                        pen=None,
                        symbol="d",
                        symbolSize=9,
                        symbolBrush=pg.mkBrush(curve_color),
                        symbolPen=pg.mkPen((20, 20, 20), width=1.25),
                    )
                    base_pen = pg.mkPen((20, 20, 20), width=1.25)
                    categorical_ticks.append((center, pd.boxplot_label))
                else:
                    item = plot.plot(
                        x,
                        y,
                        name=pd.syl or pd.sy,
                        pen=_curve_pen(color_index, width=line_width),
                        symbol=marker,
                        symbolSize=5 if marker else None,
                        symbolBrush=curve_color if marker else None,
                        symbolPen=pg.mkPen(curve_color) if marker else None,
                        skipFiniteCheck=not (logx or logy),
                    )
                    base_pen = _curve_pen(color_index, width=line_width)
                item.setClipToView(True)
                item.setDownsampling(auto=True, method="peak")
                item.setCurveClickable(True, width=8)
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
                    "boxplot": box_stats is not None,
                }
                item.sigClicked.connect(lambda clicked_item, _ev, meta=meta: self.select_curve(clicked_item, meta))
                self._curve_items.append((item, base_pen, meta))
                self._curve_records.append({
                    "plot_index": i_group,
                    # Marker annotations deliberately omit the simulation/file label.
                    "label": pd.sy,
                    "x": np.asarray(x),
                    "y": np.asarray(y),
                })
                curve_idx += 1

            if categorical_ticks:
                bottom_axis = plot.getAxis("bottom")
                bottom_axis.setTicks([categorical_ticks])
                bottom_axis.setStyle(
                    tickTextHeight=42,
                    autoExpandTextSpace=True,
                    hideOverlappingLabels=False,
                )

            if logx or logy:
                plot.setLogMode(x=logx, y=logy)
            self._apply_axis_limits(
                plot,
                axis_limits or {},
                logx=logx,
                logy=logy,
            )
            self._add_order_overlays(plot, order_overlays or [], logx=logx)

        if self._measurement_marker_enabled and self._measurement_marker_x is not None:
            self.set_measurement_marker(self._measurement_marker_x)

    @staticmethod
    def order_marker_positions(rotor_speed, speed_unit="rpm", orders=None, x_type="1/x"):
        unit = str(speed_unit or "rpm").lower()
        if unit == "rpm":
            rotor_frequency = float(rotor_speed) / 60.0
        elif unit == "hz":
            rotor_frequency = float(rotor_speed)
        elif unit in ("rad/s", "radps", "rad_per_s"):
            rotor_frequency = float(rotor_speed) / (2.0 * np.pi)
        else:
            raise ValueError("Unsupported rotor speed unit '{}'".format(speed_unit))
        if not np.isfinite(rotor_frequency) or rotor_frequency <= 0.0:
            raise ValueError("Rotor speed must be positive")

        markers = []
        for order in orders or (1.0, 3.0, 6.0):
            order = float(order)
            frequency = order * rotor_frequency
            if not np.isfinite(frequency) or frequency <= 0.0:
                continue
            if x_type == "2pi/x":
                x = 2.0 * np.pi * frequency
            elif x_type == "x":
                x = 1.0 / frequency
            else:
                x = frequency
            markers.append({
                "x": float(x),
                "label": "{:g}P".format(order),
                "frequency": float(frequency),
            })
        return markers

    def _add_order_overlays(self, plot, markers, logx=False):
        if not markers:
            return
        for marker in markers:
            marker_x = float(marker.get("x", np.nan))
            if not np.isfinite(marker_x) or (logx and marker_x <= 0.0):
                continue
            display_x = np.log10(marker_x) if logx else marker_x
            line = pg.InfiniteLine(
                pos=display_x,
                angle=90,
                movable=False,
                pen=pg.mkPen((31, 119, 180), width=1.25, style=QtCore.Qt.DotLine),
                label=str(marker.get("label", "")),
                labelOpts={
                    "position": 0.95,
                    "color": "#1f77b4",
                    "fill": pg.mkBrush(255, 255, 255, 210),
                    "movable": False,
                },
            )
            line.setZValue(28)
            plot.addItem(line, ignoreBounds=True)
            if getattr(line, "label", None) is not None:
                line.label.setZValue(29)

    @staticmethod
    def _interpolate_curve(x, y, marker_x):
        if len(x) == 0:
            return None
        finite = np.isfinite(x) & np.isfinite(y)
        x = np.asarray(x[finite], dtype=float)
        y = np.asarray(y[finite], dtype=float)
        if len(x) == 0 or marker_x < np.min(x) or marker_x > np.max(x):
            return None
        order = np.argsort(x, kind="stable")
        x = x[order]
        y = y[order]
        x, unique_indices = np.unique(x, return_index=True)
        y = y[unique_indices]
        if len(x) == 1:
            return float(y[0]) if marker_x == x[0] else None
        return float(np.interp(marker_x, x, y))

    @staticmethod
    def _marker_number(value):
        return "{:.7g}".format(float(value))

    def set_measurement_marker(self, x):
        """Place a vertical marker and show every curve value at *x*."""
        if not self._measurement_marker_enabled:
            return
        x = float(x)
        if not np.isfinite(x) or (self._logx and x <= 0):
            return
        for plot, item in self._measurement_items:
            try:
                plot.removeItem(item)
            except (RuntimeError, ValueError):
                pass
        self._measurement_items = []
        self._measurement_marker_x = x
        self.measurement_values = []

        display_x = np.log10(x) if self._logx else x
        values_by_plot = {index: [] for index in range(len(self._plots))}
        for record in self._curve_records:
            value = self._interpolate_curve(record["x"], record["y"], x)
            if value is None or (self._logy and value <= 0):
                continue
            result = {
                "plot_index": record["plot_index"],
                "label": record["label"],
                "x": x,
                "y": value,
            }
            self.measurement_values.append(result)
            values_by_plot[record["plot_index"]].append(result)

        for plot_index, plot in enumerate(self._plots):
            line = pg.InfiniteLine(
                pos=display_x,
                angle=90,
                movable=False,
                pen=pg.mkPen((198, 40, 40), width=1.5, style=QtCore.Qt.DashLine),
            )
            line.setZValue(20)
            plot.addItem(line, ignoreBounds=True)
            self._measurement_items.append((plot, line))

            plot_values = values_by_plot[plot_index]
            if plot_values:
                spots = []
                for result in plot_values:
                    display_y = np.log10(result["y"]) if self._logy else result["y"]
                    spots.append({
                        "pos": (display_x, display_y),
                        "size": 8,
                        "brush": pg.mkBrush(198, 40, 40),
                        "pen": pg.mkPen("w", width=1),
                    })
                scatter = pg.ScatterPlotItem(spots=spots)
                scatter.setZValue(21)
                plot.addItem(scatter)
                self._measurement_items.append((plot, scatter))

            marker_font = QtGui.QFont(QtWidgets.QApplication.font())
            marker_font.setPointSize(max(7, marker_font.pointSize() - 2))
            x_range, y_range = plot.getViewBox().viewRange()
            x_span = x_range[1] - x_range[0]
            y_span = y_range[1] - y_range[0]
            label_x = display_x + 0.015 * x_span
            anchor_x = 0
            if display_x > x_range[0] + 0.65 * x_span:
                anchor_x = 1
                label_x = display_x - 0.015 * x_span

            x_label = pg.TextItem(
                html='<span style="color:#c62828;"><b>x = {}</b></span>'.format(
                    self._marker_number(x)
                ),
                anchor=(anchor_x, 0),
                fill=pg.mkBrush(255, 255, 255, 225),
                border=pg.mkPen(198, 40, 40),
            )
            x_label.setFont(marker_font)
            x_label.setPos(label_x, y_range[1] - 0.025 * y_span)
            x_label.setZValue(22)
            plot.addItem(x_label, ignoreBounds=True)
            self._measurement_items.append((plot, x_label))

            for result in plot_values:
                display_y = np.log10(result["y"]) if self._logy else result["y"]
                value_label = pg.TextItem(
                    html='<span style="color:#c62828;">{}: y = {}</span>'.format(
                        html.escape(str(result["label"])),
                        self._marker_number(result["y"]),
                    ),
                    anchor=(anchor_x, 0.5),
                    fill=pg.mkBrush(255, 255, 255, 215),
                )
                value_label.setFont(marker_font)
                value_label.setPos(label_x, display_y)
                value_label.setZValue(22)
                plot.addItem(value_label, ignoreBounds=True)
                self._measurement_items.append((plot, value_label))

        self.measurementMarkerChanged.emit(list(self.measurement_values))

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
        for item, base_pen, item_meta in self._curve_items:
            if item_meta.get("boxplot"):
                item.setSymbolPen(base_pen)
            else:
                item.setPen(base_pen)
        if meta.get("boxplot"):
            selected_item.setSymbolPen(
                _selected_curve_pen(meta.get("line_width", 1.25))
            )
        else:
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
        tick_font = QtGui.QFont("DejaVu Sans")
        tick_font.setPointSize(9)
        for axis_name in ("bottom", "left", "top", "right"):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen((38, 38, 38), width=1.0))
            axis.setTextPen(pg.mkPen((38, 38, 38)))
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
        plot.getViewBox().setBorder(pg.mkPen((38, 38, 38), width=1.0))

    @staticmethod
    def _group_plot_data(plot_data, subplots):
        if not subplots:
            return [plot_data]
        labels = []
        for pd in plot_data:
            if pd.sy not in labels:
                labels.append(pd.sy)
        return [[pd for pd in plot_data if pd.sy == label] for label in labels]
