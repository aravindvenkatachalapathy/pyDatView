"""Fast PySide6/PyQtGraph plot window for pyDatView.

This module intentionally runs as a separate process from the wxPython GUI.
Mixing wx and Qt event loops in the same process is fragile; exporting the
current PlotData arrays and launching this module keeps the integration small
and stable while moving large line rendering to PyQtGraph.
"""

import argparse
import pickle
import sys

import numpy as np


def _require_qtgraph():
    try:
        from PySide6 import QtCore, QtWidgets
        import pyqtgraph as pg
    except ImportError as exc:
        raise SystemExit(
            "Fast Plot requires PySide6 and pyqtgraph.\n"
            "Install them with: pip install PySide6 pyqtgraph"
        ) from exc
    return QtCore, QtWidgets, pg


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


def _curve_pen(pg, idx):
    color = pg.intColor(idx, hues=12, values=1, maxValue=220)
    return pg.mkPen(color=color, width=1.25)


def make_numeric_axis(pg):
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
    return NumericAxisItem


class FastPlotWindow:
    def __init__(self, payload):
        self.QtCore, self.QtWidgets, self.pg = _require_qtgraph()
        self.payload = payload
        self.app = self.QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = self.QtWidgets.QApplication(sys.argv[:1])

        self.pg.setConfigOptions(useOpenGL=True, antialias=False, background="w", foreground="k")

        self.window = self.QtWidgets.QMainWindow()
        self.window.setWindowTitle(payload.get("title", "pyDatView Fast Plot"))
        self.graphics = self.pg.GraphicsLayoutWidget()
        self.graphics.setBackground("w")
        self.window.setCentralWidget(self.graphics)
        self.plots = []
        self._build()

    def _build(self):
        groups = self.payload.get("groups", [])
        if not groups:
            label = self.QtWidgets.QLabel("No numeric data to plot")
            label.setAlignment(self.QtCore.Qt.AlignCenter)
            self.window.setCentralWidget(label)
            return

        previous_plot = None
        global_curve_idx = 0
        NumericAxisItem = make_numeric_axis(self.pg)
        for i_group, group in enumerate(groups):
            plot = self.graphics.addPlot(
                row=i_group,
                col=0,
                axisItems={
                    "bottom": NumericAxisItem(orientation="bottom"),
                    "left": NumericAxisItem(orientation="left"),
                    "top": NumericAxisItem(orientation="top"),
                    "right": NumericAxisItem(orientation="right"),
                },
            )
            if previous_plot is not None and self.payload.get("sharex", True):
                plot.setXLink(previous_plot)
            previous_plot = plot
            self.plots.append(plot)

            self._style_plot(plot)
            plot.showGrid(x=self.payload.get("grid", False), y=self.payload.get("grid", False), alpha=0.25)
            plot.setLabel("bottom", group.get("xlabel", ""))
            plot.setLabel("left", group.get("ylabel", ""))
            plot.addLegend(offset=(10, 10))

            for curve in group.get("curves", []):
                try:
                    x, y = _finite_xy(curve["x"], curve["y"])
                except Exception as exc:
                    print("Skipping non-numeric curve {}: {}".format(curve.get("label", ""), exc))
                    continue
                if len(x) == 0:
                    continue
                if curve.get("swap_xy", False):
                    x, y = y, x
                item = plot.plot(
                    x,
                    y,
                    name=curve.get("label", ""),
                    pen=_curve_pen(self.pg, global_curve_idx),
                    skipFiniteCheck=True,
                )
                item.setClipToView(True)
                item.setDownsampling(auto=True, method="peak")
                global_curve_idx += 1

            if self.payload.get("logx", False):
                plot.setLogMode(x=True, y=False)
            if self.payload.get("logy", False):
                plot.setLogMode(x=self.payload.get("logx", False), y=True)

        if self.plots:
            self.plots[-1].setLabel("bottom", self.payload.get("xlabel", ""))

    def _style_plot(self, plot):
        plot.showAxis("bottom", True)
        plot.showAxis("left", True)
        plot.showAxis("top", True)
        plot.showAxis("right", True)
        tick_font = self.QtWidgets.QApplication.font()
        tick_font.setPointSize(max(8, tick_font.pointSize()))
        for axis_name in ("bottom", "left", "top", "right"):
            axis = plot.getAxis(axis_name)
            axis.setPen(self.pg.mkPen("k"))
            axis.setTextPen(self.pg.mkPen("k"))
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
        plot.getViewBox().setBorder(self.pg.mkPen((180, 180, 180)))

    def run(self):
        self.window.resize(1200, 800)
        self.window.show()
        return self.app.exec()


def load_payload(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(description="pyDatView PyQtGraph fast plot window")
    parser.add_argument("payload", help="Pickle payload exported by pyDatView")
    args = parser.parse_args(argv)
    window = FastPlotWindow(load_payload(args.payload))
    return window.run()


if __name__ == "__main__":
    raise SystemExit(main())
