"""Numerical transforms, statistics, and comparison helpers for Qt plots."""

import os

import numpy as np

from pydatview.common import no_unit, unit

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


def _sample_spacing(x):
    x = _as_float_array(x)
    if len(x) < 2:
        return np.nan
    spacing = np.abs(np.diff(x))
    spacing = spacing[np.isfinite(spacing) & (spacing > 0)]
    return float(np.median(spacing)) if len(spacing) else np.nan


def _trapezoidal_integral(y, x):
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(y, x)
    return np.trapz(y, x)


_STATS_COLUMNS = (
    ("series", "Series", False),
    ("file", "Filename", False),
    ("directory", "Directory", False),
    ("table", "Table", False),
    ("n", "n", True),
    ("dt", "dt", True),
    ("median", "Median", True),
    ("mean", "Mean", True),
    ("std", "Std", True),
    ("var", "Var", True),
    ("std_mean", "Std/Mean (TI)", True),
    ("min", "Min", True),
    ("max", "Max", True),
    ("x_at_min", "x@Min", True),
    ("x_at_max", "x@Max", True),
    ("abs_max", "Abs. Max", True),
    ("range", "Range", True),
    ("x_min", "xMin", True),
    ("x_max", "xMax", True),
    ("x_range", "xRange", True),
    ("integral", "Integral y dx", True),
    ("integral_mean", "Integral y dx / Integral dx", True),
    ("integral_x", "Integral y*x dx", True),
    ("integral_centroid", "Integral y*x dx / Integral y dx", True),
    ("integral_x2", "Integral y*x^2 dx", True),
)
_DEFAULT_STATS_COLUMNS = (
    "series", "file", "n", "dt", "mean", "std", "min", "max", "range"
)


def _series_statistics(pd, x, y, selected):
    selected = set(selected)
    filename = getattr(pd, "filename", "") or getattr(pd, "st", "")
    values = {
        "series": getattr(pd, "sy", ""),
        "file": os.path.basename(filename),
        "directory": os.path.dirname(filename),
        "table": getattr(pd, "tabname", ""),
        "n": len(y),
    }
    if "dt" in selected:
        values["dt"] = _sample_spacing(x)
    if len(y) == 0:
        return values

    need_extrema = bool(selected & {
        "min", "max", "x_at_min", "x_at_max", "abs_max", "range"
    })
    if need_extrema:
        minimum_index = int(np.argmin(y))
        maximum_index = int(np.argmax(y))
        minimum = float(y[minimum_index])
        maximum = float(y[maximum_index])
        values.update({
            "min": minimum,
            "max": maximum,
            "x_at_min": float(x[minimum_index]),
            "x_at_max": float(x[maximum_index]),
            "abs_max": max(abs(minimum), abs(maximum)),
            "range": maximum - minimum,
        })

    need_mean = bool(selected & {"mean", "std_mean"})
    mean = float(np.mean(y)) if need_mean else np.nan
    if need_mean:
        values["mean"] = mean
    need_std = bool(selected & {"std", "var", "std_mean"})
    std = float(np.std(y)) if need_std else np.nan
    if need_std:
        values["std"] = std
        values["var"] = std ** 2
    if "median" in selected:
        values["median"] = float(np.median(y))
    if "std_mean" in selected:
        values["std_mean"] = std / mean if mean != 0 else np.nan

    need_x_range = bool(selected & {
        "x_min", "x_max", "x_range", "integral_mean"
    })
    if need_x_range:
        x_minimum = float(np.min(x))
        x_maximum = float(np.max(x))
        x_range = x_maximum - x_minimum
        values.update({
            "x_min": x_minimum,
            "x_max": x_maximum,
            "x_range": x_range,
        })

    integral_keys = {
        "integral", "integral_mean", "integral_x",
        "integral_centroid", "integral_x2",
    }
    if len(x) > 1 and selected & integral_keys:
        integral = float(_trapezoidal_integral(y, x))
        values["integral"] = integral
        if "integral_mean" in selected:
            values["integral_mean"] = (
                integral / x_range if x_range != 0 else np.nan
            )
        if selected & {"integral_x", "integral_centroid"}:
            integral_x = float(_trapezoidal_integral(y * x, x))
            values["integral_x"] = integral_x
            values["integral_centroid"] = (
                integral_x / integral if integral != 0 else np.nan
            )
        if "integral_x2" in selected:
            values["integral_x2"] = float(
                _trapezoidal_integral(y * x ** 2, x)
            )
    return values


_COMPARISON_METHODS = ("Relative", "|Relative|", "Ratio", "Absolute", "Y-Y")


def _comparison_error(y, reference, method):
    with np.errstate(divide="ignore", invalid="ignore"):
        if method in ("Relative", "|Relative|"):
            denominator = (
                reference + 1.0
                if np.mean(np.abs(reference)) < 1e-7
                else reference
            )
            result = (y - reference) / denominator * 100.0
            return np.abs(result) if method == "|Relative|" else result
        if method == "Ratio":
            if np.mean(np.abs(reference)) < 1e-7:
                return (y + 1.0) / (reference + 1.0)
            return y / reference
        if method == "Absolute":
            return y - reference
    raise ValueError("Unsupported comparison method: {}".format(method))


def _comparison_axis_label(method, reference_pd, candidate_pd):
    if method == "Relative":
        return "Relative error [%]"
    if method == "|Relative|":
        return "Abs. relative error [%]"
    if method == "Ratio":
        return "Ratio [-]"
    if method == "Absolute":
        units = {unit(reference_pd.sy), unit(candidate_pd.sy)} - {""}
        return (
            "Absolute error [{}]".format(next(iter(units)))
            if len(units) == 1 else "Absolute error"
        )
    raise ValueError("Y-Y uses the candidate channel as its Y-axis label")


def _comparison_source(pd):
    filename = getattr(pd, "filename", "")
    source = os.path.abspath(filename) if filename else getattr(pd, "it", -1)
    return getattr(pd, "pane_index", 0), source


def compare_plot_data(plot_data, method):
    if method not in _COMPARISON_METHODS:
        raise ValueError("Unsupported comparison method: {}".format(method))
    if len(plot_data) < 2:
        raise ValueError("Compare requires at least two selected time series")

    source_count = len({_comparison_source(pd) for pd in plot_data})
    if source_count == 1:
        groups = [plot_data]
    else:
        grouped = {}
        for pd in plot_data:
            key = getattr(pd, "selection_index", getattr(pd, "iy", 0))
            grouped.setdefault(key, []).append(pd)
        groups = list(grouped.values())

    compared = []
    for group in groups:
        if len(group) < 2:
            continue
        reference_pd = group[0]
        if reference_pd.xIsString or reference_pd.yIsString:
            raise ValueError("String channels cannot be compared")
        if reference_pd.xIsDate or reference_pd.yIsDate:
            raise ValueError("Date channels cannot be compared")
        reference_x, reference_y = _finite_xy(reference_pd.x, reference_pd.y)
        if len(reference_x) < 2:
            raise ValueError("The comparison reference needs at least two samples")

        reference_name = "{} - {}".format(
            reference_pd.st, no_unit(reference_pd.sy)
        )
        reference_channel = no_unit(reference_pd.sy)
        for candidate_pd in group[1:]:
            if candidate_pd.xIsString or candidate_pd.yIsString:
                raise ValueError("String channels cannot be compared")
            if candidate_pd.xIsDate or candidate_pd.yIsDate:
                raise ValueError("Date channels cannot be compared")
            candidate_x, candidate_y = _finite_xy(candidate_pd.x, candidate_pd.y)
            if len(candidate_x) < 2:
                continue
            order = np.argsort(candidate_x)
            candidate_y = np.interp(
                reference_x,
                candidate_x[order],
                candidate_y[order],
            )
            candidate_name = "{} - {}".format(
                candidate_pd.st, no_unit(candidate_pd.sy)
            )
            candidate_channel = no_unit(candidate_pd.sy)
            channel_pair = (
                candidate_channel
                if candidate_channel == reference_channel
                else "{} - {}".format(candidate_channel, reference_channel)
            )
            candidate_pd.syl = "{} - {} | {}".format(
                candidate_pd.st, reference_pd.st, channel_pair
            )
            if method == "Y-Y":
                candidate_pd.x = reference_y
                candidate_pd.y = candidate_y
                candidate_pd.sx = reference_name
                candidate_pd.sy = candidate_name
            else:
                candidate_pd.x = reference_x
                candidate_pd.y = _comparison_error(
                    candidate_y, reference_y, method
                )
                candidate_pd.sx = reference_pd.sx
                candidate_pd.sy = _comparison_axis_label(
                    method, reference_pd, candidate_pd
                )
            candidate_pd.xIsString = False
            candidate_pd.yIsString = False
            candidate_pd.xIsDate = False
            candidate_pd.yIsDate = False
            candidate_pd.c = candidate_pd.y
            candidate_pd._post_init()
            compared.append(candidate_pd)

    if not compared:
        raise ValueError(
            "Compare needs at least two matching series in a comparison group"
        )
    return compared


def box_plot_data(plot_data):
    """Return one conventional distribution box for each selected file/curve."""
    from pydatview.plotdata import PlotData

    channel_count = len({
        (
            getattr(pd, "pane_index", 0),
            getattr(pd, "selection_index", getattr(pd, "iy", 0)),
            getattr(pd, "sy", ""),
        )
        for pd in plot_data
    })
    source_colors = {}
    result = []
    for position, pd in enumerate(plot_data):
        if pd.yIsString or pd.yIsDate:
            raise ValueError(
                "Box plots require a numeric Y variable"
            )
        values = np.asarray(pd.y).reshape(-1)
        if values.dtype.kind not in "biuf":
            values = _as_float_array(values).reshape(-1)
        finite = np.isfinite(values)
        if not finite.all():
            values = values[finite]
        if len(values) == 0:
            continue
        q1, median, q3 = np.percentile(values, [25.0, 50.0, 75.0])
        mean = float(np.mean(values))
        filename = getattr(pd, "filename", "")
        file_label = os.path.basename(filename) if filename else getattr(pd, "st", "")
        if not file_label:
            file_label = getattr(pd, "tabname", "") or "File {}".format(position + 1)
        tick_label = (
            "{} | {}".format(file_label, no_unit(pd.sy))
            if channel_count > 1 else file_label
        )
        source = _comparison_source(pd)
        if source not in source_colors:
            source_colors[source] = len(source_colors)

        box = PlotData(
            x=np.asarray([float(position)]),
            y=np.asarray([mean]),
            sx="File",
            sy=pd.sy,
        )
        box.syl = tick_label
        box.st = pd.st
        box.filename = filename
        box.tabname = getattr(pd, "tabname", "")
        box.it = getattr(pd, "it", None)
        box.pane_index = getattr(pd, "pane_index", 0)
        box.selection_index = getattr(
            pd, "selection_index", getattr(pd, "iy", 0)
        )
        box.color_index = source_colors[source]
        box.boxplot_label = tick_label
        box.boxplot_stats = {
            "minimum": float(np.min(values)),
            "q1": float(q1),
            "median": float(median),
            "mean": mean,
            "q3": float(q3),
            "maximum": float(np.max(values)),
        }
        # The Stats tab should continue to describe the source time series.
        box.x0 = pd.x
        box.y0 = pd.y
        result.append(box)

    if not result:
        raise ValueError("Box plots need at least one finite numeric series")
    return result


def swap_plot_axes(pd):
    pd.x, pd.y = pd.y, pd.x
    pd.sx, pd.sy = pd.sy, pd.sx
    pd.xIsString, pd.yIsString = pd.yIsString, pd.xIsString
    pd.xIsDate, pd.yIsDate = pd.yIsDate, pd.xIsDate


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
