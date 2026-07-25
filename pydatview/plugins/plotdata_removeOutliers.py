"""Outlier removal pipeline action."""

from pydatview.pipeline import PlotDataAction


_DEFAULT_DICT = {
    "active": False,
    "medianDeviation": 5,
}

_imports = ["from pydatview.tools.signal_analysis import reject_outliers"]
_data_var = "outliersData"
_code = """x, y = reject_outliers(x, y, m=outliersData['medianDeviation'])"""


def removeOutliersAction(label="removeOutlier", mainframe=None, data=None):
    """Return a reusable outlier-removal action."""
    if data is None:
        data = _DEFAULT_DICT.copy()

    return PlotDataAction(
        name=label,
        plotDataFunction=removeOutliersXY,
        guiCallback=getattr(mainframe, "redraw", None),
        data=data,
        mainframe=mainframe,
        imports=_imports,
        data_var=_data_var,
        code=_code,
    )


def removeOutliersXY(x, y, opts):
    from pydatview.tools.signal_analysis import reject_outliers

    try:
        return reject_outliers(y, x, m=opts["medianDeviation"])
    except Exception as exc:
        raise ValueError(
            "Outlier removal failed. Disable it or use a different signal."
        ) from exc
