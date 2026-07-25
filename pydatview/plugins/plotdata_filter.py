"""Signal filtering pipeline action."""

from pydatview.pipeline import PlotDataAction


_DEFAULT_DICT = {
    "active": False,
    "name": "Moving average",
    "param": 100,
    "paramName": "Window Size",
    "paramRange": [1, 100000],
    "increment": 1,
    "digits": 0,
}

_imports = ["from pydatview.tools.signal_analysis import applyFilter"]
_data_var = "filterData"
_code = """y = applyFilter(x, y, filterData)"""


def filterAction(label="filter", mainframe=None, data=None):
    """Return a reusable filtering action."""
    if data is None:
        data = _DEFAULT_DICT.copy()

    return PlotDataAction(
        name=label,
        tableFunctionAdd=filterTabAdd,
        plotDataFunction=filterXY,
        guiCallback=getattr(mainframe, "redraw", None),
        data=data,
        mainframe=mainframe,
        imports=_imports,
        data_var=_data_var,
        code=_code,
    )


def filterXY(x, y, opts):
    """Apply a configured filter to x/y arrays."""
    from pydatview.tools.signal_analysis import applyFilter

    return x, applyFilter(x, y, opts)


def filterTabAdd(tab, opts):
    """Return a filtered copy of a table and its generated name."""
    return tab.applyFiltering(opts["icol"], opts, bAdd=True)
