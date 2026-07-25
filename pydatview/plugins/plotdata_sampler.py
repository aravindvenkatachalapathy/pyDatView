"""Signal resampling pipeline action."""

from pydatview.pipeline import PlotDataAction


_DEFAULT_DICT = {
    "active": False,
    "name": "Every n",
    "param": 2,
    "paramName": "n",
}

_imports = ["from pydatview.tools.signal_analysis import applySampler"]
_data_var = "samplerData"
_code = """x, y = applySampler(x, y, samplerData)"""


def samplerAction(label="sampler", mainframe=None, data=None):
    """Return a reusable resampling action."""
    if data is None:
        data = _DEFAULT_DICT.copy()

    return PlotDataAction(
        name=label,
        tableFunctionAdd=samplerTabAdd,
        plotDataFunction=samplerXY,
        guiCallback=getattr(mainframe, "redraw", None),
        data=data,
        mainframe=mainframe,
        imports=_imports,
        data_var=_data_var,
        code=_code,
    )


def samplerXY(x, y, opts):
    from pydatview.tools.signal_analysis import applySampler

    return applySampler(x, y, opts)


def samplerTabAdd(tab, opts):
    """Return a resampled copy of a table and its generated name."""
    return tab.applyResampling(opts["icol"], opts, bAdd=True)
