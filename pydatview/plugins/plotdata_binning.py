"""Data binning pipeline action."""

import numpy as np

from pydatview.pipeline import PlotDataAction


_DEFAULT_DICT = {
    "active": False,
    "xMin": None,
    "xMax": None,
    "nBins": 50,
}

_imports = [
    "from pydatview.tools.stats import bin_signal",
    "import numpy as np",
]
_data_var = "binData"
_code = (
    "x, y = bin_signal(x, y, "
    "xbins=np.linspace(binData['xMin'], binData['xMax'], binData['nBins']+1))"
)


def binningAction(label="binning", mainframe=None, data=None):
    """Return a reusable binning action."""
    if data is None:
        data = _DEFAULT_DICT.copy()

    return PlotDataAction(
        name=label,
        tableFunctionAdd=binTabAdd,
        plotDataFunction=bin_plot,
        guiCallback=getattr(mainframe, "redraw", None),
        data=data,
        mainframe=mainframe,
        imports=_imports,
        data_var=_data_var,
        code=_code,
    )


def bin_plot(x, y, opts):
    from pydatview.tools.stats import bin_signal

    x_bins = np.linspace(opts["xMin"], opts["xMax"], opts["nBins"] + 1)
    if x_bins[0] > x_bins[1]:
        raise ValueError("xmin must be lower than xmax")
    return bin_signal(x, y, xbins=x_bins)


def bin_tab(tab, iCol, colName, opts, bAdd=True):
    from pydatview.tools.stats import bin_DF

    colName = tab.data.columns[iCol]
    x_bins = np.linspace(opts["xMin"], opts["xMax"], opts["nBins"] + 1)
    df_new = bin_DF(tab.data, xbins=x_bins, colBin=colName)

    if df_new.columns[0].lower().find("index") >= 0:
        df_new = df_new.iloc[:, 1:]

    column_names = list(df_new.columns.values)
    column_names.remove(colName)
    column_names.insert(0, colName)
    df_new = df_new.reindex(columns=column_names)

    if bAdd:
        name_new = tab.raw_name + "_binned"
    else:
        name_new = None
        tab.data = df_new
    return df_new, name_new


def binTabAdd(tab, data):
    return bin_tab(tab, data["icol"], data["colname"], data, bAdd=True)
