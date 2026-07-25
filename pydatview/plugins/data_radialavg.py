"""OpenFAST nodal/radial averaging pipeline action."""

from pydatview.common import PyDatViewException
from pydatview.pipeline import AdderAction


sAVG_METHODS = ["Last `n` seconds", "Last `n` periods"]
AVG_METHODS = ["constantwindow", "periods"]

_DEFAULT_DICT = {
    "active": False,
    "avgMethod": "constantwindow",
    "avgParam": 2,
}

_imports = ["from pydatview.fast.postpro import radialAvg"]
_data_var = "dataRadialAvg"
_code = (
    "dfs_new, names_new = radialAvg(filename, "
    "avgMethod=dataRadialAvg['avgMethod'], "
    "avgParam=dataRadialAvg['avgParam'], df=df, raiseException=False)"
)


def radialAvgAction(label, mainframe=None, data=None):
    """Return a reusable radial averaging action."""
    if data is None:
        data = _DEFAULT_DICT.copy()

    return AdderAction(
        name=label,
        tableFunctionAdd=radialAvg,
        guiCallback=getattr(mainframe, "redraw", None),
        data=data,
        mainframe=mainframe,
        imports=_imports,
        data_var=_data_var,
        code=_code,
    )


def radialAvg(tab, data=None):
    """Return one or more radial-average dataframes and names."""
    from pydatview.fast.postpro import radialAvg as radialAvgPostPro

    dfs_new, names_new = radialAvgPostPro(
        filename=tab.filename,
        df=tab.data,
        avgMethod=data["avgMethod"],
        avgParam=data["avgParam"],
    )
    if all(df is None for df in dfs_new):
        raise PyDatViewException(
            "No OpenFAST radial data found for table: " + tab.nickname
        )
    return dfs_new, names_new
