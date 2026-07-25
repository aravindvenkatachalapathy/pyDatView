"""Table masking pipeline action."""

import numpy as np
import pandas as pd

from pydatview.pipeline import ReversibleTableAction


_DEFAULT_DICT = {
    "active": False,
    "maskString": "",
    "formattedMaskString": "",
}

_imports = []
_data_var = "maskData"
_code = """df = df[eval(maskData['formattedMaskString'])]"""


def maskAction(label="mask", mainframe=None, data=None):
    """Return a reusable table masking action."""
    if data is None:
        data = _DEFAULT_DICT.copy()

    return ReversibleTableAction(
        name=label,
        tableFunctionAdd=addTabMask,
        tableFunctionApply=applyMask,
        tableFunctionCancel=removeMask,
        guiCallback=getattr(mainframe, "redraw", None),
        data=data,
        mainframe=mainframe,
        imports=_imports,
        data_var=_data_var,
        code=_code,
    )


def applyMask(tab, data):
    formatted_mask = formatMaskString(tab.data, data["maskString"])
    tab.applyMaskString(formatted_mask, bAdd=False)
    data["formattedMaskString"] = formatted_mask


def removeMask(tab, data):
    tab.clearMask()


def addTabMask(tab, opts):
    """Return a masked copy of a table and its generated name."""
    opts["formattedMaskString"] = formatMaskString(tab.data, opts["maskString"])
    return tab.applyMaskString(opts["formattedMaskString"], bAdd=True)


def formatMaskString(df, mask):
    """Replace ``{column}`` references with dataframe expressions."""
    from pydatview.common import no_unit

    for index, column in enumerate(df.columns):
        column_without_unit = no_unit(column).strip()
        reference = "{" + column_without_unit + "}"
        if isinstance(df.iloc[0, index], pd.Timestamp):
            replacement = "df[{!r}]".format(column)
        else:
            replacement = "np.asarray(df[{!r}])".format(column)
        mask = mask.replace(reference, replacement)
    return mask
