"""Registry for reusable data-processing pipeline actions.

The primary GUI is Qt. These registries remain available for pipeline and
third-party callers, but actions no longer carry frontend-specific editors.
"""

from collections import OrderedDict


def _data_mask(label, mainframe=None):
    from .data_mask import maskAction

    return maskAction(label, mainframe)


def _data_filter(label, mainframe=None):
    from .plotdata_filter import filterAction

    return filterAction(label, mainframe)


def _data_sampler(label, mainframe=None):
    from .plotdata_sampler import samplerAction

    return samplerAction(label, mainframe)


def _data_binning(label, mainframe=None):
    from .plotdata_binning import binningAction

    return binningAction(label, mainframe)


def _data_remove_outliers(label, mainframe=None):
    from .plotdata_removeOutliers import removeOutliersAction

    return removeOutliersAction(label, mainframe)


def _data_standardize_units_si(label, mainframe=None):
    from .data_standardizeUnits import standardizeUnitsAction

    return standardizeUnitsAction(label, mainframe, flavor="SI")


def _data_standardize_units_we(label, mainframe=None):
    from .data_standardizeUnits import standardizeUnitsAction

    return standardizeUnitsAction(label, mainframe, flavor="WE")


def _data_rename_fld_aero(label, mainframe=None):
    from .data_renameFldAero import renameFldAeroAction

    return renameFldAeroAction(label, mainframe)


def _data_rename_of23(label, mainframe=None):
    from .data_renameOF23 import renameOFChannelsAction

    return renameOFChannelsAction(label, mainframe)


def _data_radial_concat(label, mainframe=None):
    from .data_radialConcat import radialConcatAction

    return radialConcatAction(label, mainframe)


def _data_radial_avg(label, mainframe=None):
    from .data_radialavg import radialAvgAction

    return radialAvgAction(label, mainframe)


# Historical names are retained for callers that build pipelines from these maps.
DATA_PLUGINS_WITH_EDITOR = OrderedDict(
    [
        ("Mask", _data_mask),
        ("Remove Outliers", _data_remove_outliers),
        ("Filter", _data_filter),
        ("Resample", _data_sampler),
        ("Bin data", _data_binning),
    ]
)

DATA_PLUGINS_SIMPLE = OrderedDict(
    [
        ("Standardize Units (SI)", _data_standardize_units_si),
        ("Standardize Units (WE)", _data_standardize_units_we),
    ]
)

OF_DATA_PLUGINS_WITH_EDITOR = OrderedDict(
    [
        ("Nodal Average", _data_radial_avg),
    ]
)

OF_DATA_PLUGINS_SIMPLE = OrderedDict(
    [
        ("Nodal Time Concatenation", _data_radial_concat),
        ('v3.4 - Rename "Fld" > "Aero', _data_rename_fld_aero),
        ('v2.3 - Rename "B*N* " > "AB*N* ', _data_rename_of23),
    ]
)

# GUI-only tool panels were part of the removed wx frontend. Their numerical
# implementations remain in pydatview.tools.
TOOLS = OrderedDict()
