"""Shared Qt bootstrap for the PySide6 GUI."""

import os
import sys

def _remove_user_site_for_conda_qt():
    if "conda" not in sys.version.lower() and "conda" not in sys.prefix.lower():
        return
    try:
        import site
        user_site = site.getusersitepackages()
    except Exception:
        return
    if not user_site:
        return
    user_site = os.path.abspath(user_site)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != user_site]


_remove_user_site_for_conda_qt()


def _require_qt():
    if sys.platform.startswith("win"):
        os.environ.setdefault(
            "QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough"
        )
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        import pyqtgraph as pg
    except ImportError as exc:
        raise SystemExit(
            "pyDatView Qt requires PySide6 and pyqtgraph.\n"
            "Install them with: pip install PySide6 pyqtgraph"
        ) from exc
    return QtCore, QtGui, QtWidgets, pg


QtCore, QtGui, QtWidgets, pg = _require_qt()
