"""Compatibility entry points for the PySide6 frontend.

Imports are retained so existing launch scripts continue to open the Qt GUI.
"""

from pydatview.qt_main import MainWindow, cmdline, showApp


MainFrame = MainWindow

__all__ = ["MainWindow", "MainFrame", "showApp", "main", "cmdline"]


def main(inputfiles=None):
    return showApp(filenames=inputfiles or [])


if __name__ == "__main__":
    raise SystemExit(cmdline())
