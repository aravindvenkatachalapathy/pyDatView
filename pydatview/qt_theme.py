"""Windows-specific Qt palette and widget-style normalization."""

import sys

from pydatview.qt_compat import QtGui, QtWidgets


_WINDOWS_LIGHT_STYLESHEET = r"""
    QMainWindow, QDialog, QWidget#appBackground,
    QWidget#selectorArea {
        background: #eef1f4;
    }
    QFrame#plotControls {
        background: #ffffff;
        border: 1px solid #9ca8b4;
        border-radius: 4px;
    }
    QLabel[sectionLabel="true"] {
        color: #243342;
        background: #e7ebef;
        border-color: #b7c0c9;
        border-radius: 2px;
        padding: 3px 5px;
    }
    QLabel#statusChip {
        color: #0b4f8a;
        background: #e5f0fb;
        border-color: #7ea9d1;
    }
    QLabel#coordinateReadout {
        background: #f7f9fb;
        border-left-color: #a7b1bb;
    }
    QMenuBar {
        background: #f7f8fa;
        border-bottom: 1px solid #9ca8b4;
        padding: 1px 3px;
    }
    QMenuBar::item {
        padding: 5px 10px;
        border-radius: 2px;
    }
    QMenuBar::item:selected,
    QMenuBar::item:pressed {
        color: #ffffff;
        background: #0f6cbd;
        border-color: #0f6cbd;
    }
    QMenu {
        color: #17212b;
        background: #ffffff;
        border: 1px solid #8794a1;
        padding: 4px 2px;
    }
    QMenu::item {
        padding: 6px 32px 6px 24px;
        border: 0;
        border-radius: 2px;
        margin: 1px 3px;
    }
    QMenu::item:selected {
        color: #ffffff;
        background: #0f6cbd;
        border: 0;
    }
    QMenu::item:disabled {
        color: #87929d;
    }
    QMenu::separator {
        height: 1px;
        background: #d2d8de;
        margin: 4px 8px;
    }
    QToolBar {
        background: #f7f8fa;
        border-color: #9ca8b4;
        spacing: 4px;
        padding: 4px 6px;
    }
    QToolButton {
        background: #ffffff;
        border-color: #a6b0ba;
        border-radius: 3px;
    }
    QToolButton:hover {
        color: #0b4f8a;
        background: #e5f0fb;
        border-color: #3b82c4;
    }
    QToolButton:pressed,
    QToolButton:checked {
        background: #cee4f7;
        border-color: #0f6cbd;
    }
    QGroupBox[selectorPane="true"] {
        background: #ffffff;
        border-color: #98a5b2;
        border-radius: 4px;
    }
    QGroupBox[selectorPane="true"]::title {
        color: #0b4f8a;
        background: #ffffff;
    }
    QListWidget, QTableView, QTableWidget, QPlainTextEdit,
    QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
        color: #17212b;
        background: #ffffff;
        border: 1px solid #a4afb9;
        border-radius: 3px;
        alternate-background-color: #f1f4f7;
        selection-background-color: #0f6cbd;
        selection-color: #ffffff;
    }
    QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
        min-height: 26px;
        padding: 1px 6px;
    }
    QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus,
    QSpinBox:focus, QListWidget:focus, QTableView:focus,
    QTableWidget:focus, QPlainTextEdit:focus {
        border: 1px solid #0f6cbd;
    }
    QComboBox::drop-down {
        width: 24px;
        border: 0;
        border-left: 1px solid #c4ccd4;
        background: #f4f6f8;
    }
    QComboBox QAbstractItemView {
        color: #17212b;
        background: #ffffff;
        border: 1px solid #8794a1;
        selection-background-color: #0f6cbd;
        selection-color: #ffffff;
        outline: 0;
    }
    QListWidget::item {
        padding: 4px 6px;
        border-radius: 2px;
    }
    QListWidget::item:selected,
    QListWidget::item:selected:active,
    QListWidget::item:selected:!active,
    QTableView::item:selected,
    QTableView::item:selected:active,
    QTableView::item:selected:!active,
    QTableWidget::item:selected {
        color: #ffffff;
        background: #0f6cbd;
        border-color: #0f6cbd;
    }
    QListWidget::item:hover:!selected {
        background: #e5f0fb;
        border-color: #91b8db;
    }
    QPushButton {
        color: #17212b;
        background: #f7f9fb;
        border: 1px solid #9eabb7;
        border-radius: 3px;
        min-height: 26px;
        padding: 3px 11px;
    }
    QPushButton:hover {
        color: #0b4f8a;
        background: #e5f0fb;
        border-color: #3b82c4;
    }
    QPushButton:pressed {
        background: #cee4f7;
        border-color: #0f6cbd;
    }
    QPushButton#primaryButton,
    QPushButton#zoomAreaButton:checked {
        color: #ffffff;
        background: #0f6cbd;
        border-color: #0b5799;
    }
    QPushButton#primaryButton:hover {
        background: #0b5da5;
    }
    QPushButton:disabled, QToolButton:disabled,
    QComboBox:disabled, QLineEdit:disabled,
    QDoubleSpinBox:disabled, QSpinBox:disabled {
        color: #8a949e;
        background: #e7eaed;
        border-color: #c2c9cf;
    }
    QTabWidget::pane {
        border-color: #98a5b2;
    }
    QTabBar::tab {
        color: #344454;
        background: #dce2e8;
        border-color: #9eabb7;
        border-radius: 0;
    }
    QTabBar::tab:selected {
        color: #0b4f8a;
        background: #ffffff;
        border-top: 2px solid #0f6cbd;
        border-bottom-color: #ffffff;
    }
    QTabBar::tab:hover:!selected {
        background: #e8eef4;
    }
    QHeaderView::section {
        color: #243342;
        background: #dde3e9;
        border-right-color: #b5bec7;
        border-bottom-color: #8f9ba7;
        padding: 5px 7px;
    }
    QProgressBar {
        border-color: #9eabb7;
        background: #ffffff;
    }
    QProgressBar::chunk {
        background: #0f6cbd;
    }
    QSplitter::handle {
        background: #aeb8c2;
    }
    QSplitter::handle:hover {
        background: #0f6cbd;
    }
    QSplitter::handle:horizontal {
        width: 4px;
    }
    QSplitter::handle:vertical {
        height: 4px;
    }
    QScrollBar:vertical {
        background: #eef1f4;
        width: 14px;
    }
    QScrollBar:horizontal {
        background: #eef1f4;
        height: 14px;
    }
    QScrollBar::handle:vertical,
    QScrollBar::handle:horizontal {
        background: #99a6b2;
        border-radius: 4px;
        margin: 2px;
    }
    QScrollBar::handle:vertical:hover,
    QScrollBar::handle:horizontal:hover {
        background: #748493;
    }
    QStatusBar {
        background: #f7f8fa;
        border-top-color: #9ca8b4;
    }
"""


def configure_application(app):
    """Normalize Windows metrics while retaining native macOS styling."""
    if app is None or app.property("pydatviewThemeConfigured"):
        return
    if sys.platform.startswith("win"):
        fusion = QtWidgets.QStyleFactory.create("Fusion")
        if fusion is not None:
            app.setStyle(fusion)

        palette = app.style().standardPalette()
        colors = {
            QtGui.QPalette.Window: "#eef1f4",
            QtGui.QPalette.WindowText: "#17212b",
            QtGui.QPalette.Base: "#ffffff",
            QtGui.QPalette.AlternateBase: "#f1f4f7",
            QtGui.QPalette.ToolTipBase: "#263442",
            QtGui.QPalette.ToolTipText: "#ffffff",
            QtGui.QPalette.Text: "#17212b",
            QtGui.QPalette.Button: "#f7f9fb",
            QtGui.QPalette.ButtonText: "#17212b",
            QtGui.QPalette.BrightText: "#ffffff",
            QtGui.QPalette.Highlight: "#0f6cbd",
            QtGui.QPalette.HighlightedText: "#ffffff",
            QtGui.QPalette.PlaceholderText: "#75818d",
            QtGui.QPalette.Light: "#ffffff",
            QtGui.QPalette.Midlight: "#d8dde3",
            QtGui.QPalette.Mid: "#aab4be",
            QtGui.QPalette.Dark: "#677583",
            QtGui.QPalette.Shadow: "#263442",
            QtGui.QPalette.Link: "#0f6cbd",
        }
        for role, color in colors.items():
            palette.setColor(role, QtGui.QColor(color))
        for role in (QtGui.QPalette.Text, QtGui.QPalette.ButtonText):
            palette.setColor(
                QtGui.QPalette.Disabled,
                role,
                QtGui.QColor("#87929d"),
            )
        app.setPalette(palette)

        if "Segoe UI" in set(QtGui.QFontDatabase.families()):
            font = QtGui.QFont("Segoe UI")
            font.setPointSize(10)
            app.setFont(font)
    app.setProperty("pydatviewThemeConfigured", True)


def windows_stylesheet():
    if sys.platform.startswith("win"):
        return _WINDOWS_LIGHT_STYLESHEET
    return ""
