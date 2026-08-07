import unittest
from unittest.mock import patch


class TestQtTheme(unittest.TestCase):
    def test_windows_stylesheet_has_distinct_interaction_states(self):
        import pydatview.qt_theme as theme

        with patch.object(theme.sys, "platform", "win32"):
            stylesheet = theme.windows_stylesheet()

        self.assertIn("QMenu::item:selected", stylesheet)
        self.assertIn("selection-background-color: #0f6cbd", stylesheet)
        self.assertIn("QTabBar::tab:selected", stylesheet)
        self.assertIn("QPushButton:disabled", stylesheet)
        self.assertIn("QSplitter::handle:hover", stylesheet)

    def test_non_windows_does_not_override_native_styles(self):
        import pydatview.qt_theme as theme

        with patch.object(theme.sys, "platform", "darwin"):
            self.assertEqual(theme.windows_stylesheet(), "")


if __name__ == "__main__":
    unittest.main()
