import os
import tempfile
import unittest

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TestGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pydatview.qt_main import QtWidgets

        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_qt_main_window(self):
        from pydatview.main import MainFrame
        from pydatview.qt_main import MainWindow

        self.assertIs(MainFrame, MainWindow)
        window = MainWindow()
        self.assertTrue(window.windowTitle())
        window.close()
        self.app.processEvents()

    def test_scanned_fast_plot_uses_partial_channel_cache(self):
        import pydatview.io as weio
        from pydatview.Tables import TableList
        from pydatview.io.fast_output_file import writeBinary
        from pydatview.qt_main import MainWindow

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "partial.outb")
            time = np.linspace(0.0, 2.0, 101)
            data = np.column_stack((time, np.sin(time), np.cos(time)))
            writeBinary(
                path,
                data,
                ["Time", "Sin", "Cos"],
                ["s", "-", "-"],
                fileID=4,
            )
            file_format = next(
                fmt for fmt in weio.fileFormats()
                if fmt.name == "FAST output file"
            )

            window = MainWindow()
            window.set_lazy_file_index([(path, file_format)])
            pane = window.selector_panes[0]
            pane.y_list_widget.clearSelection()
            for row in range(pane.y_list_widget.count()):
                item = pane.y_list_widget.item(row)
                if item.text() == "Cos_[-]":
                    item.setSelected(True)
                    break

            requests = window.lazy_plot_column_requests()
            self.assertEqual(requests, {0: (0, 2)})

            loader = TableList(options=window.tab_list.options)
            tabs, warning = loader._load_file_tabs(
                path,
                fileformat=file_format,
                channel_indices=requests[0],
            )
            self.assertFalse(warning)
            window.on_lazy_load_finished(
                window.lazy_generation,
                0,
                tabs,
                "",
                0.0,
                file_format.name,
                list(requests[0]),
            )
            window.redraw_timer.stop()

            entry = window.lazy_entries[0]
            self.assertTrue(entry.loaded)
            self.assertFalse(entry.full_loaded)
            self.assertEqual(entry.loaded_column_indices, {0, 2})
            self.assertEqual(
                list(window.tab_list[entry.table_indices[0]].data.columns),
                ["Index", "Time_[s]", "Cos_[-]"],
            )
            plot_data = window.build_plot_data()
            self.assertEqual(len(plot_data), 1)
            np.testing.assert_allclose(
                plot_data[0].y,
                np.cos(time),
                rtol=2e-4,
                atol=3e-5,
            )

            for row in range(pane.y_list_widget.count()):
                item = pane.y_list_widget.item(row)
                if item.text() in ("Sin_[-]", "Cos_[-]"):
                    item.setSelected(True)
            requests = window.lazy_plot_column_requests()
            self.assertEqual(requests, {0: (0, 1, 2)})
            expanded_tabs, warning = loader._load_file_tabs(
                path,
                fileformat=file_format,
                channel_indices=requests[0],
            )
            self.assertFalse(warning)
            window.on_lazy_load_finished(
                window.lazy_generation,
                0,
                expanded_tabs,
                "",
                0.0,
                file_format.name,
                list(requests[0]),
            )
            self.assertEqual(entry.loaded_column_indices, {0, 1, 2})
            self.assertFalse(entry.full_loaded)
            self.assertEqual(
                list(window.tab_list[entry.table_indices[0]].data.columns),
                ["Index", "Time_[s]", "Sin_[-]", "Cos_[-]"],
            )
            self.assertEqual(len(window.build_plot_data()), 2)

            full_tabs, warning = loader._load_file_tabs(
                path,
                fileformat=file_format,
            )
            self.assertFalse(warning)
            window.on_lazy_load_finished(
                window.lazy_generation,
                0,
                full_tabs,
                "",
                0.0,
                file_format.name,
                None,
            )
            self.assertTrue(entry.full_loaded)
            self.assertEqual(
                list(window.tab_list[entry.table_indices[0]].data.columns),
                ["Index", "Time_[s]", "Sin_[-]", "Cos_[-]"],
            )
            window.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
