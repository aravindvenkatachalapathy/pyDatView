import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd


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

    @staticmethod
    def _bladed_tables(project_path, offset=0.0):
        from pydatview.Tables import Table

        time = np.linspace(0.0, 1.0, 5)
        return [
            Table(
                data=pd.DataFrame({
                    "Time [s]": time,
                    "Nominal pitch angle [rad]": time + offset,
                }),
                name="Control variables",
                filename=project_path,
            ),
            Table(
                data=pd.DataFrame({
                    "Time [s]": time,
                    "Generator speed [rad/s]": 10.0 * time + offset,
                }),
                name="Drivetrain variables",
                filename=project_path,
            ),
        ]

    def test_direct_bladed_project_is_one_file_with_all_variables(self):
        from pydatview.qt_main import MainWindow

        window = MainWindow()
        project_path = os.path.abspath("single_case.$PJ")
        window.tab_list.append(self._bladed_tables(project_path))
        window.populate_tables()

        pane = window.selector_panes[0]
        self.assertEqual(pane.table_list_widget.count(), 1)
        self.assertEqual(pane.table_list_widget.item(0).text(), "single_case.$PJ")
        self.assertEqual(
            pane.bladed_dataset_combo.currentText(),
            "All variable groups",
        )
        displayed_columns = [
            pane.y_list_widget.item(row).text()
            for row in range(pane.y_list_widget.count())
        ]
        self.assertIn("Nominal pitch angle [rad]", displayed_columns)
        self.assertIn("Generator speed [rad/s]", displayed_columns)

        pane.y_list_widget.clearSelection()
        for row in range(pane.y_list_widget.count()):
            item = pane.y_list_widget.item(row)
            if item.text() == "Nominal pitch angle [rad]":
                item.setSelected(True)
        plot_data = window.build_plot_data()
        self.assertEqual(len(plot_data), 1)
        self.assertEqual(plot_data[0].st, "single_case.$PJ")
        self.assertEqual(plot_data[0].sy, "Nominal pitch angle [rad]")

        window.close()
        self.app.processEvents()

    def test_scanned_bladed_projects_union_groups_and_plot_once_per_file(self):
        from pydatview.qt_main import LazyFileEntry, MainWindow

        window = MainWindow()
        paths = [os.path.abspath("case_a.$PJ"), os.path.abspath("case_b.$PJ")]
        file_format = SimpleNamespace(name="Bladed output file")
        for path_index, path in enumerate(paths):
            start = len(window.tab_list)
            tabs = self._bladed_tables(path, offset=float(path_index))
            window.tab_list.append(tabs)
            window.lazy_entries.append(LazyFileEntry(
                path=path,
                file_format=file_format,
                table_indices=list(range(start, start + len(tabs))),
                full_loaded=True,
            ))
        window.populate_tables()

        pane = window.selector_panes[0]
        pane.table_list_widget.blockSignals(True)
        for row in range(pane.table_list_widget.count()):
            pane.table_list_widget.item(row).setSelected(True)
        pane.table_list_widget.blockSignals(False)
        window.on_table_selection_changed(pane)

        displayed_columns = [
            pane.y_list_widget.item(row).text()
            for row in range(pane.y_list_widget.count())
        ]
        self.assertIn("Nominal pitch angle [rad]", displayed_columns)
        self.assertIn("Generator speed [rad/s]", displayed_columns)

        pane.y_list_widget.clearSelection()
        for row in range(pane.y_list_widget.count()):
            item = pane.y_list_widget.item(row)
            if item.text() == "Nominal pitch angle [rad]":
                item.setSelected(True)
        plot_data = window.build_plot_data()
        self.assertEqual(len(plot_data), 2)
        self.assertEqual({pd.st for pd in plot_data}, {"case_a.$PJ", "case_b.$PJ"})

        window.close()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
