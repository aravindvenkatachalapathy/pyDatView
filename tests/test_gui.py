import os
import tempfile
import unittest
from collections import deque
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
        self.assertEqual(
            window.font().pointSize(),
            max(7, self.app.font().pointSize() - 1),
        )
        self.assertEqual(
            window.selector_panes[0].table_list_widget.font().pointSize(),
            max(7, window.font().pointSize() - 1),
        )
        self.assertEqual(
            window.selector_panes[0].y_list_widget.font().pointSize(),
            max(7, window.font().pointSize() - 1),
        )
        initial_font_size = window.font().pointSize()
        window.increase_font_action.trigger()
        self.assertEqual(window.font().pointSize(), initial_font_size + 1)
        self.assertEqual(
            window.coordinate_label.font().pointSize(), initial_font_size + 1
        )
        window.decrease_font_action.trigger()
        self.assertEqual(window.font().pointSize(), initial_font_size)
        window.close()
        self.app.processEvents()

    def test_calculation_dialog_has_file_transform_mode(self):
        from pydatview.qt_dialogs import CalculationDialog

        dialog = CalculationDialog(['Index', 'Time_[s]', 'Power_[kW]'])
        dialog.operation_mode.setCurrentIndex(1)

        self.assertEqual(dialog.mode(), 'table')
        self.assertEqual(dialog.result_name.text(), '_trimmed')
        self.assertEqual(
            dialog.expression.toPlainText(),
            'trim(start=0, stop=1)',
        )
        self.assertEqual(dialog.add_button.text(), 'Transform file')
        dialog.close()

    def test_3d_netcdf_slices_open_in_side_by_side_selectors(self):
        import xarray as xr

        from pydatview.qt_main import MainWindow, QtCore

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'vector_field.nc')
            xr.Dataset(
                data_vars={
                    'velocity': (
                        ('time', 'height', 'component'),
                        np.arange(12.0).reshape(3, 2, 2),
                    ),
                },
                coords={
                    'time': [0.0, 1.0, 2.0],
                    'height': [50.0, 100.0],
                    'component': ['u', 'v'],
                },
            ).to_netcdf(path, engine='scipy')

            window = MainWindow()
            window.load_files([path])
            window.redraw_timer.stop()

            self.assertEqual(window.compare_combo.currentText(), '2')
            panes = window.visible_selector_panes()
            self.assertEqual(len(panes), 2)
            selected_names = []
            for pane in panes:
                selected = pane.table_list_widget.selectedItems()
                self.assertEqual(len(selected), 1)
                kind, table_index = selected[0].data(QtCore.Qt.UserRole)
                self.assertEqual(kind, 'table')
                selected_names.append(window.tab_list[table_index].nickname)
                self.assertEqual(pane.x_combo.currentText(), 'time')
                selected_y = pane.y_list_widget.selectedItems()
                self.assertGreaterEqual(len(selected_y), 1)
                self.assertTrue(
                    selected_y[0].text().startswith(
                        'velocity [height='
                    )
                )

            self.assertEqual(
                selected_names,
                ['velocity [component=u]', 'velocity [component=v]'],
            )
            window.close()
            self.app.processEvents()

    def test_standardize_wind_energy_openfast_units(self):
        from unittest.mock import patch

        from pydatview.Tables import Table
        from pydatview.qt_main import (
            MainWindow,
            QtWidgets,
            StandardizeUnitsDialog,
        )

        window = MainWindow()
        window.tab_list.append(Table(
            data=pd.DataFrame({
                "Time [s]": [0.0, 1.0],
                "Torque [Nm]": [2500.0, 5000.0],
                "Pitch [rad]": [0.0, np.pi / 2.0],
                "Power [W]": [1.0e6, 2.0e6],
                "Speed [rad/s]": [0.0, 2.0 * np.pi / 60.0],
            }),
            name="openfast",
            filename="openfast.out",
        ))
        window.populate_tables()

        unit_dialog = StandardizeUnitsDialog(initial_flavor="SI", parent=window)
        self.assertEqual(unit_dialog.target_flavor(), "SI")
        unit_dialog.target_combo.setCurrentIndex(
            unit_dialog.target_combo.findData("WE")
        )
        unit_dialog.apply_button.click()
        self.assertEqual(unit_dialog.result(), QtWidgets.QDialog.Accepted)

        self.assertEqual(
            window.standardize_units_action.text(),
            "Standardize units...",
        )
        with patch("pydatview.qt_main.StandardizeUnitsDialog") as dialog_class:
            dialog = dialog_class.return_value
            dialog.exec.return_value = QtWidgets.QDialog.Accepted
            dialog.target_flavor.return_value = "WE"
            window.standardize_units_action.trigger()
            dialog_class.assert_called_once()
        converted = window.tab_list[0].data
        self.assertIn("Torque [kNm]", converted.columns)
        self.assertIn("Pitch [deg]", converted.columns)
        self.assertIn("Power [kW]", converted.columns)
        self.assertIn("Speed [rpm]", converted.columns)
        np.testing.assert_allclose(converted["Torque [kNm]"], [2.5, 5.0])
        np.testing.assert_allclose(converted["Pitch [deg]"], [0.0, 90.0])
        np.testing.assert_allclose(converted["Power [kW]"], [1000.0, 2000.0])
        np.testing.assert_allclose(converted["Speed [rpm]"], [0.0, 1.0])

        window.close()
        self.app.processEvents()

    def test_plot_hover_coordinates_and_area_zoom(self):
        from pydatview.qt_main import MainWindow, QtCore, pg

        window = MainWindow()
        window.resize(1000, 700)
        plot_data = SimpleNamespace(
            x=np.array([0.0, 2.0, 4.0]),
            y=np.array([0.0, 4.0, 8.0]),
            sx="Time [s]",
            sy="Load [N]",
            syl="Load [N]",
            st="case.out",
            filename="case.out",
            it=0,
            pane_index=0,
        )
        window.canvas.plot_data([plot_data])
        window.canvas.useOpenGL(False)
        window.show()
        self.app.processEvents()

        self.assertEqual(window.canvas.cursor().shape(), QtCore.Qt.CrossCursor)
        view_box = window.canvas._plots[0].getViewBox()
        scene_position = view_box.mapViewToScene(QtCore.QPointF(2.0, 4.0))
        window.canvas._on_mouse_moved((scene_position,))
        self.assertIn("X: 2", window.coordinate_label.text())
        self.assertIn("Y: 4", window.coordinate_label.text())

        window.zoom_area_button.setChecked(True)
        self.assertTrue(window.zoom_area_action.isChecked())
        self.assertEqual(view_box.state["mouseMode"], pg.ViewBox.RectMode)
        window.zoom_area_button.setChecked(False)
        self.assertEqual(view_box.state["mouseMode"], pg.ViewBox.PanMode)
        self.assertEqual(window.canvas._display_axis_value(2.0, True), 100.0)

        window.close()
        self.app.processEvents()

    def test_x_marker_reports_all_curve_intersections(self):
        from pydatview.qt_main import MainWindow

        window = MainWindow()
        curves = [
            SimpleNamespace(
                x=np.array([0.0, 1.0, 2.0]),
                y=np.array([0.0, 2.0, 4.0]),
                sx="Time [s]",
                sy="A [N]",
                syl="case A",
                st="a.out",
                filename="a.out",
                it=0,
                pane_index=0,
            ),
            SimpleNamespace(
                x=np.array([0.0, 2.0]),
                y=np.array([3.0, 7.0]),
                sx="Time [s]",
                sy="B [N]",
                syl="case B",
                st="b.out",
                filename="b.out",
                it=1,
                pane_index=0,
            ),
        ]
        window.canvas.plot_data(curves)
        window.measurement_marker_check.setChecked(True)
        window.canvas.set_measurement_marker(1.5)

        values = {item["label"]: item for item in window.canvas.measurement_values}
        self.assertEqual(set(values), {"A [N]", "B [N]"})
        self.assertAlmostEqual(values["A [N]"]["x"], 1.5)
        self.assertAlmostEqual(values["A [N]"]["y"], 3.0)
        self.assertAlmostEqual(values["B [N]"]["y"], 6.0)
        self.assertGreaterEqual(len(window.canvas._measurement_items), 3)
        line = window.canvas._measurement_items[0][1]
        self.assertEqual(line.pen.color().red(), 198)

        window.measurement_marker_check.setChecked(False)
        self.assertEqual(window.canvas.measurement_values, [])
        window.close()
        self.app.processEvents()

    def test_statistics_copy_and_csv_export(self):
        from unittest.mock import patch

        from pydatview.qt_main import MainWindow, QtWidgets

        window = MainWindow()
        window.stats_table.setColumnCount(2)
        window.stats_table.setHorizontalHeaderLabels(["Series", "Mean"])
        window.stats_table.setRowCount(2)
        for row, values in enumerate((("case, one", "2.5"), ("case two", "4"))):
            for column, value in enumerate(values):
                window.stats_table.setItem(
                    row, column, QtWidgets.QTableWidgetItem(value)
                )

        window.copy_stats()
        copied = QtWidgets.QApplication.clipboard().text()
        self.assertEqual(copied, "Series\tMean\ncase, one\t2.5\ncase two\t4\n")

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "stats.csv")
            with patch(
                "pydatview.qt_tools.QtWidgets.QFileDialog.getSaveFileName",
                return_value=(path, "CSV files (*.csv)"),
            ):
                window.export_stats_csv()
            with open(path, encoding="utf-8", newline="") as stream:
                exported = stream.read()
        self.assertEqual(exported, 'Series,Mean\n"case, one",2.5\ncase two,4\n')
        window.close()
        self.app.processEvents()

    def test_remove_selected_simulation_only_removes_it_from_view(self):
        from pydatview.Tables import Table
        from pydatview.qt_main import MainWindow

        window = MainWindow()
        window.tab_list.append([
            Table(
                data=pd.DataFrame({"x": [0.0], "a": [1.0]}),
                name="first",
                filename="first.out",
            ),
            Table(
                data=pd.DataFrame({"x": [0.0], "b": [2.0]}),
                name="second",
                filename="second.out",
            ),
        ])
        window.populate_tables()
        pane = window.selector_panes[0]
        pane.table_list_widget.clearSelection()
        pane.table_list_widget.item(0).setSelected(True)
        window.remove_selected_sources(pane)

        self.assertEqual(len(window.tab_list), 1)
        self.assertTrue(window.tab_list[0].filename.endswith("second.out"))
        self.assertFalse(any(path.endswith("first.out") for path in window.current_files))
        window.close()
        self.app.processEvents()

    def test_reload_selected_simulation_and_open_location(self):
        from unittest.mock import patch

        from pydatview.qt_main import MainWindow

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "case.csv")
            pd.DataFrame({"Time": [0.0, 1.0], "Load": [1.0, 2.0]}).to_csv(
                path, index=False
            )
            window = MainWindow()
            window.load_files([path])
            pane = window.selector_panes[0]
            self.assertEqual(window.tab_list[0].data["Load"].iloc[-1], 2.0)

            pd.DataFrame({"Time": [0.0, 1.0], "Load": [4.0, 8.0]}).to_csv(
                path, index=False
            )
            window.reload_selected_sources(pane)
            self.assertEqual(window.tab_list[0].data["Load"].iloc[-1], 8.0)

            with patch(
                "pydatview.qt_loading.QtGui.QDesktopServices.openUrl"
            ) as open_url:
                window.open_selected_file_locations(pane)
            open_url.assert_called_once()
            self.assertEqual(open_url.call_args.args[0].toLocalFile(), directory)
            window.close()
            self.app.processEvents()

    def test_compare_plot_mode_and_swap_xy(self):
        from pydatview.Tables import Table
        from pydatview.plotdata import PlotData
        from pydatview.qt_main import (
            MainWindow,
            _COMPARISON_METHODS,
            compare_plot_data,
            swap_plot_axes,
        )

        window = MainWindow()
        plot_modes = [
            window.plot_type_combo.itemText(index)
            for index in range(window.plot_type_combo.count())
        ]
        self.assertIn("Compare", plot_modes)
        methods = [
            window.comparison_method_combo.itemText(index)
            for index in range(window.comparison_method_combo.count())
        ]
        self.assertEqual(methods, list(_COMPARISON_METHODS))
        window.plot_type_combo.setCurrentText("Compare")
        window.redraw_timer.stop()
        self.assertFalse(window.comparison_options_panel.isHidden())

        reference = PlotData(
            x=np.array([0.0, 1.0, 2.0]),
            y=np.array([1.0, 2.0, 4.0]),
            sx="Time [s]",
            sy="Load [N]",
        )
        reference.st = "reference.out"
        reference.filename = "reference.out"
        reference.it = 0
        reference.pane_index = 0
        reference.selection_index = 0
        candidate = PlotData(
            x=np.array([0.0, 1.0, 2.0]),
            y=np.array([2.0, 4.0, 8.0]),
            sx="Time [s]",
            sy="Load [N]",
        )
        candidate.st = "candidate.out"
        candidate.filename = "candidate.out"
        candidate.it = 1
        candidate.pane_index = 1
        candidate.selection_index = 0

        compared = compare_plot_data([reference, candidate], "Relative")
        self.assertEqual(len(compared), 1)
        np.testing.assert_allclose(compared[0].x, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(compared[0].y, [100.0, 100.0, 100.0])
        self.assertEqual(compared[0].sy, "Relative error [%]")
        self.assertEqual(
            compared[0].syl,
            "candidate.out - reference.out | Load",
        )

        swap_plot_axes(compared[0])
        np.testing.assert_allclose(compared[0].x, [100.0, 100.0, 100.0])
        np.testing.assert_allclose(compared[0].y, [0.0, 1.0, 2.0])
        self.assertEqual(compared[0].sx, "Relative error [%]")
        self.assertEqual(compared[0].sy, "Time [s]")
        window.swap_xy_check.setChecked(True)
        window.redraw_timer.stop()
        self.assertTrue(window.swap_xy_check.isChecked())

        window.swap_xy_check.setChecked(False)
        window.comparison_method_combo.setCurrentText("Absolute")
        window.redraw_timer.stop()
        window.tab_list.append([
            Table(
                data=pd.DataFrame({
                    "Time [s]": [0.0, 1.0, 2.0],
                    "Load [N]": [1.0, 2.0, 4.0],
                }),
                name="reference",
                filename="reference.out",
            ),
            Table(
                data=pd.DataFrame({
                    "Time [s]": [0.0, 1.0, 2.0],
                    "Load [N]": [2.0, 4.0, 8.0],
                }),
                name="candidate",
                filename="candidate.out",
            ),
        ])
        window.populate_tables()
        pane = window.selector_panes[0]
        pane.table_list_widget.blockSignals(True)
        for row in range(pane.table_list_widget.count()):
            pane.table_list_widget.item(row).setSelected(True)
        pane.table_list_widget.blockSignals(False)
        window.on_table_selection_changed(pane)
        window.redraw_timer.stop()
        pane.x_combo.setCurrentText("Time [s]")
        pane.y_list_widget.clearSelection()
        for row in range(pane.y_list_widget.count()):
            item = pane.y_list_widget.item(row)
            if item.text() == "Load [N]":
                item.setSelected(True)
        window.redraw_timer.stop()

        integrated = window.build_plot_data()
        self.assertEqual(len(integrated), 1)
        np.testing.assert_allclose(integrated[0].x, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(integrated[0].y, [1.0, 2.0, 4.0])
        self.assertEqual(integrated[0].sy, "Absolute error [N]")

        window.swap_xy_check.setChecked(True)
        window.redraw_timer.stop()
        swapped = window.build_plot_data()
        np.testing.assert_allclose(swapped[0].x, [1.0, 2.0, 4.0])
        np.testing.assert_allclose(swapped[0].y, [0.0, 1.0, 2.0])
        self.assertEqual(swapped[0].sx, "Absolute error [N]")
        self.assertEqual(swapped[0].sy, "Time [s]")

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

    def test_scanned_multi_table_file_plots_only_selected_dataset(self):
        from pydatview.Tables import Table
        from pydatview.qt_main import LazyFileEntry, MainWindow

        window = MainWindow()
        path = os.path.abspath("wind.bts")
        tables = [
            Table(
                data=pd.DataFrame({"z_[m]": [10.0, 20.0], "u_[m/s]": [8.0, 9.0]}),
                name="VertProfile",
                filename=path,
            ),
            Table(
                data=pd.DataFrame({"t_[s]": [0.0, 0.25], "u_[m/s]": [7.0, 10.0]}),
                name="ZMidLine",
                filename=path,
            ),
            Table(
                data=pd.DataFrame({"y_[m]": [-1.0, 1.0], "rho_uu_[-]": [0.5, 1.0]}),
                name="Mid_xcorr_y",
                filename=path,
            ),
        ]
        window.tab_list.append(tables)
        window.lazy_entries = [LazyFileEntry(
            path=path,
            file_format=SimpleNamespace(name="TurbSim binary"),
            table_indices=[0, 1, 2],
            full_loaded=True,
        )]
        window.populate_tables()

        pane = window.selector_panes[0]
        self.assertEqual(pane.bladed_dataset_label.text(), "DATASET")
        self.assertEqual(pane.bladed_dataset_combo.currentText(), "ZMidLine")
        pane.y_list_widget.clearSelection()
        for row in range(pane.y_list_widget.count()):
            item = pane.y_list_widget.item(row)
            if item.text() == "u_[m/s]":
                item.setSelected(True)
        plot_data = window.build_plot_data()
        self.assertEqual(len(plot_data), 1)
        np.testing.assert_allclose(plot_data[0].x, [0.0, 0.25])
        np.testing.assert_allclose(plot_data[0].y, [7.0, 10.0])

        pane.bladed_dataset_combo.setCurrentIndex(
            pane.bladed_dataset_combo.findData("VertProfile")
        )
        pane.y_list_widget.clearSelection()
        for row in range(pane.y_list_widget.count()):
            item = pane.y_list_widget.item(row)
            if item.text() == "u_[m/s]":
                item.setSelected(True)
        plot_data = window.build_plot_data()
        self.assertEqual(len(plot_data), 1)
        np.testing.assert_allclose(plot_data[0].x, [10.0, 20.0])
        np.testing.assert_allclose(plot_data[0].y, [8.0, 9.0])

        window.close()
        self.app.processEvents()

    def test_scan_append_keeps_loaded_entries_and_selection(self):
        from pydatview.Tables import Table
        from pydatview.qt_main import MainWindow

        with tempfile.TemporaryDirectory() as temp_dir:
            first_path = os.path.join(temp_dir, "first.out")
            second_path = os.path.join(temp_dir, "second.out")
            for path in (first_path, second_path):
                with open(path, "w", encoding="ascii"):
                    pass
            file_format = SimpleNamespace(name="FAST output file")
            window = MainWindow()
            window.set_lazy_file_index([(first_path, file_format)])
            old_entry = window.lazy_entries[0]
            old_table = Table(
                data=pd.DataFrame({"Time [s]": [0.0, 1.0], "Load [N]": [1.0, 2.0]}),
                name="first",
                filename=first_path,
            )
            window.tab_list.append(old_table)
            old_entry.table_indices = [0]
            old_entry.full_loaded = True
            window.lazy_loaded_total = 1
            window.populate_tables()

            added = window.set_lazy_file_index(
                [(first_path, file_format), (second_path, file_format)],
                append=True,
            )

            self.assertEqual(added, 1)
            self.assertEqual(len(window.lazy_entries), 2)
            self.assertIs(window.lazy_entries[0], old_entry)
            self.assertIs(window.tab_list[0], old_table)
            self.assertEqual(window.lazy_loaded_count(), 1)
            self.assertEqual(window.selected_lazy_indices(), [0])
            window.close()
            self.app.processEvents()

    def test_bladed_batches_use_format_specific_worker_cap(self):
        from pydatview.qt_main import LazyFileEntry, MainWindow

        window = MainWindow()
        file_format = SimpleNamespace(name="Bladed output file")
        window.lazy_entries = [
            LazyFileEntry("case_{}.$PJ".format(index), file_format)
            for index in range(6)
        ]
        window.lazy_load_queue = deque((index, None) for index in range(6))
        window.lazy_max_workers = 96
        window.bladed_worker_cap = 2

        self.assertEqual(window.effective_lazy_worker_limit(), 2)
        window.close()
        self.app.processEvents()

    def test_low_memory_rejects_bladed_load_instead_of_starting_worker(self):
        from pydatview.qt_main import LazyFileEntry, MainWindow

        window = MainWindow()
        file_format = SimpleNamespace(name="Bladed output file")
        entry = LazyFileEntry(
            "large_case.$PJ",
            file_format,
            estimated_load_bytes=2 * 1024 ** 3,
            loading=True,
        )
        window.lazy_entries = [entry]
        window.lazy_load_queue = deque([(0, None)])
        window.available_memory_bytes = lambda: 1024 ** 3
        window.begin_lazy_load_batch(1)

        window.start_next_lazy_load()
        window.finish_lazy_load_batch_if_done()

        self.assertFalse(window.lazy_load_queue)
        self.assertFalse(window.lazy_loader_threads)
        self.assertTrue(entry.attempted)
        self.assertIn("Not enough available memory", entry.warning)
        self.assertFalse(window.loading_progress.isVisible())
        window.close()
        self.app.processEvents()

    def test_stats_table_supports_multiple_del_slopes(self):
        from pydatview.Tables import Table
        from pydatview.qt_main import MainWindow
        from pydatview.tools.fatigue import equivalent_load

        window = MainWindow()
        time = np.arange(0.0, 20.0, 0.05)
        load = 5.0 * np.sin(2.0 * np.pi * time)
        window.tab_list.append(Table(
            data=pd.DataFrame({"Time [s]": time, "Load [N]": load}),
            name="fatigue",
            filename="fatigue.out",
        ))
        window.populate_tables()
        pane = window.selector_panes[0]
        pane.y_list_widget.clearSelection()
        for row in range(pane.y_list_widget.count()):
            item = pane.y_list_widget.item(row)
            if item.text() == "Load [N]":
                item.setSelected(True)
        for slope, action in window.del_slope_actions.items():
            action.blockSignals(True)
            action.setChecked(slope in (2, 4))
            action.blockSignals(False)
        for key, action in window.stats_column_actions.items():
            action.blockSignals(True)
            action.setChecked(key in {
                "series", "file", "n", "dt", "mean", "std", "min", "max",
            })
            action.blockSignals(False)
        window.update_stats_columns_button()
        window.update_del_slopes_button()
        window.plot_data = window.build_plot_data()
        window.plot_data[0].syl = "fatigue.out - Load [N]"
        window.update_stats()

        headers = [
            window.stats_table.horizontalHeaderItem(column).text()
            for column in range(window.stats_table.columnCount())
        ]
        self.assertEqual(window.stats_table.rowCount(), 1)
        self.assertEqual(
            window.stats_table.item(0, headers.index("Series")).text(),
            "Load [N]",
        )
        self.assertEqual(
            window.stats_table.item(0, headers.index("Filename")).text(),
            "fatigue.out",
        )
        self.assertAlmostEqual(
            float(window.stats_table.item(0, headers.index("dt")).text()),
            0.05,
        )
        self.assertIn("DEL m=2 (1 Hz)", headers)
        self.assertIn("DEL m=4 (1 Hz)", headers)
        for slope in (2, 4):
            actual = float(
                window.stats_table.item(
                    0,
                    headers.index("DEL m={} (1 Hz)".format(slope)),
                ).text()
            )
            expected = equivalent_load(
                time,
                load,
                m=slope,
                Teq=1,
                bins=100,
                method="rainflow_windap",
            )
            self.assertAlmostEqual(actual, expected, places=4)
        window.close()
        self.app.processEvents()

    def test_stats_columns_are_selectable_and_match_legacy_calculations(self):
        from pydatview.Tables import Table
        from pydatview.qt_main import MainWindow

        window = MainWindow()
        x = np.array([0.0, 1.0, 2.0, 4.0])
        y = np.array([2.0, 4.0, 1.0, 5.0])
        window.tab_list.append(Table(
            data=pd.DataFrame({"Time [s]": x, "Load [N]": y}),
            name="statistics",
            filename="stats.out",
        ))
        window.populate_tables()
        pane = window.selector_panes[0]
        pane.y_list_widget.clearSelection()
        for row in range(pane.y_list_widget.count()):
            item = pane.y_list_widget.item(row)
            if item.text() == "Load [N]":
                item.setSelected(True)

        selected = {
            "series", "file", "directory", "table", "n", "dt", "median",
            "mean", "std", "var", "std_mean", "min", "max", "x_at_min",
            "x_at_max", "abs_max", "range", "x_min", "x_max", "x_range",
            "integral", "integral_mean", "integral_x",
            "integral_centroid", "integral_x2",
        }
        for key, action in window.stats_column_actions.items():
            action.blockSignals(True)
            action.setChecked(key in selected)
            action.blockSignals(False)
        for action in window.del_slope_actions.values():
            action.blockSignals(True)
            action.setChecked(False)
            action.blockSignals(False)
        window.update_stats_columns_button()
        window.update_del_slopes_button()
        window.plot_data = window.build_plot_data()
        window.plot_data[0].syl = "stats.out - Load [N]"
        window.update_stats()

        headers = [
            window.stats_table.horizontalHeaderItem(column).text()
            for column in range(window.stats_table.columnCount())
        ]
        values = {
            header: window.stats_table.item(0, column).text()
            for column, header in enumerate(headers)
        }
        self.assertEqual(values["Series"], "Load [N]")
        self.assertEqual(values["Filename"], "stats.out")
        self.assertEqual(values["n"], "4")
        expected = {
            "dt": 1.0,
            "Median": 3.0,
            "Mean": 3.0,
            "Std": np.sqrt(2.5),
            "Var": 2.5,
            "Std/Mean (TI)": np.sqrt(2.5) / 3.0,
            "Min": 1.0,
            "Max": 5.0,
            "x@Min": 2.0,
            "x@Max": 4.0,
            "Abs. Max": 5.0,
            "Range": 4.0,
            "xMin": 0.0,
            "xMax": 4.0,
            "xRange": 4.0,
            "Integral y dx": 11.5,
            "Integral y dx / Integral dx": 2.875,
            "Integral y*x dx": 27.0,
            "Integral y*x dx / Integral y dx": 27.0 / 11.5,
            "Integral y*x^2 dx": 90.0,
        }
        for header, expected_value in expected.items():
            self.assertAlmostEqual(
                float(values[header]), expected_value, places=5, msg=header
            )

        window.close()
        self.app.processEvents()

    def test_fft_controls_apply_amplitude_without_averaging(self):
        from pydatview.Tables import Table
        from pydatview.qt_main import MainWindow, NumericAxisItem

        window = MainWindow()
        time = np.arange(0.0, 20.0, 0.05)
        load = 5.0 * np.sin(2.0 * np.pi * time)
        window.tab_list.append(Table(
            data=pd.DataFrame({"Time [s]": time, "Load [N]": load}),
            name="fft",
            filename="fft.out",
        ))
        window.populate_tables()
        pane = window.selector_panes[0]
        pane.y_list_widget.clearSelection()
        for row in range(pane.y_list_widget.count()):
            item = pane.y_list_widget.item(row)
            if item.text() == "Load [N]":
                item.setSelected(True)
        window.plot_type_combo.setCurrentText("FFT")
        window.redraw_timer.stop()
        window.fft_output_combo.setCurrentText("Amplitude")
        window.fft_averaging_combo.setCurrentText("None")
        window.fft_x_combo.setCurrentIndex(window.fft_x_combo.findData("1/x"))
        window.fft_detrend_check.setChecked(False)
        window.redraw_timer.stop()

        plot_data = window.build_plot_data()
        self.assertFalse(window.fft_options_panel.isHidden())
        self.assertTrue(window.logy_check.isChecked())
        axis = NumericAxisItem(orientation="left")
        axis.logMode = True
        self.assertEqual(
            axis.tickStrings([-3.0, -2.0, -1.5, 0.0, 2.0], 1.0, 1.0),
            ["10^-3", "10^-2", "", "10^0", "10^2"],
        )
        self.assertEqual(len(plot_data), 1)
        peak = int(np.argmax(plot_data[0].y))
        self.assertAlmostEqual(plot_data[0].x[peak], 1.0, places=6)
        self.assertAlmostEqual(plot_data[0].y[peak], 5.0, places=6)
        window.plot_type_combo.setCurrentText("Regular")
        self.assertFalse(window.logy_check.isChecked())
        window.close()
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
