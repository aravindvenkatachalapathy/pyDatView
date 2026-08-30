"""Calculation, units, statistics, and export workflows for the Qt window."""

import csv
import io
import os
import time
import traceback

import numpy as np

from pydatview.qt_compat import QtCore, QtGui, QtWidgets
from pydatview.qt_dialogs import (
    AxisLimitsDialog,
    CalculationDialog,
)
from pydatview.qt_math import (
    evaluate_math_expression,
    transform_file_tables,
)
from pydatview.qt_publication import (
    PublicationExportDialog,
    export_publication_plot,
)
from pydatview.qt_stats import (
    _DEFAULT_STATS_COLUMNS,
    _STATS_COLUMNS,
    _equivalent_loads,
    _finite_xy,
    _series_statistics,
)


class QtToolsStatsMixin:
    def open_calculation_dialog(self):
        panes = self.visible_selector_panes()
        if not panes:
            return
        pane = self.active_selector_pane if self.active_selector_pane in panes else panes[0]
        unloaded = [
            lazy_index for lazy_index in self.selected_lazy_indices(pane)
            if not self.lazy_entries[lazy_index].full_loaded
        ]
        if unloaded:
            self.statusBar().showMessage(
                "Use Load full selected before creating a calculated variable",
                10000,
            )
            return

        table_indices = self.selected_table_indices(load=False, pane=pane)
        if len(table_indices) != 1:
            QtWidgets.QMessageBox.information(
                self,
                "Mathematical operation",
                "Select one loaded table or one Bladed variable group.",
            )
            return

        table_index = table_indices[0]
        tab = self.tab_list[table_index]
        selected_columns = [
            str(tab.columns[index])
            for index in self.selected_y_indices_original(pane)
            if isinstance(index, int) and 0 <= index < len(tab.columns)
        ]
        dialog = CalculationDialog(
            tab.columns,
            selected_columns=selected_columns,
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return

        result_name, expression = dialog.values()
        if dialog.mode() == "table":
            self.apply_table_transform_to_file(
                table_index,
                result_name,
                expression,
                pane,
            )
            return
        if result_name in [str(column) for column in tab.data.columns]:
            QtWidgets.QMessageBox.warning(
                self,
                "Mathematical operation",
                "A variable named '{}' already exists.".format(result_name),
            )
            return

        try:
            result = evaluate_math_expression(tab.data, expression)
            tab.addColumn(
                result_name,
                result,
                i=len(tab.data.columns) - 1,
                sFormula=expression,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Mathematical operation",
                "{}: {}".format(type(exc).__name__, exc),
            )
            return

        new_column_index = len(tab.data.columns) - 1
        pane.column_filter.blockSignals(True)
        pane.column_filter.clear()
        pane.column_filter.blockSignals(False)
        self.populate_columns(pane)
        pane.y_list_widget.blockSignals(True)
        pane.y_list_widget.clearSelection()
        for row in range(pane.y_list_widget.count()):
            item = pane.y_list_widget.item(row)
            if item.data(QtCore.Qt.UserRole) == new_column_index:
                item.setSelected(True)
                pane.y_list_widget.setCurrentItem(item)
                pane.y_list_widget.scrollToItem(item)
                break
        pane.y_list_widget.blockSignals(False)
        self.update_table_preview()
        self.detail_tabs.setCurrentWidget(self.table_view)
        self.redraw()
        self.statusBar().showMessage(
            "Added calculated variable '{}' to {}".format(result_name, tab.nickname),
            10000,
        )

    def apply_table_transform_to_file(
            self,
            table_index,
            suffix,
            script,
            pane):
        try:
            pending, target_indices, trimmed_count, static_count = (
                transform_file_tables(
                    self.tab_list,
                    table_index,
                    suffix,
                    script,
                )
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Transform entire file",
                "{}: {}".format(type(exc).__name__, exc),
            )
            return

        transform_group = "table-transform-{}".format(time.time_ns())
        first_new_index = len(self.tab_list)
        for table in pending:
            table.source_metadata['transform_group'] = transform_group
            table.source_metadata['transform_script'] = script
        self.tab_list.append(pending)
        selected_offset = target_indices.index(table_index)
        display_offsets = list(range(selected_offset, len(pending)))
        display_offsets.extend(range(0, selected_offset))
        selected_indices = [
            first_new_index + offset
            for offset in display_offsets[:len(self.visible_selector_panes())]
        ]
        self.populate_tables(selected_table_indices=selected_indices)
        pane = (
            self.visible_selector_panes()[0]
            if self.visible_selector_panes()
            else pane
        )
        if pane is not None:
            self.active_selector_pane = pane
        self.update_table_preview()
        self.update_file_info()
        self.redraw()
        message = "Transformed {:,} time-dependent table(s)".format(trimmed_count)
        if static_count:
            message += "; copied {:,} static table(s) unchanged".format(
                static_count
            )
        self.statusBar().showMessage(message, 12000)

    def standardize_units(self, flavor, label):
        started = time.perf_counter()
        indices = list(range(len(self.tab_list)))
        if not indices and not self.lazy_entries:
            self.statusBar().showMessage("No loaded tables to standardize", 8000)
            return

        from pydatview.tools.pandalib import unitConversionPlan

        changed = 0
        table_plans = self.unit_conversion_plans(
            [self.tab_list[it] for it in indices], flavor
        )
        table_plans = [
            (tab, list(tab.data.columns), plan)
            for tab, plan in table_plans
        ]

        entry_columns = []
        for entry in self.lazy_entries:
            converted_columns = (
                unitConversionPlan(entry.columns, flavor)[0]
                if entry.columns else []
            )
            entry_columns.append((entry, converted_columns))

        for tab, before, plan in table_plans:
            tab.changeUnits(data={"flavor": flavor, "plan": plan})
            after = list(tab.data.columns)
            if before != after:
                changed += 1

        for entry, converted_columns in entry_columns:
            entry.unit_flavor = flavor
            if converted_columns:
                entry.columns = converted_columns
        self.unit_flavor = flavor

        for pane in self.visible_selector_panes():
            self.populate_columns(pane)
        self.update_table_preview()
        self.update_file_info()
        if self.live_plot.isChecked() and not self.has_unloaded_lazy_selection():
            self.redraw()
        self.statusBar().showMessage(
            "Standardized units to {} for {:,} loaded table(s), {:,} changed; "
            "{:,} indexed file(s) will use the same units ({:.3f}s)".format(
                label,
                len(indices),
                changed,
                len(self.lazy_entries),
                time.perf_counter() - started,
            ),
            12000,
        )

    @staticmethod
    def unit_conversion_plans(tabs, flavor):
        from pydatview.tools.pandalib import (
            unitConversionPlan,
            validateUnitConversion,
        )

        cached = {}
        result = []
        for tab in tabs:
            key = tuple(tab.data.columns)
            plan = cached.get(key)
            if plan is None:
                plan = unitConversionPlan(key, flavor)
                cached[key] = plan
            validateUnitConversion(tab.data, plan)
            result.append((tab, plan))
        return result

    def apply_active_units_to_tabs(self, tabs):
        if not self.unit_flavor or not tabs:
            return
        for tab, plan in self.unit_conversion_plans(tabs, self.unit_flavor):
            tab.changeUnits(data={
                "flavor": self.unit_flavor,
                "plan": plan,
            })

    def clear(self):
        self.canvas.clear_measurement_marker()
        self.canvas.clear_plot()
        self.plot_data = []
        self.update_stats()

    def open_axis_limits_dialog(self):
        dialog = AxisLimitsDialog(
            self.axis_limits,
            logx=self.logx_check.isChecked(),
            logy=self.logy_check.isChecked(),
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        self.axis_limits = dialog.values()
        self.update_axis_limits_button()
        self.redraw()

    def update_axis_limits_button(self):
        active = any(value is not None for value in self.axis_limits.values())
        self.axis_limits_button.setProperty("limitsActive", active)
        values = []
        for label, minimum_key, maximum_key in (
            ("X", "xmin", "xmax"),
            ("Y", "ymin", "ymax"),
        ):
            minimum = self.axis_limits.get(minimum_key)
            maximum = self.axis_limits.get(maximum_key)
            if minimum is not None or maximum is not None:
                values.append(
                    "{} [{}, {}]".format(
                        label,
                        "auto" if minimum is None else "{:.6g}".format(minimum),
                        "auto" if maximum is None else "{:.6g}".format(maximum),
                    )
                )
        self.axis_limits_button.setToolTip(
            "Set X and Y plot limits" if not values else "; ".join(values)
        )
        self.axis_limits_button.style().unpolish(self.axis_limits_button)
        self.axis_limits_button.style().polish(self.axis_limits_button)

    def auto_range(self):
        self.axis_limits = {key: None for key in ("xmin", "xmax", "ymin", "ymax")}
        self.update_axis_limits_button()
        for plot in self.canvas._plots:
            plot.autoRange()

    def on_zoom_area_toggled(self, enabled):
        self.zoom_area_button.setChecked(enabled)
        self.canvas.set_zoom_mode(enabled)

    def on_measurement_marker_toggled(self, enabled):
        self.canvas.set_measurement_marker_enabled(enabled)
        self.coordinate_label.setToolTip(
            "Click a plot or its X axis to place the marker"
            if enabled else "Live mouse coordinates"
        )

    def change_ui_font_size(self, delta):
        new_size = max(7, min(20, self._ui_font_size + int(delta)))
        if new_size == self._ui_font_size:
            return
        self._ui_font_size = new_size
        font = QtGui.QFont(self.font())
        font.setPointSize(new_size)
        self.setFont(font)
        selector_font = QtGui.QFont(font)
        selector_font.setPointSize(max(7, new_size - 1))
        for pane in self.selector_panes:
            pane.table_list_widget.setFont(selector_font)
            pane.bladed_dataset_combo.setFont(selector_font)
            pane.x_combo.setFont(selector_font)
            pane.y_list_widget.setFont(selector_font)
        coordinate_font = QtGui.QFont(self.coordinate_label.font())
        coordinate_font.setPointSize(new_size)
        self.coordinate_label.setFont(coordinate_font)
        self.statusBar().showMessage(
            "Interface font size: {} pt".format(new_size), 5000
        )

    def marker_symbol(self):
        return {
            "None": None,
            "Circle": "o",
            "Square": "s",
            "Triangle": "t",
            "Diamond": "d",
        }.get(self.marker_combo.currentText(), None)

    def update_table_preview(self):
        indices = self.selected_table_indices(load=False)
        if not indices:
            self.table_model.set_dataframe(None)
            return
        self.table_model.set_dataframe(self.tab_list[indices[0]].data)

    def update_file_info(self):
        lazy_indices = self.selected_lazy_indices()
        if lazy_indices:
            lines = []
            for lazy_index in lazy_indices:
                entry = self.lazy_entries[lazy_index]
                if entry.full_loaded:
                    status = "loaded"
                elif entry.loaded:
                    status = "partial ({}/{} variables)".format(
                        len(entry.loaded_column_indices),
                        len(entry.columns) if entry.columns else "?",
                    )
                elif entry.loading:
                    status = "loading"
                elif entry.attempted:
                    status = "failed"
                else:
                    status = "indexed"
                lines.append("File: {}".format(entry.path))
                lines.append("Format: {}".format(getattr(entry.file_format, "name", "auto")))
                lines.append("Status: {}".format(status))
                lines.append("Size: {:.3f} MB".format(entry.size / (1024 * 1024) if entry.size else 0.0))
                if entry.mtime:
                    lines.append("Modified: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.mtime))))
                if entry.warning:
                    lines.append("Warning: {}".format(entry.warning.splitlines()[0]))
                lines.append("")
            self.info_text.setPlainText("\n".join(lines))
            return

        indices = self.selected_table_indices(load=False)
        if not indices:
            self.info_text.clear()
            return
        lines = []
        for it in indices:
            tab = self.tab_list[it]
            lines.append("Table: {}".format(tab.active_name))
            lines.append("File: {}".format(tab.filename))
            lines.append("Format: {}".format(tab.fileformat_name))
            lines.append("Shape: {}".format(tab.shapestring))
            if tab.source_metadata.get('lazy_values'):
                lines.append("Data: NetCDF values loaded on demand")
            if tab.source_metadata.get('slice_tables_truncated'):
                lines.append("Slices: showing {} of {} (safety limit)".format(
                    tab.source_metadata['slice_tables_shown'],
                    tab.source_metadata['slice_tables_total'],
                ))
            lines.append("Columns: {}".format(", ".join(map(str, tab.columns[:40]))))
            if len(tab.columns) > 40:
                lines.append("...")
            lines.append("")
        self.info_text.setPlainText("\n".join(lines))

    def selected_del_slopes(self):
        return [
            slope for slope, action in self.del_slope_actions.items()
            if action.isChecked()
        ]

    def selected_stats_columns(self):
        return [
            key for key, _label, _numeric in _STATS_COLUMNS
            if self.stats_column_actions[key].isChecked()
        ]

    def update_stats_columns_button(self):
        count = len(self.selected_stats_columns())
        self.stats_columns_button.setText("{} selected".format(count))
        self.stats_columns_button.setToolTip(
            "Select statistics displayed for each plotted time series"
        )

    def on_stats_columns_changed(self, _checked=False):
        selected = self.selected_stats_columns()
        self.settings.setValue("stats/columns", selected)
        self.update_stats_columns_button()
        self.update_stats()

    def set_stats_columns(self, selected):
        selected = set(selected)
        for key, action in self.stats_column_actions.items():
            action.blockSignals(True)
            action.setChecked(key in selected)
            action.blockSignals(False)
        self.on_stats_columns_changed()

    def select_all_stats_columns(self):
        self.set_stats_columns(key for key, _label, _numeric in _STATS_COLUMNS)

    def reset_stats_columns(self):
        self.set_stats_columns(_DEFAULT_STATS_COLUMNS)

    def update_del_slopes_button(self):
        slopes = self.selected_del_slopes()
        self.del_slopes_button.setText(
            "m = {}".format(", ".join(map(str, slopes))) if slopes else "None"
        )
        self.del_slopes_button.setToolTip(
            "Select one or more Wöhler slopes for 1 Hz damage-equivalent loads"
        )

    def on_del_slopes_changed(self, _checked=False):
        slopes = self.selected_del_slopes()
        self.settings.setValue("stats/del_slopes", [str(slope) for slope in slopes])
        self.update_del_slopes_button()
        self.update_stats()

    @staticmethod
    def _stats_table_item(value, numeric=False):
        if numeric:
            if isinstance(value, (int, np.integer)):
                text = "{:,}".format(int(value))
            else:
                try:
                    value = float(value)
                    text = "{:.6g}".format(value) if np.isfinite(value) else "N/A"
                except (TypeError, ValueError):
                    text = "N/A"
        else:
            text = str(value)
        item = QtWidgets.QTableWidgetItem(text)
        if numeric:
            item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        return item

    def update_stats(self):
        selected = self.selected_stats_columns()
        selected_set = set(selected)
        selected_definitions = [
            definition for definition in _STATS_COLUMNS
            if definition[0] in selected_set
        ]
        slopes = self.selected_del_slopes()
        headers = [label for _key, label, _numeric in selected_definitions]
        headers.extend("DEL m={} (1 Hz)".format(slope) for slope in slopes)
        numeric_columns = [
            numeric for _key, _label, numeric in selected_definitions
        ] + [True] * len(slopes)
        self.stats_table.setSortingEnabled(False)
        self.stats_table.clear()
        self.stats_table.setColumnCount(len(headers))
        self.stats_table.setHorizontalHeaderLabels(headers)
        if not self.plot_data:
            self.stats_table.setRowCount(0)
            return

        rows = []
        for pd in self.plot_data:
            try:
                x_raw, y_raw = _finite_xy(pd.x0, pd.y0)
            except Exception:
                continue
            if len(y_raw) == 0:
                continue
            statistics = _series_statistics(pd, x_raw, y_raw, selected)
            del_values = _equivalent_loads(x_raw, y_raw, slopes)
            rows.append((
                pd,
                [statistics.get(key, np.nan if numeric else "")
                 for key, _label, numeric in selected_definitions]
                + [del_values[slope] for slope in slopes],
            ))

        self.stats_table.setRowCount(len(rows))
        for row_index, (pd, row_values) in enumerate(rows):
            for column_index, value in enumerate(row_values):
                item = self._stats_table_item(
                    value, numeric=numeric_columns[column_index]
                )
                if column_index < len(selected_definitions):
                    key = selected_definitions[column_index][0]
                    if key == "file":
                        item.setToolTip(
                            getattr(pd, "filename", "") or getattr(pd, "st", "")
                        )
                self.stats_table.setItem(row_index, column_index, item)
        header = self.stats_table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        if "series" in selected:
            header.setSectionResizeMode(
                selected.index("series"), QtWidgets.QHeaderView.Stretch
            )
        if "file" in selected:
            file_column = selected.index("file")
            header.setSectionResizeMode(
                file_column, QtWidgets.QHeaderView.Interactive
            )
            header.resizeSection(file_column, 150)
        if "directory" in selected:
            directory_column = selected.index("directory")
            header.setSectionResizeMode(
                directory_column, QtWidgets.QHeaderView.Interactive
            )
            header.resizeSection(directory_column, 220)
        self.stats_table.resizeRowsToContents()

    def stats_table_text(self, delimiter=",", selected_rows_only=False):
        """Return the visible statistics table in a spreadsheet-safe format."""
        if self.stats_table.columnCount() == 0:
            return ""
        selected_rows = {
            index.row() for index in self.stats_table.selectionModel().selectedRows()
        }
        if not selected_rows_only or not selected_rows:
            selected_rows = set(range(self.stats_table.rowCount()))
        output = io.StringIO(newline="")
        writer = csv.writer(output, delimiter=delimiter, lineterminator="\n")
        writer.writerow([
            self.stats_table.horizontalHeaderItem(column).text()
            for column in range(self.stats_table.columnCount())
        ])
        for row in range(self.stats_table.rowCount()):
            if row not in selected_rows:
                continue
            writer.writerow([
                self.stats_table.item(row, column).text()
                if self.stats_table.item(row, column) is not None else ""
                for column in range(self.stats_table.columnCount())
            ])
        return output.getvalue()

    def copy_stats(self):
        text = self.stats_table_text(delimiter="\t", selected_rows_only=True)
        if not text or self.stats_table.rowCount() == 0:
            self.statusBar().showMessage("No statistics to copy", 5000)
            return
        QtWidgets.QApplication.clipboard().setText(text)
        selected_count = len(self.stats_table.selectionModel().selectedRows())
        row_count = selected_count or self.stats_table.rowCount()
        self.statusBar().showMessage(
            "Copied statistics for {:,} series".format(row_count), 5000
        )

    def export_stats_csv(self):
        if self.stats_table.rowCount() == 0:
            self.statusBar().showMessage("No statistics to export", 5000)
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export statistics",
            "statistics.csv",
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".csv"
        try:
            with open(path, "w", encoding="utf-8", newline="") as stream:
                stream.write(self.stats_table_text(delimiter=","))
            self.statusBar().showMessage(
                "Statistics exported to {}".format(path), 10000
            )
        except Exception as exc:
            self.show_exception("Failed to export statistics", exc)

    def export_plot_image(self):
        if not self.plot_data:
            self.statusBar().showMessage(
                "Create a plot before exporting", 5000
            )
            return
        dialog = PublicationExportDialog(
            initial={
                "grid": self.grid_check.isChecked(),
                "legend": self.legend_check.isChecked(),
                "line_width": self.line_width_spin.value(),
                "x_label": getattr(self.plot_data[-1], "sx", ""),
                "y_label": (
                    ""
                    if self.mode_combo.currentText() == "Subplots"
                    else " and ".join(sorted(set(
                        getattr(pd, "sy", "") for pd in self.plot_data
                    )))
                ),
                "legend_sources": [
                    getattr(pd, "syl", "") or getattr(pd, "sy", "")
                    for pd in self.plot_data
                ],
                "legend_labels": [
                    "Set {}".format(index + 1)
                    for index in range(len(self.plot_data))
                ],
            },
            settings=self.settings,
            parent=self,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return

        options = dialog.options()
        try:
            self.statusBar().showMessage(
                "Exporting publication plot ...", 0
            )
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            QtWidgets.QApplication.processEvents()
            export_publication_plot(
                self.plot_data,
                options,
                subplots=self.mode_combo.currentText() == "Subplots",
                sharex=True,
                logx=self.logx_check.isChecked(),
                logy=self.logy_check.isChecked(),
                marker=self.marker_symbol(),
                step=self.plot_type_combo.currentText() == "MinMax",
                axis_limits=self.axis_limits,
            )
            self.statusBar().showMessage(
                "Publication plot exported to {}".format(options.path),
                10000,
            )
        except Exception as exc:
            self.show_exception("Failed to export publication plot", exc)
        finally:
            if QtWidgets.QApplication.overrideCursor() is not None:
                QtWidgets.QApplication.restoreOverrideCursor()

    def export_selected_table(self):
        partial = [
            lazy_index for lazy_index in self.selected_lazy_indices()
            if not self.lazy_entries[lazy_index].full_loaded
        ]
        if partial:
            self.statusBar().showMessage(
                "Use Load full selected before exporting a complete table",
                10000,
            )
            return
        indices = self.selected_table_indices()
        if not indices:
            return
        tab = self.tab_list[indices[0]]
        default = (tab.basename if tab.filename else tab.name) + ".csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export selected table",
            default,
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            try:
                tab.export(path=path, fformat="csv")
            except Exception as exc:
                self.show_exception("Failed to export table", exc)

    def show_exception(self, title, exc):
        traceback.print_exc()
        QtWidgets.QMessageBox.critical(self, title, "{}\n\n{}".format(exc, traceback.format_exc(limit=5)))
