"""Column selection and plot-data workflows for the Qt window."""

import os

import numpy as np

from pydatview.plotdata import PlotData
from pydatview.qt_compat import QtCore, QtWidgets
from pydatview.qt_stats import (
    box_plot_data,
    compare_plot_data,
    swap_plot_axes,
)


class QtSelectionPlotMixin:
    def populate_bladed_datasets(self, pane):
        previous_group = pane.bladed_dataset_combo.currentData()
        table_indices = self.bladed_project_table_indices(pane, group="__all__")
        groups = []
        for table_index in table_indices:
            group = self.tab_list[table_index].nickname
            if group not in groups:
                groups.append(group)

        lazy_indices = self.selected_lazy_indices(pane)
        lazy_datasets = []
        has_lazy_multi_table = False
        if not groups:
            for lazy_index in lazy_indices:
                entry = self.lazy_entries[lazy_index]
                if len(entry.table_indices) > 1:
                    has_lazy_multi_table = True
                for table_index in entry.table_indices:
                    dataset = self.tab_list[table_index].nickname
                    if dataset not in lazy_datasets:
                        lazy_datasets.append(dataset)

        pane.bladed_dataset_combo.blockSignals(True)
        pane.bladed_dataset_combo.clear()
        if groups:
            pane.dataset_mode = "bladed"
            pane.bladed_dataset_label.setText("BLADED VARIABLE GROUP")
            pane.bladed_dataset_combo.addItem("All variable groups", "__all__")
            for group in groups:
                pane.bladed_dataset_combo.addItem(group, group)
            selected_group = previous_group if previous_group in groups else "__all__"
            pane.bladed_dataset_combo.setCurrentIndex(
                pane.bladed_dataset_combo.findData(selected_group)
            )
        elif has_lazy_multi_table and lazy_datasets:
            pane.dataset_mode = "lazy"
            pane.bladed_dataset_label.setText("DATASET")
            for dataset in lazy_datasets:
                pane.bladed_dataset_combo.addItem(dataset, dataset)
            if previous_group in lazy_datasets:
                selected_dataset = previous_group
            else:
                selected_dataset = next(
                    (
                        preferred
                        for preferred in ("ZMidLine", "TSHubLine")
                        if preferred in lazy_datasets
                    ),
                    lazy_datasets[0],
                )
            pane.bladed_dataset_combo.setCurrentIndex(
                pane.bladed_dataset_combo.findData(selected_dataset)
            )
        else:
            pane.dataset_mode = None
        visible = bool(groups) or (has_lazy_multi_table and bool(lazy_datasets))
        pane.bladed_dataset_label.setVisible(visible)
        pane.bladed_dataset_combo.setVisible(visible)
        pane.bladed_dataset_combo.blockSignals(False)

    def on_bladed_dataset_changed(self, pane):
        if pane.bladed_dataset_combo.isHidden():
            return
        self.active_selector_pane = pane
        pane.y_list_widget.clearSelection()
        self.populate_columns(pane)
        self.update_table_preview()
        self.update_file_info()
        self.on_selection_changed()

    def populate_columns(self, pane=None):
        pane = pane or self.selector_panes[0]
        previous_x_name = pane.x_combo.currentText()
        previous_y_names = {
            item.text() for item in pane.y_list_widget.selectedItems()
        }
        lazy_indices = self.selected_lazy_indices(pane)
        indices = []
        columns = []
        project_indices = self.bladed_project_table_indices(pane)
        pane.bladed_project_mode = bool(project_indices)
        if project_indices:
            for table_index in project_indices:
                for column in self.tab_list[table_index].columns:
                    column = str(column)
                    if column not in columns:
                        columns.append(column)
        if lazy_indices:
            lazy_index = lazy_indices[0]
            entry = self.lazy_entries[lazy_index]
            self.ensure_lazy_header(lazy_index)
            if entry.columns:
                columns = list(entry.columns)
            elif entry.loaded and not project_indices:
                indices = self.selected_table_indices(load=False, pane=pane)
        if not lazy_indices and not project_indices:
            indices = self.selected_table_indices(load=False, pane=pane)
        if not indices and len(self.tab_list) > 0 and not self.lazy_entries:
            indices = [0]
        if indices and not columns:
            columns = list(self.tab_list[indices[0]].columns)
        pane.display_columns = list(columns)
        all_columns = [(i, str(col)) for i, col in enumerate(columns)]
        text_filter = pane.column_filter.text().strip().lower()
        visible_y = [(i, col) for i, col in all_columns
                     if not text_filter or text_filter in col.lower()]

        pane.x_combo.blockSignals(True)
        pane.y_list_widget.blockSignals(True)
        pane.x_combo.clear()
        pane.y_list_widget.clear()
        for original_i, col in all_columns:
            pane.x_combo.addItem(col, original_i)
        for original_i, col in visible_y:
            item = QtWidgets.QListWidgetItem(col)
            item.setData(QtCore.Qt.UserRole, original_i)
            pane.y_list_widget.addItem(item)

        if all_columns:
            all_names = [col for _, col in all_columns]
            if previous_x_name in all_names:
                x_to_select = all_columns[all_names.index(previous_x_name)][0]
            else:
                x_to_select = next(
                    (
                        i
                        for i, col in all_columns
                        if col.lower().startswith("time")
                        or col.lower() == "t"
                        or col.lower().startswith("t_[")
                    ),
                    next(
                        (
                            i
                            for i, col in all_columns
                            if not col.lower().startswith("index")
                        ),
                        all_columns[0][0],
                    ),
                )
            pane.x_combo.setCurrentIndex(
                next(
                    row for row in range(pane.x_combo.count())
                    if pane.x_combo.itemData(row) == x_to_select
                )
            )
        if visible_y and not previous_y_names:
            x_current = pane.x_combo.currentData()
            default_row = next((row for row, (i, _) in enumerate(visible_y) if i != x_current), 0)
            pane.y_list_widget.item(default_row).setSelected(True)
        else:
            for row in range(pane.y_list_widget.count()):
                item = pane.y_list_widget.item(row)
                if item.text() in previous_y_names:
                    item.setSelected(True)
        pane.x_combo.blockSignals(False)
        pane.y_list_widget.blockSignals(False)

    def on_selection_changed(self):
        if self.live_plot.isChecked():
            self.redraw_timer.start()

    def has_unloaded_lazy_selection(self):
        for lazy_index, request in self.lazy_plot_column_requests().items():
            entry = self.lazy_entries[lazy_index]
            if (
                not self.lazy_request_satisfied(entry, request)
                and not (entry.attempted and not entry.loaded)
            ):
                return True
        return False

    def select_all_y(self):
        for pane in self.visible_selector_panes():
            pane.y_list_widget.blockSignals(True)
            for row in range(pane.y_list_widget.count()):
                pane.y_list_widget.item(row).setSelected(True)
            pane.y_list_widget.blockSignals(False)
        self.on_selection_changed()

    def select_none_y(self):
        for pane in self.visible_selector_panes():
            pane.y_list_widget.blockSignals(True)
            for row in range(pane.y_list_widget.count()):
                pane.y_list_widget.item(row).setSelected(False)
            pane.y_list_widget.blockSignals(False)
        self.on_selection_changed()

    def selected_y_indices(self, pane=None):
        return self.selected_y_indices_original(pane)

    def selected_y_indices_original(self, pane=None):
        pane = pane or self.selector_panes[0]
        return [item.data(QtCore.Qt.UserRole) for item in pane.y_list_widget.selectedItems()]

    def lazy_plot_column_requests(self):
        requests = {}
        for pane in self.visible_selector_panes():
            lazy_indices = self.selected_lazy_indices(pane)
            if not lazy_indices:
                continue
            ix = pane.x_combo.currentData()
            y_indices = self.selected_y_indices_original(pane)
            if ix is None or not y_indices:
                continue

            reference = self.lazy_entries[lazy_indices[0]]
            self.ensure_lazy_header(lazy_indices[0])
            if not reference.columns:
                for lazy_index in lazy_indices:
                    requests[lazy_index] = None
                continue

            requested_names = []
            for column_index in [ix] + y_indices:
                if 0 <= column_index < len(reference.columns):
                    name = reference.columns[column_index]
                    if name not in requested_names:
                        requested_names.append(name)

            for lazy_index in lazy_indices:
                if requests.get(lazy_index) is None and lazy_index in requests:
                    continue
                entry = self.lazy_entries[lazy_index]
                self.ensure_lazy_header(lazy_index)
                if not entry.columns:
                    requests[lazy_index] = None
                    continue
                mapped = []
                for name in requested_names:
                    try:
                        mapped.append(entry.columns.index(name))
                    except ValueError:
                        mapped = []
                        break
                previous = set(requests.get(lazy_index, ()))
                requests[lazy_index] = tuple(sorted(previous.union(mapped)))
        return requests

    def build_plot_data(self):
        plot_data = []
        pane_payloads = []
        total_table_count = 0
        for pane_index, pane in enumerate(self.visible_selector_panes()):
            table_sources = []
            lazy_indices = self.selected_lazy_indices(pane)
            project_indices = self.bladed_project_table_indices(pane)
            if project_indices:
                table_sources = [(table_index, None) for table_index in project_indices]
            elif lazy_indices:
                for lazy_index in lazy_indices:
                    entry = self.lazy_entries[lazy_index]
                    for table_index in entry.table_indices:
                        if (
                            getattr(pane, "dataset_mode", None) == "lazy"
                            and self.tab_list[table_index].nickname
                            != pane.bladed_dataset_combo.currentData()
                        ):
                            continue
                        table_sources.append((table_index, entry))
            else:
                table_sources = [
                    (table_index, None)
                    for table_index in self.selected_table_indices(
                        load=False,
                        pane=pane,
                    )
                ]
            y_indices = self.selected_y_indices(pane)
            ix = pane.x_combo.currentData()
            if ix is None or not y_indices or not table_sources:
                continue
            pane_payloads.append((
                pane_index,
                table_sources,
                y_indices,
                ix,
                pane.bladed_project_mode,
                list(pane.display_columns),
            ))
            if pane.bladed_project_mode:
                total_table_count += len({
                    self.normalized_file_path(self.tab_list[table_index].filename)
                    for table_index, _entry in table_sources
                })
            else:
                total_table_count += len(table_sources)

        same_col = total_table_count > 1 or len(pane_payloads) > 1
        for pane_index, table_sources, y_indices, ix, project_mode, display_columns in pane_payloads:
            seen_project_curves = set()
            for it, entry in table_sources:
                tab = self.tab_list[it]
                tab_columns = [str(column) for column in tab.columns]
                if project_mode:
                    if ix >= len(display_columns):
                        continue
                    x_name = display_columns[ix]
                    try:
                        actual_ix = tab_columns.index(x_name)
                    except ValueError:
                        continue
                elif entry is not None and entry.columns:
                    if ix >= len(entry.columns):
                        continue
                    x_name = entry.columns[ix]
                    try:
                        actual_ix = tab_columns.index(x_name)
                    except ValueError:
                        continue
                else:
                    if ix >= len(display_columns):
                        continue
                    x_name = display_columns[ix]
                    try:
                        actual_ix = tab_columns.index(x_name)
                    except ValueError:
                        continue
                if actual_ix >= len(tab.columns):
                    continue
                for selection_index, iy in enumerate(y_indices):
                    if project_mode:
                        if iy >= len(display_columns):
                            continue
                        y_name = display_columns[iy]
                        try:
                            actual_iy = tab_columns.index(y_name)
                        except ValueError:
                            continue
                        curve_key = (self.normalized_file_path(tab.filename), y_name)
                        if curve_key in seen_project_curves:
                            continue
                        seen_project_curves.add(curve_key)
                    elif entry is not None and entry.columns:
                        if iy >= len(entry.columns):
                            continue
                        y_name = entry.columns[iy]
                        try:
                            actual_iy = tab_columns.index(y_name)
                        except ValueError:
                            continue
                    else:
                        if iy >= len(display_columns):
                            continue
                        y_name = display_columns[iy]
                        try:
                            actual_iy = tab_columns.index(y_name)
                        except ValueError:
                            continue
                    if actual_iy >= len(tab.columns):
                        continue
                    idx = (
                        it,
                        actual_ix,
                        actual_iy,
                        str(tab.columns[actual_ix]),
                        str(tab.columns[actual_iy]),
                        tab.active_name,
                    )
                    pd = PlotData()
                    pd.fromIDs(self.tab_list, len(plot_data), idx, same_col, pipeline=None)
                    pd.pane_index = pane_index
                    pd.selection_index = selection_index
                    if project_mode:
                        pd.st = os.path.basename(tab.filename)
                    self.apply_plot_type(pd)
                    if same_col:
                        pd.syl = "Set {}: {} - {}".format(pane_index + 1, pd.st, pd.sy)
                    else:
                        pd.syl = pd.sy
                    plot_data.append(pd)
        plot_type = self.plot_type_combo.currentText()
        if plot_type == "Box Plot":
            plot_data = box_plot_data(plot_data)
        elif plot_type == "Compare":
            if len(plot_data) < 2:
                self.statusBar().showMessage(
                    "Compare requires at least two selected time series", 8000
                )
                return []
            plot_data = compare_plot_data(
                plot_data, self.comparison_method_combo.currentText()
            )
        if self.swap_xy_check.isChecked() and plot_type != "Box Plot":
            for pd in plot_data:
                swap_plot_axes(pd)
        return plot_data

    def apply_plot_type(self, pd):
        plot_type = self.plot_type_combo.currentText()
        if plot_type == "PDF":
            pd.toPDF(nBins=101, smooth=False)
        elif plot_type == "FFT":
            pd.toFFT(
                yType=self.fft_output_combo.currentText(),
                xType=self.fft_x_combo.currentData(),
                avgMethod=self.fft_averaging_combo.currentText(),
                avgWindow=self.fft_window_combo.currentText(),
                bDetrend=self.fft_detrend_check.isChecked(),
                nExp=self.fft_nexp_spin.value(),
                nPerDecade=self.fft_bins_spin.value(),
            )
        elif plot_type == "Cumulative PSD":
            pd.toCumulativePSD(
                avgMethod=self.fft_averaging_combo.currentText(),
                avgWindow=self.fft_window_combo.currentText(),
                bDetrend=self.fft_detrend_check.isChecked(),
                nExp=self.fft_nexp_spin.value(),
                nPerDecade=self.fft_bins_spin.value(),
            )
        elif plot_type == "MinMax":
            pd.toMinMax(xScale=False, yScale=True, yCenter="None")

    def redraw(self):
        try:
            if self.redraw_timer.isActive():
                self.redraw_timer.stop()
            column_requests = self.lazy_plot_column_requests()
            missing = [
                lazy_index
                for lazy_index, request in column_requests.items()
                if not self.lazy_request_satisfied(
                    self.lazy_entries[lazy_index],
                    request,
                )
                and not (
                    self.lazy_entries[lazy_index].attempted
                    and not self.lazy_entries[lazy_index].loaded
                )
            ]
            if missing:
                pending = self.pending_lazy_indices(
                    missing,
                    column_requests=column_requests,
                )
                self.lazy_selected_batch = set(missing)
                self.plot_after_lazy_load = True
                self.begin_lazy_load_batch(len(pending))
                for lazy_index in pending:
                    self.ensure_lazy_loaded(
                        lazy_index,
                        show_warning=False,
                        channel_indices=column_requests[lazy_index],
                    )
                self.statusBar().showMessage(
                    "Loading selected X/Y variables from {:,} files ...".format(
                        len(missing)
                    ),
                    8000,
                )
                return
            self.plot_data = self.build_plot_data()
            self.canvas.plot_data(
                self.plot_data,
                subplots=self.mode_combo.currentText() == "Subplots",
                sharex=True,
                grid=self.grid_check.isChecked(),
                logx=(
                    self.logx_check.isChecked()
                    and self.plot_type_combo.currentText() != "Box Plot"
                ),
                logy=(
                    self.logy_check.isChecked()
                    and self.plot_type_combo.currentText() != "Box Plot"
                ),
                show_legend=self.legend_check.isChecked(),
                line_width=self.line_width_spin.value(),
                marker=self.marker_symbol(),
                axis_limits=self.axis_limits,
                order_overlays=self.order_overlay_markers(),
            )
            n_curves = len(self.plot_data)
            n_points = sum(len(pd.y) for pd in self.plot_data)
            self.update_stats()
            self.statusBar().showMessage("{} curves, {:,} points".format(n_curves, n_points))
        except Exception as exc:
            self.show_exception("Failed to plot data", exc)

    def on_curve_selected(self, meta):
        self.highlight_curve_table(meta)
        message = "Selected: {label} | file/table: {file} | y: {y} | x: {x} | {points:,} points".format(
            label=meta.get("label", ""),
            file=meta.get("file", ""),
            y=meta.get("y", ""),
            x=meta.get("x", ""),
            points=meta.get("points", 0),
        )
        self.statusBar().showMessage(message)

    @staticmethod
    def _format_hover_value(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return "{:.7g}".format(value) if np.isfinite(value) else "N/A"

    def on_plot_hover(self, coordinates):
        if not coordinates:
            self.coordinate_label.setText("X: --   Y: --")
            return
        prefix = ""
        if coordinates.get("plot_count", 1) > 1:
            prefix = "Plot {}   ".format(coordinates.get("plot_index", 0) + 1)
        self.coordinate_label.setText(
            "{}X: {}   Y: {}".format(
                prefix,
                self._format_hover_value(coordinates.get("x")),
                self._format_hover_value(coordinates.get("y")),
            )
        )

    def highlight_curve_table(self, meta):
        table_index = meta.get("table_index")
        if table_index is None:
            return
        pane_index = meta.get("pane_index", 0)
        panes = self.visible_selector_panes()
        if not panes:
            return
        pane = panes[pane_index] if isinstance(pane_index, int) and pane_index < len(panes) else panes[0]
        target_row = None
        for row in range(pane.table_list_widget.count()):
            item = pane.table_list_widget.item(row)
            data = item.data(QtCore.Qt.UserRole)
            if isinstance(data, tuple) and data == ("table", table_index):
                target_row = row
                break
            if (
                isinstance(data, tuple)
                and data[0] == "bladed_project"
                and self.normalized_file_path(self.tab_list[table_index].filename)
                == self.normalized_file_path(data[1])
            ):
                target_row = row
                break
            if isinstance(data, tuple) and data[0] == "lazy":
                entry = self.lazy_entries[data[1]]
                if table_index in entry.table_indices:
                    target_row = row
                    break
        if target_row is None:
            return
        pane.table_list_widget.blockSignals(True)
        pane.table_list_widget.clearSelection()
        item = pane.table_list_widget.item(target_row)
        item.setSelected(True)
        pane.table_list_widget.setCurrentItem(item)
        pane.table_list_widget.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
        pane.table_list_widget.blockSignals(False)
        self.update_table_preview()
        self.update_file_info()
