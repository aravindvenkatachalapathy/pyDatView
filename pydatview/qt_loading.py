"""Scanning, lazy loading, and table-index workflows for the Qt window."""

import os
import time
from collections import deque

from pydatview.qt_compat import QtCore, QtGui, QtWidgets
from pydatview.qt_dialogs import ScanDialog
from pydatview.qt_io import (
    LazyFileEntry,
    LazyLoadWorker,
    _default_lazy_workers,
    read_lazy_columns,
    scan_readable_file_matches,
)


class QtLoadingMixin:
    def update_lazy_worker_limit(self):
        text = self.load_workers_combo.currentText()
        if text == "Auto":
            self.lazy_max_workers = _default_lazy_workers()
        else:
            self.lazy_max_workers = max(1, min(max(1, os.cpu_count() or 1), int(text)))
        self.statusBar().showMessage(
            "Parallel workers: {} overall, {} for Bladed".format(
                self.lazy_max_workers,
                min(self.lazy_max_workers, self.bladed_worker_cap),
            ),
            8000,
        )
        self.start_next_lazy_load()

    def set_loading_controls_enabled(self, enabled):
        for action in (
            self.open_action,
            self.add_action,
            self.reload_action,
            self.scan_action,
            self.autorange_action,
            self.zoom_area_action,
            self.axis_limits_action,
            self.standardize_units_action,
            self.export_table_action,
            self.export_plot_action,
            self.math_action,
        ):
            action.setEnabled(enabled)
        for widget in (
            self.plot_type_combo,
            self.mode_combo,
            self.compare_combo,
            self.live_plot,
            self.swap_xy_check,
            self.grid_check,
            self.logx_check,
            self.logy_check,
            self.legend_check,
            self.measurement_marker_check,
            self.line_width_spin,
            self.marker_combo,
            self.axis_limits_button,
            self.zoom_area_button,
            self.load_workers_combo,
            self.plot_button,
            self.clear_button,
            self.select_all_y_button,
            self.select_none_y_button,
            self.load_selected_button,
            self.math_button,
            self.copy_stats_button,
            self.export_stats_button,
            self.fft_options_panel,
            self.comparison_options_panel,
        ):
            widget.setEnabled(enabled)
        for pane in self.selector_panes:
            pane.frame.setEnabled(enabled)

    def begin_lazy_load_batch(self, total):
        if total <= 0:
            return
        if self.lazy_batch_total == 0:
            self.lazy_batch_done = 0
            self.lazy_batch_total = total
        else:
            self.lazy_batch_total += total
        self.loading_progress.setRange(0, self.lazy_batch_total)
        self.loading_progress.setValue(self.lazy_batch_done)
        self.loading_progress.setFormat("Loading %v/%m")
        self.loading_progress.setVisible(True)
        self.lazy_last_ui_update = 0.0
        self.set_loading_controls_enabled(False)

    def advance_lazy_load_progress(self):
        if self.lazy_batch_total <= 0:
            return False
        self.lazy_batch_done = min(self.lazy_batch_done + 1, self.lazy_batch_total)
        now = time.perf_counter()
        refresh = (
            self.lazy_batch_done >= self.lazy_batch_total
            or now - self.lazy_last_ui_update >= 0.1
        )
        if refresh:
            self.lazy_last_ui_update = now
            self.loading_progress.setValue(self.lazy_batch_done)
            self.loading_progress.setFormat(
                "Loading {}/{}".format(self.lazy_batch_done, self.lazy_batch_total)
            )
        return refresh

    def finish_lazy_load_batch_if_done(self):
        if self.lazy_batch_total <= 0:
            return
        if self.lazy_load_queue or self.lazy_loader_threads:
            return
        self.loading_progress.setValue(self.lazy_batch_total)
        self.loading_progress.setFormat("Loaded {}/{}".format(self.lazy_batch_done, self.lazy_batch_total))
        self.loading_progress.setVisible(False)
        self.status_label.setText(
            "{:,} files indexed, {:,} loaded, 0 active".format(
                len(self.lazy_entries), self.lazy_loaded_count()
            )
        )
        self.lazy_batch_total = 0
        self.lazy_batch_done = 0
        self.lazy_last_ui_update = 0.0
        self.flush_lazy_selection_refresh()
        self.lazy_selected_batch = set()
        self.set_loading_controls_enabled(True)
        if self.lazy_warning_backlog:
            warning_count = len(self.lazy_warning_backlog)
            first_warning = self.lazy_warning_backlog[0].splitlines()[0]
            self.lazy_warning_backlog = []
            self.statusBar().showMessage(
                "{} load warning(s): {}".format(
                    warning_count,
                    first_warning,
                ),
                20000,
            )

    def flush_lazy_selection_refresh(self):
        needs_plot = self.plot_after_lazy_load
        self.plot_after_lazy_load = False
        if self.lazy_selection_refresh_pending:
            self.lazy_selection_refresh_pending = False
            self.on_table_selection_changed()
            if needs_plot and not self.live_plot.isChecked():
                self.redraw()
        elif needs_plot:
            self.redraw()

    def _show_file_format_errors(self):
        for err in self.file_format_errors:
            self.statusBar().showMessage(str(err), 10000)

    def select_files(self, add=False):
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open files",
            "",
            "All supported files (*);;All files (*)",
        )
        if filenames:
            self.load_files(filenames, add=add)

    def scan_folder(self):
        dialog = ScanDialog(self.file_formats, self, settings=self.settings)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return

        folder = dialog.selected_folder()
        format_entries = dialog.selected_format_entries()
        recursive = dialog.recursive()
        bladed_suffixes = dialog.bladed_suffixes()
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            t0 = time.perf_counter()
            self.statusBar().showMessage("Scanning {} ...".format(folder))
            QtWidgets.QApplication.processEvents()
            matches = scan_readable_file_matches(
                folder,
                format_entries,
                recursive=recursive,
                bladed_suffixes=bladed_suffixes,
            )
            scan_seconds = time.perf_counter() - t0
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        if not matches:
            QtWidgets.QMessageBox.information(
                self,
                "Scan folder",
                "No matching readable files were found in:\n{}".format(folder),
            )
            self.statusBar().showMessage("Scan found no files in {:.3f}s".format(scan_seconds), 8000)
            return

        added = self.set_lazy_file_index(
            matches,
            append=dialog.keep_existing(),
        )
        self.statusBar().showMessage(
            "Indexed {:,} new files in {:.3f}s; {:,} total, {:,} loaded".format(
                added,
                scan_seconds,
                len(self.lazy_entries),
                self.lazy_loaded_count(),
            ),
            12000,
        )

    def load_files(self, filenames, add=False, fileformats=None, status_prefix="Loading files"):
        t0 = time.perf_counter()
        try:
            if fileformats is None:
                pairs = [(f, None) for f in filenames if os.path.isfile(f)]
            else:
                pairs = [(f, ff) for f, ff in zip(filenames, fileformats) if os.path.isfile(f)]
            pairs = sorted(pairs, key=lambda item: item[0])
            filenames = [f for f, _ in pairs]
            fileformats = [ff for _, ff in pairs]
            if not filenames:
                return None
            if self.lazy_entries:
                self.lazy_generation += 1
                self.lazy_load_queue = deque()
                self.lazy_warning_backlog = []
                self.lazy_memory_reservations = {}
                self.lazy_entries = []
                self.lazy_item_widgets = {}
                self.lazy_loaded_total = 0
                self.lazy_selected_batch = set()
                self.lazy_selection_refresh_pending = False
            if not add:
                self.tab_list.clean()
                self.current_files = []

            last_status = {"t": 0.0}

            def status_function(i):
                now = time.perf_counter()
                if i == 0 or i == len(filenames) - 1 or now - last_status["t"] > 0.15:
                    last_status["t"] = now
                    self.status_label.setText("{} {}/{}".format(status_prefix, i + 1, len(filenames)))
                    self.statusBar().showMessage("{} {}/{}".format(status_prefix, i + 1, len(filenames)))
                    QtWidgets.QApplication.processEvents()

            new_tabs, warnings = self.tab_list.load_tables_from_files(
                filenames=filenames,
                fileformats=fileformats,
                bAdd=add,
                bReload=False,
                statusFunction=status_function,
            )
            self.current_files = self.tab_list.filenames
            warnings = [warning for warning in warnings if warning]
            if warnings:
                shown = "\n\n".join(warnings[:5])
                if len(warnings) > 5:
                    shown += "\n\n... {} more warnings".format(len(warnings) - 5)
                QtWidgets.QMessageBox.warning(self, "Load warnings", shown)
            if len(new_tabs) == 0 and len(self.tab_list) == 0:
                self.status_label.setText("No tables loaded")
                return time.perf_counter() - t0
            self.populate_tables()
            self.status_label.setText("{} tables loaded".format(len(self.tab_list)))
            self.redraw()
            return time.perf_counter() - t0
        except Exception as exc:
            self.show_exception("Failed to load files", exc)
            return None

    @staticmethod
    def normalized_file_path(path):
        return os.path.normcase(os.path.abspath(path))

    def selected_lazy_paths_by_pane(self):
        selected = []
        for pane in self.visible_selector_panes():
            selected.append({
                self.normalized_file_path(
                    self.lazy_entries[data[1]].path
                )
                for item in pane.table_list_widget.selectedItems()
                for data in [item.data(QtCore.Qt.UserRole)]
                if isinstance(data, tuple) and data[0] == "lazy"
            })
        return selected

    def set_lazy_file_index(self, matches, append=False):
        if append and self.lazy_entries:
            selected_paths = self.selected_lazy_paths_by_pane()
            known_paths = {
                self.normalized_file_path(entry.path)
                for entry in self.lazy_entries
            }
            added = 0
            for path, fmt in matches:
                normalized = self.normalized_file_path(path)
                if normalized in known_paths:
                    continue
                try:
                    stat = os.stat(path)
                    size = stat.st_size
                    mtime = stat.st_mtime
                except OSError:
                    size = 0
                    mtime = 0.0
                self.lazy_entries.append(
                    LazyFileEntry(
                        path=path,
                        file_format=fmt,
                        size=size,
                        mtime=mtime,
                    )
                )
                known_paths.add(normalized)
                added += 1
            self.current_files = [entry.path for entry in self.lazy_entries]
            self.populate_tables(selected_lazy_paths=selected_paths)
            self.status_label.setText(
                "{:,} files indexed, {:,} loaded".format(
                    len(self.lazy_entries),
                    self.lazy_loaded_count(),
                )
            )
            return added

        self.lazy_generation += 1
        self.lazy_load_queue = deque()
        self.lazy_warning_backlog = []
        self.lazy_memory_reservations = {}
        self.lazy_item_widgets = {}
        self.lazy_loaded_total = 0
        self.lazy_selected_batch = set()
        self.lazy_selection_refresh_pending = False
        self.tab_list.clean()
        self.current_files = [path for path, _ in matches]
        self.lazy_entries = []
        for path, fmt in matches:
            try:
                stat = os.stat(path)
                size = stat.st_size
                mtime = stat.st_mtime
            except OSError:
                size = 0
                mtime = 0.0
            self.lazy_entries.append(LazyFileEntry(path=path, file_format=fmt, size=size, mtime=mtime))
        self.populate_tables()
        self.clear()
        self.status_label.setText("{:,} files indexed, 0 loaded".format(len(self.lazy_entries)))
        return len(self.lazy_entries)

    def lazy_loaded_count(self):
        return self.lazy_loaded_total

    def lazy_item_text(self, entry):
        if entry.full_loaded:
            state = "loaded"
        elif entry.loaded:
            total = len(entry.columns) if entry.columns else "?"
            state = "partial {}/{}".format(
                len(entry.loaded_column_indices),
                total,
            )
        elif entry.loading:
            state = "loading"
        elif entry.attempted:
            state = "failed"
        else:
            state = "indexed"
        size_mb = entry.size / (1024 * 1024) if entry.size else 0.0
        fmt_name = getattr(entry.file_format, "name", "auto")
        return "{}  [{} | {:.2f} MB | {}]".format(entry.basename, state, size_mb, fmt_name)

    def ensure_lazy_header(self, lazy_index):
        entry = self.lazy_entries[lazy_index]
        if entry.columns or entry.header_attempted:
            return
        entry.header_attempted = True
        try:
            entry.columns = read_lazy_columns(entry.path, entry.file_format)
        except Exception as exc:
            entry.warning = "Header read failed: {}: {}".format(type(exc).__name__, exc)

    def lazy_request_satisfied(self, entry, channel_indices):
        if entry.full_loaded:
            return True
        if channel_indices is None:
            return False
        if not channel_indices:
            return True
        return entry.loaded and set(channel_indices).issubset(
            entry.loaded_column_indices
        )

    def is_lazy_queued(self, lazy_index):
        return any(item[0] == lazy_index for item in self.lazy_load_queue)

    @staticmethod
    def is_bladed_entry(entry):
        return getattr(entry.file_format, "name", "") == "Bladed output file"

    @staticmethod
    def available_memory_bytes():
        try:
            import psutil
            return int(psutil.virtual_memory().available)
        except Exception:
            return None

    def estimate_lazy_load_bytes(self, entry):
        if entry.estimated_load_bytes > 0:
            return entry.estimated_load_bytes
        source_bytes = max(0, int(entry.size))
        if self.is_bladed_entry(entry) and self.is_bladed_project_path(entry.path):
            directory = os.path.dirname(os.path.abspath(entry.path))
            directory_key = os.path.normcase(directory)
            if directory_key not in self._directory_file_sizes:
                files = []
                try:
                    with os.scandir(directory) as entries:
                        for candidate in entries:
                            try:
                                if candidate.is_file(follow_symlinks=False):
                                    files.append((candidate.name.lower(), candidate.stat().st_size))
                            except OSError:
                                continue
                except OSError:
                    pass
                self._directory_file_sizes[directory_key] = files
            project_root = os.path.splitext(os.path.basename(entry.path))[0].lower()
            binary_prefix = project_root + ".$"
            source_bytes = sum(
                size for name, size in self._directory_file_sizes[directory_key]
                if name.startswith(binary_prefix)
            ) or source_bytes
        # Dataframes, index columns, and decoder scratch space add overhead.
        entry.estimated_load_bytes = max(
            64 * 1024 * 1024,
            int(source_bytes * 2.0),
        )
        return entry.estimated_load_bytes

    def effective_lazy_worker_limit(self):
        has_bladed = any(
            self.is_bladed_entry(self.lazy_entries[index])
            for index in self.lazy_loader_threads
            if index < len(self.lazy_entries)
        ) or any(
            index < len(self.lazy_entries) and self.is_bladed_entry(self.lazy_entries[index])
            for index, _channels in self.lazy_load_queue
        )
        if has_bladed:
            return min(self.lazy_max_workers, self.bladed_worker_cap)
        return self.lazy_max_workers

    def lazy_memory_allows_start(self, entry):
        available = self.available_memory_bytes()
        if available is None:
            return True, ""
        required = self.estimate_lazy_load_bytes(entry)
        reserved = sum(self.lazy_memory_reservations.values())
        reserve_floor = max(1024 ** 3, int(available * 0.10))
        if available - reserved - required >= reserve_floor:
            return True, ""
        return False, (
            "Not enough available memory to load {} safely: estimated {:.2f} GB "
            "required with {:.2f} GB available. Reduce the selection, unload data, "
            "or lower the worker count."
        ).format(
            entry.basename,
            required / 1024 ** 3,
            max(0, available - reserved) / 1024 ** 3,
        )

    def reject_lazy_load(self, lazy_index, warning):
        entry = self.lazy_entries[lazy_index]
        entry.loading = False
        entry.attempted = not entry.loaded
        entry.warning = warning
        self.lazy_warning_backlog.append(warning)
        self.advance_lazy_load_progress()
        self.update_lazy_item(lazy_index)
        print("[pyDatView] {}".format(warning))

    def ensure_lazy_loaded(
            self,
            lazy_index,
            show_warning=True,
            channel_indices=None):
        entry = self.lazy_entries[lazy_index]
        if self.lazy_request_satisfied(entry, channel_indices):
            return entry.table_indices
        if entry.attempted and not entry.loaded:
            if entry.warning and show_warning:
                QtWidgets.QMessageBox.warning(self, "Load warning", entry.warning)
            return []
        self.queue_lazy_load(lazy_index, channel_indices=channel_indices)
        return []

    def pending_lazy_indices(self, lazy_indices, column_requests=None):
        pending = []
        for lazy_index in lazy_indices:
            entry = self.lazy_entries[lazy_index]
            request = (
                column_requests.get(lazy_index)
                if column_requests is not None
                else None
            )
            if self.lazy_request_satisfied(entry, request):
                continue
            if (
                entry.loading
                or (entry.attempted and not entry.loaded)
                or self.is_lazy_queued(lazy_index)
            ):
                continue
            pending.append(lazy_index)
        return pending

    def queue_lazy_load(self, lazy_index, channel_indices=None):
        entry = self.lazy_entries[lazy_index]
        if self.lazy_request_satisfied(entry, channel_indices):
            return
        if (
            entry.loading
            or (entry.attempted and not entry.loaded)
            or self.is_lazy_queued(lazy_index)
        ):
            return
        if channel_indices is not None:
            channel_indices = tuple(sorted(
                entry.loaded_column_indices.union(channel_indices)
            ))
        if self.lazy_batch_total == 0:
            self.begin_lazy_load_batch(1)
        entry.loading = True
        self.lazy_load_queue.append((lazy_index, channel_indices))
        if self.lazy_batch_total <= 1:
            self.status_label.setText("Loading {}".format(entry.basename))
            self.statusBar().showMessage("Queued {}".format(entry.path))
            self.update_lazy_item(lazy_index)
        self.start_next_lazy_load()

    def start_next_lazy_load(self):
        while self.lazy_load_queue:
            if len(self.lazy_loader_threads) >= self.effective_lazy_worker_limit():
                break
            lazy_index, _channel_indices = self.lazy_load_queue[0]
            if lazy_index >= len(self.lazy_entries):
                self.lazy_load_queue.popleft()
                continue
            allowed, warning = self.lazy_memory_allows_start(
                self.lazy_entries[lazy_index]
            )
            if not allowed:
                if self.lazy_loader_threads:
                    self.statusBar().showMessage(
                        "Waiting for memory before loading {}".format(
                            self.lazy_entries[lazy_index].basename
                        )
                    )
                    break
                self.lazy_load_queue.popleft()
                self.reject_lazy_load(lazy_index, warning)
                continue
            self.start_one_lazy_load()

    def start_one_lazy_load(self):
        if not self.lazy_load_queue:
            return
        lazy_index, channel_indices = self.lazy_load_queue.popleft()
        if lazy_index >= len(self.lazy_entries):
            self.start_next_lazy_load()
            return
        entry = self.lazy_entries[lazy_index]
        self.lazy_memory_reservations[lazy_index] = self.estimate_lazy_load_bytes(entry)
        if self.lazy_batch_total <= 1:
            self.status_label.setText("Loading {}".format(entry.basename))
            self.statusBar().showMessage("Loading {}".format(entry.path))

        generation = self.lazy_generation
        thread = QtCore.QThread(self)
        worker = LazyLoadWorker(
            generation,
            lazy_index,
            entry.path,
            entry.file_format,
            self.tab_list.options,
            channel_indices=channel_indices,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self.on_lazy_load_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda idx=lazy_index: self.on_lazy_thread_finished(idx))
        self.lazy_loader_threads[lazy_index] = thread
        self.lazy_loader_workers[lazy_index] = worker
        thread.start()

    def on_lazy_load_finished(
            self,
            generation,
            lazy_index,
            tabs,
            warning,
            elapsed,
            format_name,
            loaded_column_indices):
        if generation != self.lazy_generation:
            return
        if lazy_index >= len(self.lazy_entries):
            return
        entry = self.lazy_entries[lazy_index]
        was_loaded = entry.loaded
        if tabs:
            if was_loaded and len(entry.table_indices) == len(tabs):
                for table_index, tab in zip(entry.table_indices, tabs):
                    self.tab_list._tabs[table_index] = tab
            else:
                start = len(self.tab_list)
                self.tab_list.append(tabs)
                entry.table_indices = list(range(start, start + len(tabs)))
                if not was_loaded:
                    self.lazy_loaded_total += 1
            if loaded_column_indices is None:
                entry.full_loaded = True
                entry.loaded_column_indices = set(range(len(entry.columns)))
            else:
                entry.loaded_column_indices = set(loaded_column_indices)
        entry.warning = warning or ""
        entry.attempted = not tabs and not entry.loaded
        entry.loading = False
        refresh_ui = self.advance_lazy_load_progress()
        self.update_lazy_item(lazy_index)
        if refresh_ui:
            self.status_label.setText(
                "{:,} files indexed, {:,} loaded, {:,} active".format(
                    len(self.lazy_entries), self.lazy_loaded_count(), len(self.lazy_loader_threads)
                )
            )
        n_rows = sum(getattr(tab, "nRows", 0) for tab in tabs) if tabs else 0
        n_cols = sum(getattr(tab, "nCols", 0) for tab in tabs) if tabs else 0
        if refresh_ui:
            self.statusBar().showMessage(
                "Loaded {} in {:.3f}s ({}, {:,} rows, {:,} cols)".format(
                    entry.basename, elapsed, format_name, n_rows, n_cols
                ),
                12000,
            )
        if entry.warning:
            self.lazy_warning_backlog.append(entry.warning)
        if self.is_lazy_selected(lazy_index):
            self.lazy_selection_refresh_pending = True
        if not self.has_unloaded_lazy_selection():
            self.flush_lazy_selection_refresh()
        self.finish_lazy_load_batch_if_done()

    def on_lazy_thread_finished(self, lazy_index):
        self.lazy_loader_threads.pop(lazy_index, None)
        self.lazy_loader_workers.pop(lazy_index, None)
        self.lazy_memory_reservations.pop(lazy_index, None)
        self.start_next_lazy_load()
        self.finish_lazy_load_batch_if_done()

    def is_lazy_selected(self, lazy_index):
        if self.lazy_selected_batch:
            return lazy_index in self.lazy_selected_batch
        for pane in self.visible_selector_panes():
            for item in pane.table_list_widget.selectedItems():
                data = item.data(QtCore.Qt.UserRole)
                if isinstance(data, tuple) and data == ("lazy", lazy_index):
                    return True
        return False

    def update_lazy_item(self, lazy_index):
        text = self.lazy_item_text(self.lazy_entries[lazy_index])
        for item in self.lazy_item_widgets.get(lazy_index, ()):
            item.setText(text)

    def load_selected_lazy_files(self):
        lazy_indices = self.selected_lazy_indices()
        if not lazy_indices:
            return
        pending = self.pending_lazy_indices(lazy_indices)
        self.lazy_selected_batch = set(lazy_indices)
        self.begin_lazy_load_batch(len(pending))
        if pending:
            self.statusBar().showMessage(
                "Queueing {:,} selected files".format(len(pending))
            )
        for lazy_index in lazy_indices:
            self.ensure_lazy_loaded(
                lazy_index,
                show_warning=False,
                channel_indices=None,
            )
        if not pending:
            self.lazy_selected_batch = set()
        self.finish_lazy_load_batch_if_done()
        self.on_table_selection_changed()

    def show_table_context_menu(self, pane, position):
        item = pane.table_list_widget.itemAt(position)
        if item is None:
            return
        if not item.isSelected():
            pane.table_list_widget.clearSelection()
            item.setSelected(True)
            pane.table_list_widget.setCurrentItem(item)
        self.active_selector_pane = pane

        menu = QtWidgets.QMenu(pane.table_list_widget)
        remove_action = menu.addAction("Remove from pyDatView")
        reload_action = menu.addAction("Reload")
        location_action = menu.addAction("Open file location")
        paths = self.selected_source_paths(pane)
        reload_action.setEnabled(bool(paths))
        location_action.setEnabled(any(os.path.exists(path) for path in paths))
        chosen = menu.exec(pane.table_list_widget.mapToGlobal(position))
        if chosen is remove_action:
            self.remove_selected_sources(pane)
        elif chosen is reload_action:
            self.reload_selected_sources(pane)
        elif chosen is location_action:
            self.open_selected_file_locations(pane)

    def selected_source_paths(self, pane=None):
        pane = pane or self.active_selector_pane or self.selector_panes[0]
        paths = []
        seen = set()
        for item in pane.table_list_widget.selectedItems():
            data = item.data(QtCore.Qt.UserRole)
            path = ""
            if isinstance(data, tuple) and data[0] == "lazy":
                path = self.lazy_entries[data[1]].path
            elif isinstance(data, tuple) and data[0] == "bladed_project":
                path = data[1]
            elif isinstance(data, tuple) and data[0] == "table":
                path = self.tab_list[data[1]].filename
            if path:
                normalized = self.normalized_file_path(path)
                if normalized not in seen:
                    paths.append(path)
                    seen.add(normalized)
        return paths

    def _delete_table_indices(self, indices):
        indices = set(indices)
        if not indices:
            return
        old_to_new = {}
        kept = []
        for old_index, tab in enumerate(self.tab_list._tabs):
            if old_index in indices:
                continue
            old_to_new[old_index] = len(kept)
            kept.append(tab)
        self.tab_list._tabs = kept
        for entry in self.lazy_entries:
            entry.table_indices = [
                old_to_new[index]
                for index in entry.table_indices
                if index in old_to_new
            ]

    def remove_selected_sources(self, pane=None):
        pane = pane or self.active_selector_pane or self.selector_panes[0]
        selected_items = list(pane.table_list_widget.selectedItems())
        if not selected_items:
            return
        if self.lazy_entries:
            lazy_indices = sorted({
                data[1]
                for item in selected_items
                for data in [item.data(QtCore.Qt.UserRole)]
                if isinstance(data, tuple) and data[0] == "lazy"
            })
            table_indices = {
                table_index
                for lazy_index in lazy_indices
                for table_index in self.lazy_entries[lazy_index].table_indices
            }
            self.lazy_generation += 1
            self.lazy_load_queue = deque()
            self.lazy_warning_backlog = []
            self.lazy_selected_batch = set()
            self._delete_table_indices(table_indices)
            remove_set = set(lazy_indices)
            self.lazy_entries = [
                entry for index, entry in enumerate(self.lazy_entries)
                if index not in remove_set
            ]
            self.lazy_loaded_total = sum(entry.loaded for entry in self.lazy_entries)
            self.current_files = [entry.path for entry in self.lazy_entries]
            removed = len(lazy_indices)
        else:
            table_indices = set()
            for item in selected_items:
                data = item.data(QtCore.Qt.UserRole)
                if not isinstance(data, tuple):
                    continue
                if data[0] == "table":
                    table_indices.add(data[1])
                elif data[0] == "bladed_project":
                    path = self.normalized_file_path(data[1])
                    table_indices.update(
                        index for index, tab in enumerate(self.tab_list)
                        if tab.filename
                        and self.normalized_file_path(tab.filename) == path
                    )
            removed = len(table_indices)
            self._delete_table_indices(table_indices)
            self.current_files = list(dict.fromkeys(
                tab.filename for tab in self.tab_list if tab.filename
            ))
        self.clear()
        self.populate_tables()
        self.status_label.setText(
            "{:,} files indexed, {:,} loaded".format(
                len(self.lazy_entries), self.lazy_loaded_count()
            ) if self.lazy_entries else "{} tables loaded".format(len(self.tab_list))
        )
        self.statusBar().showMessage(
            "Removed {:,} simulation(s) from pyDatView".format(removed), 8000
        )

    def reload_selected_sources(self, pane=None):
        pane = pane or self.active_selector_pane or self.selector_panes[0]
        paths = self.selected_source_paths(pane)
        if not paths:
            self.statusBar().showMessage("Selected table has no source file", 5000)
            return
        normalized_paths = {self.normalized_file_path(path) for path in paths}
        if self.lazy_entries:
            selected_indices = [
                index for index, entry in enumerate(self.lazy_entries)
                if self.normalized_file_path(entry.path) in normalized_paths
            ]
            requests = {}
            table_indices = set()
            for index in selected_indices:
                entry = self.lazy_entries[index]
                table_indices.update(entry.table_indices)
                requests[index] = (
                    None if entry.full_loaded or not entry.loaded_column_indices
                    else tuple(sorted(entry.loaded_column_indices))
                )
            self.lazy_generation += 1
            self.lazy_load_queue = deque()
            self.lazy_warning_backlog = []
            self.lazy_selected_batch = set(selected_indices)
            self._delete_table_indices(table_indices)
            for index in selected_indices:
                entry = self.lazy_entries[index]
                entry.warning = ""
                entry.attempted = False
                entry.loading = False
                entry.columns = []
                entry.header_attempted = False
                entry.loaded_column_indices = set()
                entry.full_loaded = False
            self.lazy_loaded_total = sum(entry.loaded for entry in self.lazy_entries)
            selected_by_pane = [set(normalized_paths) for _ in self.visible_selector_panes()]
            self.populate_tables(selected_lazy_paths=selected_by_pane)
            self.begin_lazy_load_batch(len(selected_indices))
            for index in selected_indices:
                self.ensure_lazy_loaded(
                    index,
                    show_warning=False,
                    channel_indices=requests[index],
                )
            self.statusBar().showMessage(
                "Reloading {:,} simulation(s)".format(len(selected_indices)), 8000
            )
            return

        old_tabs = list(self.tab_list._tabs)
        replacements = {}
        warnings = []
        for path in paths:
            matching = [
                tab for tab in old_tabs
                if tab.filename
                and self.normalized_file_path(tab.filename)
                == self.normalized_file_path(path)
            ]
            if not matching:
                continue
            tabs, warning = self.tab_list._load_file_tabs(
                path,
                fileformat=matching[0].fileformat,
                bReload=True,
            )
            if warning:
                warnings.append(warning)
            if tabs:
                for old, new in zip(matching, tabs):
                    new.name = old.name
                    new.active_name = old.active_name
                replacements[self.normalized_file_path(path)] = tabs

        rebuilt = []
        inserted = set()
        for tab in old_tabs:
            key = (
                self.normalized_file_path(tab.filename) if tab.filename else None
            )
            if key not in replacements:
                rebuilt.append(tab)
            elif key not in inserted:
                rebuilt.extend(replacements[key])
                inserted.add(key)
        self.tab_list._tabs = rebuilt
        self.current_files = list(dict.fromkeys(
            tab.filename for tab in self.tab_list if tab.filename
        ))
        self.populate_tables()
        self.redraw()
        if warnings:
            QtWidgets.QMessageBox.warning(
                self, "Reload warnings", "\n\n".join(warnings[:5])
            )
        self.statusBar().showMessage(
            "Reloaded {:,} simulation(s)".format(len(replacements)), 8000
        )

    def open_selected_file_locations(self, pane=None):
        paths = self.selected_source_paths(pane)
        directories = []
        for path in paths:
            directory = os.path.dirname(os.path.abspath(path))
            if os.path.isdir(directory) and directory not in directories:
                directories.append(directory)
        for directory in directories:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(directory))
        if not directories:
            self.statusBar().showMessage("No source file location is available", 5000)

    def load_dfs(self, dataframes, names=None):
        if not isinstance(dataframes, list):
            dataframes = [dataframes]
        if names is None:
            names = ["df{}".format(i + 1) for i in range(len(dataframes))]
        if not isinstance(names, list):
            names = [names]
        self.lazy_generation += 1
        self.lazy_load_queue = deque()
        self.lazy_warning_backlog = []
        self.lazy_memory_reservations = {}
        self.lazy_entries = []
        self.lazy_item_widgets = {}
        self.lazy_loaded_total = 0
        self.lazy_selected_batch = set()
        self.lazy_selection_refresh_pending = False
        self.tab_list.from_dataframes(dataframes=dataframes, names=names, bAdd=False)
        self.populate_tables()
        self.status_label.setText("{} tables loaded".format(len(self.tab_list)))
        self.redraw()

    def reload_files(self):
        if self.lazy_entries:
            self.lazy_generation += 1
            self.lazy_load_queue = deque()
            self.lazy_warning_backlog = []
            self.lazy_memory_reservations = {}
            self.lazy_batch_total = 0
            self.lazy_batch_done = 0
            self.loading_progress.setVisible(False)
            self.set_loading_controls_enabled(True)
            self.lazy_loaded_total = 0
            self.lazy_selected_batch = set()
            self.lazy_selection_refresh_pending = False
            for entry in self.lazy_entries:
                entry.table_indices = []
                entry.warning = ""
                entry.attempted = False
                entry.loading = False
                entry.columns = []
                entry.header_attempted = False
                entry.loaded_column_indices = set()
                entry.full_loaded = False
            self.tab_list.clean()
            self.populate_tables()
            self.clear()
            self.status_label.setText("{:,} files indexed, 0 loaded".format(len(self.lazy_entries)))
            return
        filenames = sorted(set(f for f in self.current_files if f))
        if filenames:
            self.load_files(filenames, add=False)

    def populate_tables(self, selected_lazy_paths=None):
        visible = self.visible_selector_panes()
        names = self.tab_list.getDisplayTabNames() if not self.lazy_entries else []
        self.lazy_item_widgets = {}
        for pane_index, pane in enumerate(visible):
            pane.table_list_widget.blockSignals(True)
            pane.table_list_widget.clear()
            if self.lazy_entries:
                for i, entry in enumerate(self.lazy_entries):
                    item = QtWidgets.QListWidgetItem(self.lazy_item_text(entry))
                    item.setData(QtCore.Qt.UserRole, ("lazy", i))
                    pane.table_list_widget.addItem(item)
                    self.lazy_item_widgets.setdefault(i, []).append(item)
            else:
                displayed_projects = set()
                for i, tab in enumerate(self.tab_list):
                    if self.is_bladed_project_path(tab.filename):
                        project_path = os.path.abspath(tab.filename)
                        if project_path in displayed_projects:
                            continue
                        displayed_projects.add(project_path)
                        group_count = sum(
                            1 for candidate in self.tab_list
                            if os.path.abspath(candidate.filename) == project_path
                        )
                        item = QtWidgets.QListWidgetItem(os.path.basename(tab.filename))
                        item.setToolTip("{} Bladed variable groups".format(group_count))
                        item.setData(
                            QtCore.Qt.UserRole,
                            ("bladed_project", project_path),
                        )
                        pane.table_list_widget.addItem(item)
                        continue
                    item = QtWidgets.QListWidgetItem("{}  ({})".format(names[i], tab.shapestring))
                    item.setData(QtCore.Qt.UserRole, ("table", i))
                    pane.table_list_widget.addItem(item)
            restored_selection = False
            if self.lazy_entries and selected_lazy_paths is not None:
                paths = (
                    selected_lazy_paths[pane_index]
                    if pane_index < len(selected_lazy_paths)
                    else set()
                )
                for row in range(pane.table_list_widget.count()):
                    item = pane.table_list_widget.item(row)
                    data = item.data(QtCore.Qt.UserRole)
                    if (
                        isinstance(data, tuple)
                        and data[0] == "lazy"
                        and self.normalized_file_path(
                            self.lazy_entries[data[1]].path
                        ) in paths
                    ):
                        item.setSelected(True)
                        restored_selection = True
            if pane.table_list_widget.count() > 0 and not restored_selection:
                default_row = min(pane_index, pane.table_list_widget.count() - 1)
                pane.table_list_widget.item(default_row).setSelected(True)
            pane.table_list_widget.blockSignals(False)
        self.on_table_selection_changed()

    def selected_lazy_indices(self, pane=None):
        panes = [pane] if pane is not None else self.visible_selector_panes()
        indices = []
        seen = set()
        for p in panes:
            for item in p.table_list_widget.selectedItems():
                data = item.data(QtCore.Qt.UserRole)
                if isinstance(data, tuple) and data[0] == "lazy" and data[1] not in seen:
                    indices.append(data[1])
                    seen.add(data[1])
        return indices

    @staticmethod
    def is_bladed_project_path(path):
        return bool(path) and os.path.splitext(path)[1].lower() == ".$pj"

    def selected_bladed_project_paths(self, pane):
        paths = []
        seen = set()
        for item in pane.table_list_widget.selectedItems():
            data = item.data(QtCore.Qt.UserRole)
            path = None
            if isinstance(data, tuple) and data[0] == "bladed_project":
                path = data[1]
            elif isinstance(data, tuple) and data[0] == "lazy":
                entry = self.lazy_entries[data[1]]
                if entry.loaded and self.is_bladed_project_path(entry.path):
                    path = entry.path
            elif isinstance(data, tuple) and data[0] == "table":
                tab = self.tab_list[data[1]]
                if self.is_bladed_project_path(tab.filename):
                    path = tab.filename
            if path:
                normalized = os.path.abspath(path)
                if normalized not in seen:
                    paths.append(normalized)
                    seen.add(normalized)
        return paths

    def selected_bladed_group(self, pane):
        if pane.bladed_dataset_combo.isHidden():
            return "__all__"
        return pane.bladed_dataset_combo.currentData() or "__all__"

    def bladed_project_table_indices(self, pane, group=None):
        paths = set(self.selected_bladed_project_paths(pane))
        if not paths:
            return []
        group = self.selected_bladed_group(pane) if group is None else group
        return [
            i for i, tab in enumerate(self.tab_list)
            if os.path.abspath(tab.filename) in paths
            and (group == "__all__" or tab.nickname == group)
        ]

    def selected_table_indices(self, load=True, show_warning=False, pane=None):
        panes = [pane] if pane is not None else self.visible_selector_panes()
        indices = []
        seen = set()
        for p in panes:
            project_paths = set(self.selected_bladed_project_paths(p))
            for table_index in self.bladed_project_table_indices(p):
                if table_index not in seen:
                    indices.append(table_index)
                    seen.add(table_index)
            for item in p.table_list_widget.selectedItems():
                data = item.data(QtCore.Qt.UserRole)
                if isinstance(data, tuple) and data[0] == "table":
                    if data[1] not in seen:
                        indices.append(data[1])
                        seen.add(data[1])
                elif isinstance(data, tuple) and data[0] == "lazy":
                    entry = self.lazy_entries[data[1]]
                    if os.path.abspath(entry.path) in project_paths:
                        continue
                    if entry.loaded:
                        for table_index in entry.table_indices:
                            if (
                                getattr(p, "dataset_mode", None) == "lazy"
                                and self.tab_list[table_index].nickname
                                != p.bladed_dataset_combo.currentData()
                            ):
                                continue
                            if table_index not in seen:
                                indices.append(table_index)
                                seen.add(table_index)
                    elif load:
                        for table_index in self.ensure_lazy_loaded(data[1], show_warning=show_warning):
                            if table_index not in seen:
                                indices.append(table_index)
                                seen.add(table_index)
        return indices

    def on_table_selection_changed(self, active_pane=None):
        if active_pane in self.visible_selector_panes():
            self.active_selector_pane = active_pane
        for pane in self.visible_selector_panes():
            self.populate_bladed_datasets(pane)
            self.populate_columns(pane)
        self.update_table_preview()
        self.update_file_info()
        self.on_selection_changed()
