import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TestPublicationExport(unittest.TestCase):
    def test_minmax_downsample_preserves_extrema(self):
        from pydatview.qt_publication import _minmax_downsample

        x = np.arange(1000, dtype=float)
        y = np.sin(x / 20.0)
        y[123] = 25.0
        y[877] = -30.0
        reduced_x, reduced_y = _minmax_downsample(x, y, 80)

        self.assertLessEqual(len(reduced_x), 80)
        self.assertEqual(reduced_x[0], x[0])
        self.assertEqual(reduced_x[-1], x[-1])
        self.assertEqual(np.max(reduced_y), 25.0)
        self.assertEqual(np.min(reduced_y), -30.0)

    def test_tex_escape_preserves_math_segments(self):
        from pydatview.qt_publication import _tex_escape

        self.assertEqual(
            _tex_escape("case_01: $C_P$ [%]"),
            r"case\_01: $C_P$ [\%]",
        )

    def test_export_raster_and_vector_formats(self):
        import matplotlib.image as mpimg

        from pydatview.qt_publication import (
            PublicationExportOptions,
            export_publication_plot,
        )

        x = np.linspace(0.0, 10.0, 10000)
        plot_data = [
            SimpleNamespace(
                x=x,
                y=np.sin(x),
                sx="Time [s]",
                sy="Load [kN]",
                syl="case_01",
            ),
            SimpleNamespace(
                x=x,
                y=np.cos(x),
                sx="Time [s]",
                sy="Load [kN]",
                syl="case_02",
            ),
        ]
        with tempfile.TemporaryDirectory() as folder:
            png_path = os.path.join(folder, "plot.png")
            options = PublicationExportOptions(
                path=png_path,
                width=4.0,
                height=3.0,
                dpi=100,
                max_points=1000,
                grid=True,
                legend=True,
            )
            export_publication_plot(plot_data, options)
            image = mpimg.imread(png_path)
            self.assertEqual(image.shape[:2], (300, 400))

            for extension in ("svg", "pdf"):
                path = os.path.join(folder, "plot." + extension)
                export_publication_plot(
                    plot_data,
                    PublicationExportOptions(
                        path=path,
                        width=4.0,
                        height=3.0,
                        max_points=1000,
                        x_label="Elapsed time [s]",
                        y_label="Rotor load [kN]",
                        legend=True,
                        legend_labels=("Baseline", "Controlled"),
                    ),
                )
                self.assertGreater(os.path.getsize(path), 1000)
                if extension == "svg":
                    with open(path, encoding="utf-8") as exported:
                        content = exported.read()
                    self.assertIn("Elapsed time [s]", content)
                    self.assertIn("Rotor load [kN]", content)
                    self.assertIn("Baseline", content)
                    self.assertIn("Controlled", content)
                    self.assertNotIn("case_01", content)

    def test_export_legend_defaults_to_compact_set_names(self):
        from pydatview.qt_publication import (
            PublicationExportOptions,
            export_publication_plot,
        )

        x = np.arange(4, dtype=float)
        plot_data = [
            SimpleNamespace(
                x=x,
                y=x,
                sx="Time [s]",
                sy="Load [N]",
                syl="/very/long/path/case_01.out | Load [N]",
            ),
            SimpleNamespace(
                x=x,
                y=x + 1,
                sx="Time [s]",
                sy="Load [N]",
                syl="/very/long/path/case_02.out | Load [N]",
            ),
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "legend.svg")
            export_publication_plot(
                plot_data,
                PublicationExportOptions(path=path, legend=True),
            )
            with open(path, encoding="utf-8") as exported:
                content = exported.read()

        self.assertIn("Set 1", content)
        self.assertIn("Set 2", content)
        self.assertNotIn("/very/long/path", content)

    def test_dialog_allows_legend_label_renaming(self):
        from pydatview.qt_compat import QtWidgets
        from pydatview.qt_publication import PublicationExportDialog

        app = (
            QtWidgets.QApplication.instance()
            or QtWidgets.QApplication([])
        )
        dialog = PublicationExportDialog(initial={
            "legend": True,
            "legend_sources": ["/simulations/case_01.out"],
            "legend_labels": ["Set 1"],
        })
        dialog.legend_table.item(0, 1).setText("Baseline")

        self.assertEqual(
            dialog.options().legend_labels, ("Baseline",)
        )
        self.assertEqual(
            dialog.legend_table.item(0, 0).toolTip(),
            "/simulations/case_01.out",
        )
        dialog.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
