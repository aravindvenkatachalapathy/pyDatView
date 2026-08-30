import unittest

import numpy as np

from pydatview.plotdata import PlotData
from pydatview.qt_stats import box_plot_data


def _curve(values, source, *, selection=0, label="Load [N]"):
    curve = PlotData(
        np.arange(len(values), dtype=float),
        np.asarray(values),
        "Time [s]",
        label,
    )
    curve.filename = source
    curve.st = source
    curve.it = selection
    curve.pane_index = 0
    curve.selection_index = selection
    return curve


class TestBoxPlotData(unittest.TestCase):
    def test_creates_one_distribution_box_per_file(self):
        result = box_plot_data([
            _curve([1.0, 2.0, 3.0, 4.0], "/runs/a.out"),
            _curve([2.0, 4.0, 6.0, 8.0], "/runs/b.out"),
        ])

        self.assertEqual(len(result), 2)
        self.assertEqual([box.boxplot_label for box in result], ["a.out", "b.out"])
        np.testing.assert_allclose([box.x[0] for box in result], [0.0, 1.0])
        self.assertEqual(result[0].boxplot_stats, {
            "minimum": 1.0,
            "q1": 1.75,
            "median": 2.5,
            "mean": 2.5,
            "q3": 3.25,
            "maximum": 4.0,
        })
        self.assertEqual(result[0].color_index, 0)
        self.assertEqual(result[1].color_index, 1)

    def test_ignores_nonfinite_values_and_preserves_source_stats_data(self):
        source = _curve([1.0, np.nan, 5.0], "case.out")

        result = box_plot_data([source])

        self.assertEqual(result[0].boxplot_stats["mean"], 3.0)
        self.assertIs(result[0].y0, source.y)
        self.assertIs(result[0].x0, source.x)

    def test_multiple_variables_are_identified_in_x_labels(self):
        result = box_plot_data([
            _curve([1.0, 2.0], "case.out", selection=0, label="Load [N]"),
            _curve([3.0, 4.0], "case.out", selection=1, label="Power [W]"),
        ])

        self.assertEqual(
            [box.boxplot_label for box in result],
            ["case.out | Load", "case.out | Power"],
        )
        self.assertEqual(result[0].color_index, result[1].color_index)

    def test_rejects_non_numeric_variables(self):
        curve = _curve([1.0, 2.0], "case.out")
        curve.yIsString = True
        with self.assertRaisesRegex(ValueError, "numeric Y variable"):
            box_plot_data([curve])


if __name__ == "__main__":
    unittest.main()
