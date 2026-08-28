import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from pydatview.io.fast_output_file import FASTOutputFile, writeBinary
from pydatview.io.load_estimates import estimate_decoded_load_bytes


class TestFastSelectiveLoading(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.time = np.linspace(0.0, 10.0, 1001)
        self.data = np.column_stack(
            (
                self.time,
                np.sin(self.time),
                np.cos(self.time),
                self.time ** 2,
            )
        )
        self.names = ["Time", "Sin", "Cos", "Square"]
        self.units = ["s", "-", "-", "s^2"]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_binary_selected_channels_match_full_read(self):
        for file_id in (2, 4):
            with self.subTest(file_id=file_id):
                path = os.path.join(
                    self.temp_dir.name,
                    "selected-{}.outb".format(file_id),
                )
                writeBinary(
                    path,
                    self.data,
                    self.names,
                    self.units,
                    fileID=file_id,
                )

                full = FASTOutputFile(path)
                selected = FASTOutputFile(path, channel_indices=[0, 2])

                self.assertEqual(
                    list(selected.data.columns),
                    ["Time_[s]", "Cos_[-]"],
                )
                np.testing.assert_allclose(
                    selected.data.values,
                    full.data.iloc[:, [0, 2]].values,
                )
                native = selected.get_numpy_plot_data()
                self.assertIsNotNone(native)
                self.assertEqual(native[0].shape, (1001, 2))

    def test_ascii_selected_channels_match_full_read(self):
        path = os.path.join(self.temp_dir.name, "selected.out")
        with open(path, "w", encoding="ascii") as stream:
            stream.write("Synthetic OpenFAST output\n")
            stream.write("Time Sin Cos Square\n")
            stream.write("(s) (-) (-) (s^2)\n")
            np.savetxt(stream, self.data)

        full = FASTOutputFile(path)
        selected = FASTOutputFile(path, channel_indices=[0, 3])

        self.assertEqual(
            list(selected.data.columns),
            ["Time_[s]", "Square_[s^2]"],
        )
        np.testing.assert_allclose(
            selected.data.values,
            full.data.iloc[:, [0, 3]].values,
        )

    def test_binary_memory_estimate_uses_decoded_matrix_shape(self):
        path = os.path.join(self.temp_dir.name, "estimate.outb")
        writeBinary(
            path,
            self.data,
            self.names,
            self.units,
            fileID=4,
        )

        estimated = estimate_decoded_load_bytes(
            path,
            SimpleNamespace(name='FAST output file'),
        )

        self.assertEqual(estimated, self.data.size * 8)


if __name__ == "__main__":
    unittest.main()
