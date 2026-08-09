import os
import struct
import tempfile
import unittest
import warnings

import numpy as np


class TestFASTWindFiles(unittest.TestCase):
    def test_uniform_wnd_accepts_documented_comments_and_separators(self):
        from pydatview.io.fast_wind_file import FASTWndFile

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "uniform.wnd")
            with open(path, "w", encoding="ascii") as stream:
                stream.write("Uniform wind # generated for test\n")
                stream.write("% time, speed, direction, vertical, shears, gust\n")
                stream.write("0, 8, 0, 0, 0, 0.2, 0, 0\n")
                stream.write("1  9  5  0  0  0.2  0  0\n")

            frame = FASTWndFile(path).toDataFrame()
            self.assertEqual(frame.shape, (2, 8))
            self.assertEqual(frame.columns[1], "WindSpeed_[m/s]")
            np.testing.assert_allclose(frame["WindSpeed_[m/s]"], [8.0, 9.0])

    def test_binary_bladed_wnd_is_converted_to_wind_tables(self):
        from pydatview.io.fast_wind_file import FASTWndFile

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "full_field.wnd")
            mean_speed = 10.0
            ti_percent = np.array([10.0, 8.0, 5.0], dtype=np.float32)
            nt, ny, nz = 4, 2, 2
            raw = np.arange(3 * nt * ny * nz, dtype=np.int16).reshape(
                (3, nt, ny, nz)
            )
            with open(path, "wb") as stream:
                stream.write(struct.pack("<hh", -99, 4))
                stream.write(struct.pack("<i6f", 3, 45.0, 0.03, 90.0, *ti_percent))
                stream.write(struct.pack(
                    "<3fi4f4i",
                    5.0, 6.0, 2.0, nt // 2, mean_speed,
                    0.0, 0.0, 0.0, 0, 1, nz, ny,
                ))
                stream.write(struct.pack("<6f", *([0.0] * 6)))
                stream.write(raw.transpose(0, 2, 3, 1).tobytes(order="F"))

            wind = FASTWndFile(path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                tables = wind.toDataFrame()
            self.assertIn("VertProfile", tables)
            self.assertIn("ZMidLine", tables)
            self.assertEqual(tables["ZMidLine"].shape, (nt, 4))
            self.assertAlmostEqual(tables["ZMidLine"]["t_[s]"].iloc[-1], 0.6)
            self.assertTrue(np.all(tables["ZMidLine"]["u_[m/s]"] >= mean_speed))


if __name__ == "__main__":
    unittest.main()
