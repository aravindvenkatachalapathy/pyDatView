import unittest

import numpy as np
import pandas as pd

from pydatview.tools.pandalib import changeUnits


class TestUnitConversionSafety(unittest.TestCase):
    def test_validation_prevents_partial_conversion(self):
        frame = pd.DataFrame({
            "Force [N]": [1000.0, 2000.0],
            "Power [W]": ["bad", "data"],
        })
        original = frame.copy()

        with self.assertRaisesRegex(TypeError, "Power.*nonnumeric|nonnumeric.*Power"):
            changeUnits(frame, flavor="WE")

        pd.testing.assert_frame_equal(frame, original)

    def test_milliwatt_and_millinewton_are_not_megawatt_or_meganewton(self):
        frame = pd.DataFrame({
            "Power [mW]": [1.0, 2.0],
            "Force [mN]": [3.0, 4.0],
        })

        changeUnits(frame, flavor="WE")

        self.assertEqual(list(frame.columns), ["Power [mW]", "Force [mN]"])
        np.testing.assert_allclose(frame["Power [mW]"], [1.0, 2.0])
        np.testing.assert_allclose(frame["Force [mN]"], [3.0, 4.0])

    def test_conversion_rejects_new_duplicate_column_names(self):
        frame = pd.DataFrame(
            [[1000.0, 7.0]],
            columns=["Force [N]", "Force [kN]"],
        )
        original = frame.copy()

        with self.assertRaisesRegex(ValueError, "duplicate column"):
            changeUnits(frame, flavor="WE")

        pd.testing.assert_frame_equal(frame, original)

    def test_existing_duplicate_source_columns_can_convert_together(self):
        frame = pd.DataFrame(
            [[1000.0, 2000.0]],
            columns=["Force [N]", "Force [N]"],
        )

        changeUnits(frame, flavor="WE")

        self.assertEqual(list(frame.columns), ["Force [kN]", "Force [kN]"])
        np.testing.assert_allclose(frame.iloc[0, :], [1.0, 2.0])

    def test_single_letter_variable_without_separator_is_converted(self):
        frame = pd.DataFrame({"F[N]": [1000.0, 2500.0]})

        changeUnits(frame, flavor="WE")

        self.assertEqual(list(frame.columns), ["F[kN]"])
        np.testing.assert_allclose(frame["F[kN]"], [1.0, 2.5])


if __name__ == "__main__":
    unittest.main()
