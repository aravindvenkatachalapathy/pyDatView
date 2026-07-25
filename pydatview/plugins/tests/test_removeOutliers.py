import unittest

import numpy as np

from pydatview.plugins.plotdata_removeOutliers import (
    removeOutliersAction,
    removeOutliersXY,
)


class TestRemoveOutliers(unittest.TestCase):
    def test_headless_action(self):
        action = removeOutliersAction(data={"active": False, "medianDeviation": 1})
        self.assertIsNone(action.guiEditorClass)
        x = np.arange(5.0)
        x_new, y_new = removeOutliersXY(
            x, np.array([0.0, 0.0, 100.0, 0.0, 0.0]), action.data
        )
        self.assertEqual(x_new.shape, y_new.shape)
        self.assertLessEqual(len(x_new), len(x))
        self.assertTrue(np.isfinite(x_new).all())
        self.assertTrue(np.isfinite(y_new).all())


if __name__ == "__main__":
    unittest.main()
