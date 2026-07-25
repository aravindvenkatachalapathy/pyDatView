import unittest

import numpy as np

from pydatview.plugins.plotdata_filter import filterAction, filterXY


class TestFilter(unittest.TestCase):
    def test_headless_action(self):
        action = filterAction(data={"active": False, "name": "Moving average", "param": 3})
        self.assertIsNone(action.guiEditorClass)
        x = np.arange(8.0)
        x_new, y_new = filterXY(x, x, action.data)
        np.testing.assert_array_equal(x_new, x)
        self.assertEqual(y_new.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
