import unittest

import numpy as np

from pydatview.plugins.plotdata_sampler import samplerAction, samplerXY


class TestSampler(unittest.TestCase):
    def test_headless_action(self):
        action = samplerAction(data={"active": False, "name": "Every n", "param": 2})
        self.assertIsNone(action.guiEditorClass)
        x = np.arange(8.0)
        x_new, y_new = samplerXY(x, x * 2, action.data)
        np.testing.assert_array_equal(x_new, np.array([0.0, 2.0, 4.0, 6.0]))
        np.testing.assert_array_equal(y_new, x_new * 2)


if __name__ == "__main__":
    unittest.main()
