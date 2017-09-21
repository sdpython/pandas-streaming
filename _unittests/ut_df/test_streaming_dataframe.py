#-*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""

import sys
import os
import unittest


try:
    import pyquickhelper as skip_
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..",
                "..",
                "pyquickhelper",
                "src")))
    if path not in sys.path:
        sys.path.append(path)
    import pyquickhelper as skip_


try:
    import src
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..")))
    if path not in sys.path:
        sys.path.append(path)
    import src

from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from src.pandas_streaming.data import dummy_streaming_dataframe
from src.pandas_streaming.exc import StreamingInefficientException


class TestStreamingDataFrame(ExtTestCase):

    def test_shape(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        sdf = dummy_streaming_dataframe(100)
        dfs = [df for df in sdf]
        self.assertEqual(len(dfs), 10)
        self.assertEqual(len(dfs), 10)
        shape = sdf.shape
        self.assertEqual(shape, (100, 2))
        self.assertRaise(lambda: sdf.sort_values(
            "r"), StreamingInefficientException)


if __name__ == "__main__":
    unittest.main()
