#-*- coding: utf-8 -*-
"""
@brief      test log(time=4s)
"""

import sys
import os
import unittest
import pandas
import numpy


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
from src.pandas_streaming.df import dataframe_unfold


class TestDataFrameHelpersSimple(ExtTestCase):

    def test_unfold(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        df = pandas.DataFrame([dict(a=1, b="e,f"),
                               dict(a=2, b="g"),
                               dict(a=3)])
        df2 = dataframe_unfold(df, "b")

        exp = pandas.DataFrame([dict(a=1, b="e,f", b_unfold="e"),
                                dict(a=1, b="e,f", b_unfold="f"),
                                dict(a=2, b="g", b_unfold="g"),
                                dict(a=3)])
        self.assertEqualDataFrame(df2, exp)

        # fold
        folded = df2.groupby('a').apply(lambda row: ','.join(
            row['b_unfold'].dropna()) if len(row['b_unfold'].dropna()) > 0 else numpy.nan)
        bf = folded.reset_index(drop=False)
        bf.columns = ['a', 'b']
        self.assertEqualDataFrame(df, bf)


if __name__ == "__main__":
    unittest.main()
