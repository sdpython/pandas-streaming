# -*- coding: utf-8 -*-
"""
@brief      test log(time=4s)
"""

import sys
import os
import unittest
import numpy
import pandas


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
from src.pandas_streaming.df import dataframe_hash_columns


class TestDataFrameHelpers(ExtTestCase):

    def test_hash_columns(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        df = pandas.DataFrame([dict(a=1, b="e", c=5.6, ind="a1", ai=1),
                               dict(b="f", c=5.7, ind="a2", ai=2),
                               dict(a=4, b="g", ind="a3", ai=3),
                               dict(a=8, b="h", c=5.9, ai=4),
                               dict(a=16, b="i", c=6.2, ind="a5", ai=5)])
        df2 = dataframe_hash_columns(df)
        self.assertEqual(df2.shape, df.shape)
        for j in range(df.shape[1]):
            self.assertEqual(df.columns[j], df2.columns[j])
            self.assertEqual(df.dtypes[j], df2.dtypes[j])
            for i in range(df.shape[0]):
                v1 = df.iloc[i, j]
                v2 = df2.iloc[i, j]
                if isinstance(v1, float):
                    if numpy.isnan(v1):
                        self.assertTrue(numpy.isnan(v2))
                    else:
                        self.assertEqual(type(v1), type(v2))
                else:
                    self.assertEqual(type(v1), type(v2))

    def test_hash_columns_bigger(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        data = os.path.join(os.path.dirname(__file__), "data")
        name = os.path.join(data, "buggy_hash.csv")
        df = pandas.read_csv(name, sep="\t", encoding="utf-8")
        df2 = dataframe_hash_columns(df)
        self.assertEqual(df.shape, df2.shape)


if __name__ == "__main__":
    unittest.main()
