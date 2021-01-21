# -*- coding: utf-8 -*-
"""
@brief      test log(time=4s)
"""
import unittest
import pandas
import numpy
from pyquickhelper.pycode import ExtTestCase
from pandas_streaming.df import dataframe_unfold
from pandas_streaming.df.dataframe_helpers import hash_int, hash_str, hash_float


class TestDataFrameHelpersSimple(ExtTestCase):

    def test_unfold(self):
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

    def test_hash_except(self):
        self.assertRaise(lambda: hash_int(0.1, 3),
                         ValueError, "numpy.nan expected")
        r = hash_int(numpy.nan, 3)
        self.assertTrue(numpy.isnan(r))

        self.assertRaise(lambda: hash_str(0.1, 3),
                         ValueError, "numpy.nan expected")
        r = hash_str(numpy.nan, 3)
        self.assertTrue(numpy.isnan(r))

        self.assertRaise(lambda: hash_float("0.1", 3), TypeError, "isnan")
        r = hash_float(numpy.nan, 3)
        self.assertTrue(numpy.isnan(r))
        r = hash_str("3", 100)
        self.assertLess(len(r), 100)


if __name__ == "__main__":
    unittest.main()
