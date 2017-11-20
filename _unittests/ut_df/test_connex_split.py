#-*- coding: utf-8 -*-
"""
@brief      test log(time=4s)
"""

import sys
import os
import unittest
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
from src.pandas_streaming.df import dataframe_shuffle, train_test_split_weights, train_test_connex_split


class TestConnexSplit(ExtTestCase):

    def test_shuffle(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        df = pandas.DataFrame([dict(a=1, b="e", c=5.6, ind="a1"),
                               dict(a=2, b="f", c=5.7, ind="a2"),
                               dict(a=4, b="g", c=5.8, ind="a3"),
                               dict(a=8, b="h", c=5.9, ind="a4"),
                               dict(a=16, b="i", c=6.2, ind="a5")])
        shuffled = dataframe_shuffle(df, seed=0)
        sorted = shuffled.sort_values('a')
        self.assertEqualDataFrame(df, sorted)

        df2 = df.set_index('ind')
        shuffled = dataframe_shuffle(df2, seed=0)
        sorted = shuffled.sort_values('a')
        self.assertEqualDataFrame(df2, sorted)

        df2 = df.set_index(['ind', 'c'])
        shuffled = dataframe_shuffle(df2, seed=0)
        sorted = shuffled.sort_values('a')
        self.assertEqualDataFrame(df2, sorted)

    def test_split_weights(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        df = pandas.DataFrame([dict(a=1, b="e", c=1),
                               dict(a=2, b="f", c=1),
                               dict(a=4, b="g", c=1),
                               dict(a=8, b="h", c=1),
                               dict(a=12, b="h", c=1),
                               dict(a=16, b="i", c=1)])

        train, test = train_test_split_weights(df, test_size=0.5)
        self.assertEqual(train.shape[1], test.shape[1])
        self.assertEqual(train.shape[0] + test.shape[0], df.shape[0])

        train, test = train_test_split_weights(df, test_size=0.5, weights='c')
        self.assertEqual(train.shape[1], test.shape[1])
        self.assertEqual(train.shape[0] + test.shape[0], df.shape[0])

        train, test = train_test_split_weights(
            df, test_size=0.5, weights=df['c'])
        self.assertEqual(train.shape[1], test.shape[1])
        self.assertEqual(train.shape[0] + test.shape[0], df.shape[0])

        df = pandas.DataFrame([dict(a=1, b="e", c=1),
                               dict(a=2, b="f", c=2),
                               dict(a=4, b="g", c=3),
                               dict(a=8, b="h", c=1),
                               dict(a=12, b="h", c=2),
                               dict(a=16, b="i", c=3)])

        train, test = train_test_split_weights(df, test_size=0.5, weights='c',
                                               fail_imbalanced=0.4)
        self.assertEqual(train.shape[1], test.shape[1])
        self.assertEqual(train.shape[0] + test.shape[0], df.shape[0])
        w1, w2 = train['c'].sum(), test['c'].sum()
        delta = abs(w1 - w2) / (w1 + w2)
        self.assertGreater(0.4, delta)

    def test_split_connex(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        df = pandas.DataFrame([dict(user="UA", prod="PA", card="C1"),
                               dict(user="UA", prod="PB", card="C1"),
                               dict(user="UB", prod="PC", card="C2"),
                               dict(user="UB", prod="PD", card="C2"),
                               dict(user="UC", prod="PE", card="C3"),
                               dict(user="UC", prod="PF", card="C4"),
                               dict(user="UD", prod="PG", card="C5"),
                               ])

        train, test = train_test_connex_split(df, test_size=0.5,
                                              groups=['user', 'prod', 'card'],
                                              fail_imbalanced=0.4, fLOG=fLOG)

        self.assertEqual(train.shape[0] + test.shape[0], df.shape[0])
        for col in ['user', 'prod', 'card']:
            s1 = set(train[col])
            s2 = set(test[col])
            if s1 & s2:
                raise Exception(
                    'Non empty intersection {0} & {1}\n{2}\n{3}'.format(s1, s2, train, test))

    def test_split_connex2(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        df = pandas.DataFrame([dict(user="UA", prod="PAA", card="C1"),
                               dict(user="UA", prod="PB", card="C1"),
                               dict(user="UB", prod="PC", card="C2"),
                               dict(user="UB", prod="PD", card="C2"),
                               dict(user="UC", prod="PAA", card="C3"),
                               dict(user="UC", prod="PF", card="C4"),
                               dict(user="UD", prod="PG", card="C5"),
                               ])

        train, test = train_test_connex_split(df, test_size=0.5,
                                              groups=['user', 'prod', 'card'],
                                              fail_imbalanced=0.4, fLOG=fLOG)

        self.assertEqual(train.shape[0] + test.shape[0], df.shape[0])
        for col in ['user', 'prod', 'card']:
            s1 = set(train[col])
            s2 = set(test[col])
            if s1 & s2:
                raise Exception(
                    'Non empty intersection {0} & {1}\n{2}\n{3}'.format(s1, s2, train, test))


if __name__ == "__main__":
    unittest.main()
    # TestStreamingDataFrame().test_train_test_split_streaming_tiny()
