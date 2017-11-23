#-*- coding: utf-8 -*-
"""
@brief      test log(time=4s)
"""

import sys
import os
import unittest
from collections import Counter
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
from src.pandas_streaming.df import train_test_apart_stratify


class TestConnexSplitCat(ExtTestCase):

    def test_cat_strat(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        df = pandas.DataFrame([dict(a=1, b="e"),
                               dict(a=2, b="e"),
                               dict(a=4, b="f"),
                               dict(a=8, b="f"),
                               dict(a=32, b="f"),
                               dict(a=16, b="f")])

        train, test = train_test_apart_stratify(
            df, group="a", stratify="b", test_size=0.5)
        self.assertEqual(train.shape[1], test.shape[1])
        self.assertEqual(train.shape[0] + test.shape[0], df.shape[0])
        c1 = Counter(train["b"])
        c2 = Counter(train["b"])
        self.assertEqual(c1, c2)

        self.assertRaise(lambda: train_test_apart_stratify(df, group=None, stratify="b", test_size=0.5),
                         ValueError)
        self.assertRaise(lambda: train_test_apart_stratify(df, group="b", test_size=0.5),
                         ValueError)

    def test_cat_strat_multi(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        df = pandas.DataFrame([dict(a=1, b="e"),
                               dict(a=1, b="f"),
                               dict(a=2, b="e"),
                               dict(a=2, b="f"),
                               ])

        train, test = train_test_apart_stratify(
            df, group="a", stratify="b", test_size=0.5)
        self.assertEqual(train.shape[1], test.shape[1])
        self.assertEqual(train.shape[0] + test.shape[0], df.shape[0])
        c1 = Counter(train["b"])
        c2 = Counter(train["b"])
        self.assertEqual(c1, c2)
        self.assertEqual(len(set(train['a'])), 1)
        self.assertEqual(len(set(test['a'])), 1)
        self.assertTrue(set(train['a']) != set(test['a']))

    def test_cat_strat_multi_force(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        df = pandas.DataFrame([dict(a=1, b="e"),
                               dict(a=1, b="f"),
                               dict(a=2, b="e"),
                               dict(a=2, b="f"),
                               ])

        train, test = train_test_apart_stratify(
            df, group="a", stratify="b", test_size=0.1, force=True)
        self.assertEqual(train.shape[1], test.shape[1])
        self.assertEqual(train.shape[0] + test.shape[0], df.shape[0])
        c1 = Counter(train["b"])
        c2 = Counter(train["b"])
        self.assertEqual(c1, c2)
        self.assertEqual(len(set(train['a'])), 1)
        self.assertEqual(len(set(test['a'])), 1)
        self.assertTrue(set(train['a']) != set(test['a']))


if __name__ == "__main__":
    unittest.main()
