# -*- coding: utf-8 -*-

import unittest
from collections import Counter
import pandas
from pandas_streaming.ext_test_case import ExtTestCase
from pandas_streaming.df import train_test_apart_stratify


class TestConnexSplitCat(ExtTestCase):
    def test_cat_strat(self):
        df = pandas.DataFrame(
            [
                dict(a=1, b="e"),
                dict(a=2, b="e"),
                dict(a=4, b="f"),
                dict(a=8, b="f"),
                dict(a=32, b="f"),
                dict(a=16, b="f"),
            ]
        )

        train, test = train_test_apart_stratify(
            df, group="a", stratify="b", test_size=0.5
        )
        self.assertEqual(train.shape[1], test.shape[1])
        self.assertEqual(train.shape[0] + test.shape[0], df.shape[0])
        c1 = Counter(train["b"])
        c2 = Counter(train["b"])
        self.assertEqual(c1, c2)

        self.assertRaise(
            lambda: train_test_apart_stratify(
                df, group=None, stratify="b", test_size=0.5
            ),
            ValueError,
        )
        self.assertRaise(
            lambda: train_test_apart_stratify(df, group="b", test_size=0.5), ValueError
        )

    def test_cat_strat_sorted(self):
        df = pandas.DataFrame(
            [
                dict(a=1, b="e"),
                dict(a=2, b="e"),
                dict(a=4, b="f"),
                dict(a=8, b="f"),
                dict(a=32, b="f"),
                dict(a=16, b="f"),
            ]
        )

        train, test = train_test_apart_stratify(
            df, group="a", stratify="b", test_size=0.5, sorted_indices=True
        )
        self.assertEqual(train.shape[1], test.shape[1])
        self.assertEqual(train.shape[0] + test.shape[0], df.shape[0])
        c1 = Counter(train["b"])
        c2 = Counter(train["b"])
        self.assertEqual(c1, c2)

        self.assertRaise(
            lambda: train_test_apart_stratify(
                df, group=None, stratify="b", test_size=0.5, sorted_indices=True
            ),
            ValueError,
        )
        self.assertRaise(
            lambda: train_test_apart_stratify(df, group="b", test_size=0.5), ValueError
        )

    def test_cat_strat_multi(self):
        df = pandas.DataFrame(
            [
                dict(a=1, b="e"),
                dict(a=1, b="f"),
                dict(a=2, b="e"),
                dict(a=2, b="f"),
            ]
        )

        train, test = train_test_apart_stratify(
            df, group="a", stratify="b", test_size=0.5
        )
        self.assertEqual(train.shape[1], test.shape[1])
        self.assertEqual(train.shape[0] + test.shape[0], df.shape[0])
        c1 = Counter(train["b"])
        c2 = Counter(train["b"])
        self.assertEqual(c1, c2)
        self.assertEqual(len(set(train["a"])), 1)
        self.assertEqual(len(set(test["a"])), 1)
        self.assertTrue(set(train["a"]) != set(test["a"]))

    def test_cat_strat_multi_force(self):
        df = pandas.DataFrame(
            [
                dict(a=1, b="e"),
                dict(a=1, b="f"),
                dict(a=2, b="e"),
                dict(a=2, b="f"),
            ]
        )

        train, test = train_test_apart_stratify(
            df, group="a", stratify="b", test_size=0.1, force=True
        )
        self.assertEqual(train.shape[1], test.shape[1])
        self.assertEqual(train.shape[0] + test.shape[0], df.shape[0])
        c1 = Counter(train["b"])
        c2 = Counter(train["b"])
        self.assertEqual(c1, c2)
        self.assertEqual(len(set(train["a"])), 1)
        self.assertEqual(len(set(test["a"])), 1)
        self.assertTrue(set(train["a"]) != set(test["a"]))


if __name__ == "__main__":
    unittest.main()
