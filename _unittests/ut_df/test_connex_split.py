# -*- coding: utf-8 -*-
"""
@brief      test log(time=4s)
"""
import unittest
import pandas
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from pandas_streaming.df import dataframe_shuffle, train_test_split_weights, train_test_connex_split


class TestConnexSplit(ExtTestCase):

    def test_shuffle(self):
        df = pandas.DataFrame([dict(a=1, b="e", c=5.6, ind="a1"),
                               dict(a=2, b="f", c=5.7, ind="a2"),
                               dict(a=4, b="g", c=5.8, ind="a3"),
                               dict(a=8, b="h", c=5.9, ind="a4"),
                               dict(a=16, b="i", c=6.2, ind="a5")])
        shuffled = dataframe_shuffle(df, random_state=0)
        sorted_ = shuffled.sort_values('a')
        self.assertEqualDataFrame(df, sorted_)

        df2 = df.set_index('ind')
        shuffled = dataframe_shuffle(df2, random_state=0)
        sorted_ = shuffled.sort_values('a')
        self.assertEqualDataFrame(df2, sorted_)

        df2 = df.set_index(['ind', 'c'])
        shuffled = dataframe_shuffle(df2, random_state=0)
        sorted_ = shuffled.sort_values('a')
        self.assertEqualDataFrame(df2, sorted_)

    def test_split_weights_errors(self):
        df = pandas.DataFrame([dict(a=1, b="e", c=1),
                               dict(a=2, b="f", c=1),
                               dict(a=4, b="g", c=1),
                               dict(a=8, b="h", c=1),
                               dict(a=12, b="h", c=1),
                               dict(a=16, b="i", c=1)])

        train, test = train_test_split_weights(df, train_size=0.5, weights='c')
        self.assertTrue(train is not None)
        self.assertTrue(test is not None)
        self.assertRaise(lambda: train_test_split_weights(
            df, test_size=0.5, weights=[0.5, 0.5]), ValueError, 'Dimension')
        self.assertRaise(lambda: train_test_split_weights(
            df, test_size=0), ValueError, 'null')
        self.assertRaise(lambda: train_test_split_weights(
            df, test_size=0, weights='c'), ValueError, 'null')

    def test_split_weights(self):
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

        df['connex'] = 'ole'
        train, test = train_test_connex_split(df, test_size=0.5,
                                              groups=['user', 'prod', 'card'],
                                              fail_imbalanced=0.4, fLOG=fLOG)
        self.assertEqual(train.shape[0] + test.shape[0], df.shape[0])

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

        train_test_connex_split(df, test_size=0.5, groups=['user', 'prod', 'card'],
                                fail_imbalanced=0.5, fLOG=fLOG, return_cnx=True)
        train, test, stats = train_test_connex_split(df, test_size=0.5,
                                                     groups=[
                                                         'user', 'prod', 'card'],
                                                     fail_imbalanced=0.5, fLOG=fLOG,
                                                     return_cnx=True, random_state=0)

        self.assertEqual(train.shape[0] + test.shape[0], df.shape[0])
        for col in ['user', 'prod', 'card']:
            s1 = set(train[col])
            s2 = set(test[col])
            if s1 & s2:
                rows = []
                for k, v in sorted(stats[0].items()):
                    rows.append("{0}={1}".format(k, v))
                raise Exception(
                    'Non empty intersection {0} & {1}\n{2}\n{3}\n{4}'.format(s1, s2, train, test, "\n".join(rows)))

    def test_split_connex_missing(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        df = pandas.DataFrame([dict(user="UA", prod="PAA", card="C1"),
                               dict(user="UA", prod="PB", card="C1"),
                               dict(user="UB", prod="PC", card="C2"),
                               dict(user="UB", prod="PD", card="C2"),
                               dict(user="UC", prod="PAA", card="C3"),
                               dict(user="UC", card="C4"),
                               dict(user="UD", prod="PG"),
                               ])

        train, test, stats = train_test_connex_split(df, test_size=0.5,
                                                     groups=[
                                                         'user', 'prod', 'card'],
                                                     fail_imbalanced=0.4, fLOG=fLOG,
                                                     return_cnx=True, random_state=0)

        self.assertEqual(train.shape[0] + test.shape[0], df.shape[0])
        for col in ['user', 'prod', 'card']:
            s1 = set(train[col])
            s2 = set(test[col])
            if s1 & s2:
                rows = []
                for k, v in sorted(stats[0].items()):
                    rows.append("{0}={1}".format(k, v))
                raise Exception(
                    'Non empty intersection {0} & {1}\n{2}\n{3}\n{4}'.format(s1, s2, train, test, "\n".join(rows)))


if __name__ == "__main__":
    unittest.main()
