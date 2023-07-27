# -*- coding: utf-8 -*-
import os
import unittest
from collections import Counter
import pandas
from pandas_streaming.ext_test_case import ExtTestCase
from pandas_streaming.df import train_test_connex_split


class TestConnexSplitBig(ExtTestCase):
    def test_connex_big(self):
        data = os.path.join(os.path.dirname(__file__), "data")
        name = os.path.join(data, "buggy_hash.csv")
        df = pandas.read_csv(name, sep="\t", encoding="utf-8")
        train, test, stats = train_test_connex_split(
            df,
            groups=["cart_id", "mail", "product_id"],
            fail_imbalanced=0.9,
            return_cnx=True,
        )
        self.assertGreater(train.shape[0], 0)
        self.assertGreater(test.shape[0], 0)
        elements = stats[1]["connex"]
        counts = Counter(elements)
        nbc = len(counts)
        maxi = max(counts.values())
        self.assertEqual(nbc, 5376)
        self.assertEqual(maxi, 14181)

    def test_connex_big_approx(self):
        data = os.path.join(os.path.dirname(__file__), "data")
        name = os.path.join(data, "buggy_hash.csv")
        df = pandas.read_csv(name, sep="\t", encoding="utf-8")
        train, test, stats = train_test_connex_split(
            df,
            groups=["cart_id", "mail", "product_id"],
            stop_if_bigger=0.05,
            return_cnx=True,
            keep_balance=0.8,
        )
        self.assertGreater(train.shape[0], 0)
        self.assertGreater(test.shape[0], 0)
        elements = stats[1]["connex"]
        counts = Counter(elements)
        nbc = len(counts)
        maxi = max(counts.values())
        self.assertGreater(nbc, 5376)
        self.assertLesser(maxi, 14181)

    def test_connex_big_approx_must(self):
        data = os.path.join(os.path.dirname(__file__), "data")
        name = os.path.join(data, "buggy_hash.csv")
        df = pandas.read_csv(name, sep="\t", encoding="utf-8")
        train, test, stats = train_test_connex_split(
            df,
            groups=["cart_id", "mail", "product_id"],
            stop_if_bigger=0.05,
            return_cnx=True,
            keep_balance=0.8,
            must_groups=["product_id"],
        )
        self.assertGreater(train.shape[0], 0)
        self.assertGreater(test.shape[0], 0)
        elements = stats[1]["connex"]
        counts = Counter(elements)
        nbc = len(counts)
        maxi = max(counts.values())
        self.assertGreater(nbc, 5376)
        self.assertLesser(maxi, 14181)
        train_ids = set(train.product_id)
        test_ids = set(test.product_id)
        inter = train_ids & test_ids
        self.assertEqual(len(inter), 0)


if __name__ == "__main__":
    unittest.main()
