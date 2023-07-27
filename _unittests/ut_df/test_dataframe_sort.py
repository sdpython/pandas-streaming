import os
import tempfile
import unittest
import pandas
from pandas_streaming.ext_test_case import ExtTestCase
from pandas_streaming.df import StreamingDataFrame


class TestDataFrameSort(ExtTestCase):
    def test_sort_values(self):
        with tempfile.TemporaryDirectory() as temp:
            name = os.path.join(temp, "_data_")
            df = pandas.DataFrame(
                [
                    dict(a=1, b="eé", c=5.6, ind="a1", ai=1),
                    dict(a=5, b="f", c=5.7, ind="a2", ai=2),
                    dict(a=4, b="g", ind="a3", ai=3),
                    dict(a=8, b="h", c=5.9, ai=4),
                    dict(a=16, b="i", c=6.2, ind="a5", ai=5),
                ]
            )
            sdf = StreamingDataFrame.read_df(df, chunksize=2)
            sorted_df = df.sort_values(by="a")
            res = sdf.sort_values(by="a", temp_file=name)
            res_df = res.to_df()
            self.assertEqualDataFrame(sorted_df, res_df)

    def test_sort_values_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            name = os.path.join(temp, "_data_")
            df = pandas.DataFrame(
                [
                    dict(a=1, b="eé", c=5.6, ind="a1", ai=1),
                    dict(a=5, b="f", c=5.7, ind="a2", ai=2),
                    dict(a=4, b="g", ind="a3", ai=3),
                    dict(a=8, b="h", c=5.9, ai=4),
                    dict(a=16, b="i", c=6.2, ind="a5", ai=5),
                ]
            )
            sdf = StreamingDataFrame.read_df(df, chunksize=2)
            sorted_df = df.sort_values(by="a")
            res = sdf.sort_values(by="a", temp_file=name)
            res_df = res.to_df()
            self.assertEqualDataFrame(sorted_df, res_df)
            res_df = res.to_df()
            self.assertEqualDataFrame(sorted_df, res_df)

    def test_sort_values_reverse(self):
        with tempfile.TemporaryDirectory() as temp:
            name = os.path.join(temp, "_data_")
            df = pandas.DataFrame(
                [
                    dict(a=1, b="eé", c=5.6, ind="a1", ai=1),
                    dict(a=5, b="f", c=5.7, ind="a2", ai=2),
                    dict(a=4, b="g", ind="a3", ai=3),
                    dict(a=8, b="h", c=5.9, ai=4),
                    dict(a=16, b="i", c=6.2, ind="a5", ai=5),
                ]
            )
            sdf = StreamingDataFrame.read_df(df, chunksize=2)
            sorted_df = df.sort_values(by="a", ascending=False)
            res = sdf.sort_values(by="a", temp_file=name, ascending=False)
            res_df = res.to_df()
            self.assertEqualDataFrame(sorted_df, res_df)

    def test_sort_values_nan_last(self):
        with tempfile.TemporaryDirectory() as temp:
            name = os.path.join(temp, "_data_")
            df = pandas.DataFrame(
                [
                    dict(a=1, b="eé", c=5.6, ind="a1", ai=1),
                    dict(b="f", c=5.7, ind="a2", ai=2),
                    dict(b="f", c=5.8, ind="a2", ai=2),
                    dict(a=4, b="g", ind="a3", ai=3),
                    dict(a=8, b="h", c=5.9, ai=4),
                    dict(a=16, b="i", c=6.2, ind="a5", ai=5),
                ]
            )
            sdf = StreamingDataFrame.read_df(df, chunksize=2)
            sorted_df = df.sort_values(by="a", na_position="last")
            res = sdf.sort_values(by="a", temp_file=name, na_position="last")
            res_df = res.to_df()
            self.assertEqualDataFrame(sorted_df, res_df)

    def test_sort_values_nan_first(self):
        with tempfile.TemporaryDirectory() as temp:
            name = os.path.join(temp, "_data_")
            df = pandas.DataFrame(
                [
                    dict(a=1, b="eé", c=5.6, ind="a1", ai=1),
                    dict(b="f", c=5.7, ind="a2", ai=2),
                    dict(b="f", c=5.8, ind="a2", ai=2),
                    dict(a=4, b="g", ind="a3", ai=3),
                    dict(a=8, b="h", c=5.9, ai=4),
                    dict(a=16, b="i", c=6.2, ind="a5", ai=5),
                ]
            )
            sdf = StreamingDataFrame.read_df(df, chunksize=2)
            sorted_df = df.sort_values(by="a", na_position="first")
            res = sdf.sort_values(by="a", temp_file=name, na_position="first")
            res_df = res.to_df()
            self.assertEqualDataFrame(sorted_df, res_df)


if __name__ == "__main__":
    unittest.main()
