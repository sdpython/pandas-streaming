#-*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
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
from pyquickhelper.pycode import ExtTestCase, get_temp_folder
from src.pandas_streaming.data import dummy_streaming_dataframe
from src.pandas_streaming.exc import StreamingInefficientException
from src.pandas_streaming.df import StreamingDataFrame


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

    def test_to_csv(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        sdf = dummy_streaming_dataframe(100)
        st = sdf.to_csv()
        self.assertStartsWith(",cint,cstr\n0,0,s0", st)

    def test_iterrows(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        sdf = dummy_streaming_dataframe(100)
        rows = list(sdf.iterrows())
        self.assertEqual(sdf.shape[0], len(rows))

    def test_head(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        sdf = dummy_streaming_dataframe(100)
        st = sdf.head()
        self.assertEqual(st.shape, (5, 2))
        st = sdf.head(n=20)
        self.assertEqual(st.shape, (20, 2))

    def test_tail(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        sdf = dummy_streaming_dataframe(100)
        st = sdf.tail()
        self.assertEqual(st.shape, (5, 2))
        st = sdf.tail(n=20)
        self.assertEqual(st.shape, (10, 2))

    def test_read_csv(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        temp = get_temp_folder(__file__, "temp_read_dsv")
        df = pandas.DataFrame(data=dict(a=[5, 6], b=["er", "r"]))
        name = os.path.join(temp, "df.csv")
        name2 = os.path.join(temp, "df2.csv")
        name3 = os.path.join(temp, "df3.csv")
        df.to_csv(name, index=False)
        df.to_csv(name2, index=True)
        sdf = StreamingDataFrame.read_csv(name)
        text = sdf.to_csv(index=False)
        sdf2 = StreamingDataFrame.read_csv(name2, index_col=0)
        text2 = sdf2.to_csv(index=True)
        sdf2.to_csv(name3, index=True)
        with open(name, "r") as f:
            exp = f.read()
        with open(name2, "r") as f:
            exp2 = f.read()
        with open(name3, "r") as f:
            text3 = f.read()
        self.assertEqual(text, exp)
        sdf2 = StreamingDataFrame.read_df(df)
        self.assertEqualDataFrame(sdf.to_dataframe(), sdf2.to_dataframe())
        self.assertEqual(text2, exp2)
        self.assertEqual(text3, exp2)

    def test_where(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        sdf = dummy_streaming_dataframe(100)
        cols = sdf.columns
        self.assertEqual(list(cols), ['cint', 'cstr'])
        dts = sdf.dtypes
        self.assertEqual(len(dts), 2)
        res = sdf.where(lambda row: row["cint"] == 1)
        st = res.to_csv()
        self.assertStartsWith(",cint,cstr\n0,,\n1,1.0,s1", st)

    def test_dataframe(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        sdf = dummy_streaming_dataframe(100)
        df = sdf.to_dataframe()
        self.assertEqual(df.shape, (100, 2))

    def test_sample(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        sdf = dummy_streaming_dataframe(100)
        res = sdf.sample(frac=0.1)
        self.assertLesser(res.shape[0], 30)
        self.assertRaise(lambda: sdf.sample(n=5), ValueError)

    def test_apply(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        sdf = dummy_streaming_dataframe(100)
        self.assertNotEmpty(list(sdf))
        sdf = sdf.applymap(str)
        self.assertNotEmpty(list(sdf))
        sdf = sdf.apply(lambda row: row["cint"] + "r", axis=1)
        self.assertNotEmpty(list(sdf))
        text = sdf.to_csv()
        self.assertStartsWith("0,0r\n1,1r\n2,2r\n3,3r", text)

    def test_train_test_split(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        sdf = dummy_streaming_dataframe(100)
        tr, te = sdf.train_test_split(index=False)
        trsdf = StreamingDataFrame.read_str(tr)
        tesdf = StreamingDataFrame.read_str(te)
        trdf = trsdf.to_dataframe()
        tedf = tesdf.to_dataframe()
        df_exp = sdf.to_dataframe()
        df_val = pandas.concat([trdf, tedf])
        self.assertEqual(df_exp.shape, df_val.shape)
        df_val = df_val.sort_values("cint").reset_index(drop=True)
        self.assertEqualDataFrame(df_val, df_exp)

    def test_train_test_split_file(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        temp = get_temp_folder(__file__, "temp_train_test_split_file")
        names = [os.path.join(temp, "train.txt"),
                 os.path.join(temp, "test.txt")]
        sdf = dummy_streaming_dataframe(100)
        tr, te = sdf.train_test_split(names, index=False)
        trsdf = StreamingDataFrame.read_csv(names[0])
        tesdf = StreamingDataFrame.read_csv(names[1])
        trdf = trsdf.to_dataframe()
        tedf = tesdf.to_dataframe()
        df_exp = sdf.to_dataframe()
        df_val = pandas.concat([trdf, tedf])
        self.assertEqual(df_exp.shape, df_val.shape)
        df_val = df_val.sort_values("cint").reset_index(drop=True)
        self.assertEqualDataFrame(df_val, df_exp)


if __name__ == "__main__":
    unittest.main()
