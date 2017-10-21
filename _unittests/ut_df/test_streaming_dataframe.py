#-*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
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
        tr, te = sdf.train_test_split(index=False, streaming=False)
        trsdf = StreamingDataFrame.read_str(tr)
        tesdf = StreamingDataFrame.read_str(te)
        trdf = trsdf.to_dataframe()
        tedf = tesdf.to_dataframe()
        df_exp = sdf.to_dataframe()
        df_val = pandas.concat([trdf, tedf])
        self.assertEqual(df_exp.shape, df_val.shape)
        df_val = df_val.sort_values("cint").reset_index(drop=True)
        self.assertEqualDataFrame(df_val, df_exp)

    def test_train_test_split_streaming(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        sdf = dummy_streaming_dataframe(100, asfloat=True)
        trsdf, tesdf = sdf.train_test_split(
            streaming=True, unique_rows=True, partitions=[0.7, 0.3])
        trdf = trsdf.to_dataframe()
        tedf = tesdf.to_dataframe()
        df_exp = sdf.to_dataframe()
        df_val = pandas.concat([trdf, tedf])
        self.assertEqual(df_exp.shape, df_val.shape)
        df_val = df_val.sort_values("cfloat").reset_index(drop=True)
        self.assertEqualDataFrame(df_val, df_exp)
        trdf2 = trsdf.to_dataframe()
        tedf2 = tesdf.to_dataframe()
        df_val = pandas.concat([trdf2, tedf2])
        self.assertEqual(df_exp.shape, df_val.shape)
        df_val = df_val.sort_values("cfloat").reset_index(drop=True)
        self.assertEqualDataFrame(df_val, df_exp)
        self.assertEqual(trdf.shape, trdf2.shape)
        self.assertEqual(tedf.shape, tedf2.shape)
        self.assertGreater(trdf.shape[0], tedf.shape[0])
        self.assertGreater(trdf2.shape[0], tedf2.shape[0])

    def test_train_test_split_streaming_tiny(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")
        df = pandas.DataFrame(data=dict(X=[4.5, 6, 7], Y=["a", "b", "c"]))

        sdf2 = StreamingDataFrame.read_df(pandas.concat([df, df]))
        sdftr, sdfte = sdf2.train_test_split(test_size=0.5)
        df1 = sdfte.head()
        df2 = sdfte.head()
        self.assertEqualDataFrame(df1, df2)
        df1 = sdftr.head()
        df2 = sdftr.head()
        self.assertEqualDataFrame(df1, df2)
        sdf = StreamingDataFrame.read_df(df)
        sdf2 = sdf.concat(sdf)
        sdftr, sdfte = sdf2.train_test_split(test_size=0.5)
        df1 = sdfte.head()
        df2 = sdfte.head()
        self.assertEqualDataFrame(df1, df2)
        df1 = sdftr.head()
        df2 = sdftr.head()
        self.assertEqualDataFrame(df1, df2)

    def test_train_test_split_streaming_strat(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        sdf = dummy_streaming_dataframe(100, asfloat=True,
                                        tify=["t1" if i % 3 else "t0" for i in range(0, 100)])
        trsdf, tesdf = sdf.train_test_split(
            streaming=True, unique_rows=True, stratify="tify")
        trdf = trsdf.to_dataframe()
        tedf = tesdf.to_dataframe()
        df_exp = sdf.to_dataframe()
        df_val = pandas.concat([trdf, tedf])
        self.assertEqual(df_exp.shape, df_val.shape)
        df_val = df_val.sort_values("cfloat").reset_index(drop=True)
        self.assertEqualDataFrame(df_val, df_exp)
        trdf = trsdf.to_dataframe()
        tedf = tesdf.to_dataframe()
        df_val = pandas.concat([trdf, tedf])
        self.assertEqual(df_exp.shape, df_val.shape)
        df_val = df_val.sort_values("cfloat").reset_index(drop=True)
        self.assertEqualDataFrame(df_val, df_exp)
        trgr = trdf.groupby("tify").count()
        trgr["part"] = 0
        tegr = tedf.groupby("tify").count()
        tegr["part"] = 1
        gr = pandas.concat([trgr, tegr])
        self.assertGreater(gr['cfloat'].min(), 4)

    def test_train_test_split_file(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        temp = get_temp_folder(__file__, "temp_train_test_split_file")
        names = [os.path.join(temp, "train.txt"),
                 os.path.join(temp, "test.txt")]
        sdf = dummy_streaming_dataframe(100)
        tr, te = sdf.train_test_split(names, index=False, streaming=False)
        trsdf = StreamingDataFrame.read_csv(names[0])
        tesdf = StreamingDataFrame.read_csv(names[1])
        trdf = trsdf.to_dataframe()
        tedf = tesdf.to_dataframe()
        df_exp = sdf.to_dataframe()
        df_val = pandas.concat([trdf, tedf])
        self.assertEqual(df_exp.shape, df_val.shape)
        df_val = df_val.sort_values("cint").reset_index(drop=True)
        self.assertEqualDataFrame(df_val, df_exp)

    def test_train_test_split_file_pattern(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        temp = get_temp_folder(__file__, "temp_train_test_split_file_pattern")
        sdf = dummy_streaming_dataframe(100)
        names = os.path.join(temp, "spl_{0}.txt")
        self.assertRaise(lambda: sdf.train_test_split(
            names, index=False, streaming=False), ValueError)
        names = os.path.join(temp, "spl_{}.txt")
        tr, te = sdf.train_test_split(names, index=False, streaming=False)
        trsdf = StreamingDataFrame.read_csv(tr)
        tesdf = StreamingDataFrame.read_csv(te)
        trdf = trsdf.to_dataframe()
        tedf = tesdf.to_dataframe()
        df_exp = sdf.to_dataframe()
        df_val = pandas.concat([trdf, tedf])
        self.assertEqual(df_exp.shape, df_val.shape)
        df_val = df_val.sort_values("cint").reset_index(drop=True)
        self.assertEqualDataFrame(df_val, df_exp)

    def test_merge(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        def compares(a, b, how):
            m = a.merge(b, on="cint", indicator=True)
            dm = m.to_dataframe()
            da = a.to_dataframe()
            db = b.to_dataframe()
            exp = da.merge(db, on="cint", indicator=True)
            self.assertEqualDataFrame(dm.reset_index(drop=True),
                                      exp.reset_index(drop=True))

        sdf20 = dummy_streaming_dataframe(20)
        sdf30 = dummy_streaming_dataframe(30)
        # itself
        hows = "inner left right outer".split()
        for how in hows:
            compares(sdf20, sdf20, how)
            compares(sdf20, sdf20, how)
        for how in hows:
            compares(sdf20, sdf30, how)
            compares(sdf20, sdf30, how)
        for how in hows:
            compares(sdf30, sdf20, how)
            compares(sdf30, sdf20, how)
        sdf20.merge(sdf20.to_dataframe(), on="cint", indicator=True)

    def test_concat(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        sdf20 = dummy_streaming_dataframe(20)
        sdf30 = dummy_streaming_dataframe(30)
        df20 = sdf20.to_dataframe()
        df30 = sdf30.to_dataframe()
        df = pandas.concat([df20, df30])

        m1 = sdf20.concat(sdf30)
        self.assertEqualDataFrame(m1.to_dataframe(), df)
        m1 = sdf20.concat(df30)
        self.assertEqualDataFrame(m1.to_dataframe(), df)
        m1 = sdf20.concat(map(lambda x: x, [df30]))
        self.assertEqualDataFrame(m1.to_dataframe(), df)
        m1 = sdf20.concat(map(lambda x: x, [df30]))
        self.assertEqualDataFrame(m1.to_dataframe(), df)

        df30["g"] = 4
        self.assertRaise(lambda: sdf20.concat(df30).to_dataframe(
        ), ValueError, "Frame others[0] do not have the same column names")
        df20["cint"] = df20["cint"].astype(float)
        self.assertRaise(lambda: sdf20.concat(df20).to_dataframe(
        ), ValueError, "Frame others[0] do not have the same column types")

    def test_groupby(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        df20 = dummy_streaming_dataframe(20).to_dataframe()
        df20["key"] = df20["cint"].apply(lambda i: i % 3 == 0)
        sdf20 = StreamingDataFrame.read_df(df20, chunk_size=5)
        gr = sdf20.groupby("key", lambda gr: gr.sum())
        gr2 = df20.groupby("key").sum()
        self.assertEqualDataFrame(gr, gr2)
        self.assertRaise(lambda: sdf20.groupby(
            "key", in_memory=False), NotImplementedError)

        gr2 = df20.groupby("key").agg([numpy.sum, lambda c:sum(c)])
        gr = sdf20.groupby("key", lambda gr: gr.agg(
            [numpy.sum, lambda c:sum(c)]))
        self.assertEqualDataFrame(gr, gr2)

        gr = sdf20.groupby("key", lambda gr: gr.count())
        gr2 = df20.groupby("key").count()
        self.assertEqualDataFrame(gr, gr2)

        df = pandas.DataFrame(dict(A=[3, 4, 3], B=[5, 6, 7]))
        sdf = StreamingDataFrame.read_df(df)
        gr = sdf.groupby("A")
        gr2 = df.groupby("A").sum()
        self.assertEqualDataFrame(gr, gr2)

    def test_merge_2(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        df = pandas.DataFrame(data=dict(X=[4.5, 6, 7], Y=["a", "b", "c"]))
        df2 = pandas.concat([df, df])
        sdf = StreamingDataFrame.read_df(df)
        sdf2 = sdf.concat(sdf)
        self.assertEqualDataFrame(df2, sdf2.to_dataframe())
        self.assertEqualDataFrame(df2, sdf2.to_dataframe())
        m = pandas.DataFrame(dict(Y=["a", "b"], Z=[10, 20]))
        jm = df2.merge(m, left_on="Y", right_on="Y", how="outer")
        sjm = sdf2.merge(m, left_on="Y", right_on="Y", how="outer")
        self.assertEqualDataFrame(jm.sort_values(["X", "Y"]).reset_index(drop=True),
                                  sjm.to_dataframe().sort_values(["X", "Y"]).reset_index(drop=True))


if __name__ == "__main__":
    unittest.main()
    # TestStreamingDataFrame().test_train_test_split_streaming_tiny()
