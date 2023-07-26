import os
import tempfile
import unittest
from io import StringIO
import pandas
import numpy
from pandas_streaming.ext_test_case import ExtTestCase, ignore_warnings
from pandas_streaming.data import dummy_streaming_dataframe
from pandas_streaming.df import StreamingDataFrame
from pandas_streaming.df.dataframe import StreamingDataFrameSchemaError


class TestStreamingDataFrame(ExtTestCase):
    def test_shape(self):
        sdf = dummy_streaming_dataframe(100)
        dfs = list(sdf)
        self.assertEqual(len(dfs), 10)
        self.assertEqual(len(dfs), 10)
        shape = sdf.shape
        self.assertEqual(shape, (100, 2))

    def test_init(self):
        sdf = dummy_streaming_dataframe(100)
        df1 = sdf.to_df()
        sdf2 = StreamingDataFrame(sdf)
        df2 = sdf2.to_df()
        self.assertEqualDataFrame(df1, df2)

    def test_to_csv(self):
        sdf = dummy_streaming_dataframe(100)
        st = sdf.to_csv()
        self.assertStartsWith(",cint,cstr\n0,0,s0", st.replace("\r", ""))
        st = sdf.to_csv()
        self.assertStartsWith(",cint,cstr\n0,0,s0", st.replace("\r", ""))

    def test_iterrows(self):
        sdf = dummy_streaming_dataframe(100)
        rows = list(sdf.iterrows())
        self.assertEqual(sdf.shape[0], len(rows))
        rows = list(sdf.iterrows())
        self.assertEqual(sdf.shape[0], len(rows))

    def test_head(self):
        sdf = dummy_streaming_dataframe(100)
        st = sdf.head()
        self.assertEqual(st.shape, (5, 2))
        st = sdf.head(n=20)
        self.assertEqual(st.shape, (20, 2))
        st = sdf.head(n=20)
        self.assertEqual(st.shape, (20, 2))

    def test_tail(self):
        sdf = dummy_streaming_dataframe(100)
        st = sdf.tail()
        self.assertEqual(st.shape, (5, 2))
        st = sdf.tail(n=20)
        self.assertEqual(st.shape, (10, 2))

    def test_read_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            df = pandas.DataFrame(data=dict(a=[5, 6], b=["er", "r"]))
            name = os.path.join(temp, "df.csv")
            name2 = os.path.join(temp, "df2.csv")
            name3 = os.path.join(temp, "df3.csv")
            df.to_csv(name, index=False)
            df.to_csv(name2, index=True)
            sdf = StreamingDataFrame.read_csv(name)
            text = sdf.to_csv(index=False)
            self.assertRaise(
                lambda: StreamingDataFrame.read_csv(name2, index_col=0, chunksize=None),
                ValueError,
            )
            self.assertRaise(
                lambda: StreamingDataFrame.read_csv(name2, index_col=0, iterator=False),
                ValueError,
            )
            sdf2 = StreamingDataFrame.read_csv(name2, index_col=0)
            text2 = sdf2.to_csv(index=True)
            sdf2.to_csv(name3, index=True)
            with open(name, "r", encoding="utf-8") as f:
                exp = f.read()
            with open(name2, "r", encoding="utf-8") as f:
                exp2 = f.read()
            with open(name3, "r", encoding="utf-8") as f:
                text3 = f.read()
            self.assertEqual(text.replace("\r", ""), exp)
            sdf2 = StreamingDataFrame.read_df(df)
            self.assertEqualDataFrame(sdf.to_dataframe(), sdf2.to_dataframe())
            self.assertEqual(text2.replace("\r", ""), exp2)
            self.assertEqual(
                text3.replace("\r", "").replace("\n\n", "\n"), exp2.replace("\r", "")
            )

    def test_where(self):
        sdf = dummy_streaming_dataframe(100)
        cols = sdf.columns
        self.assertEqual(list(cols), ["cint", "cstr"])
        dts = sdf.dtypes
        self.assertEqual(len(dts), 2)
        res = sdf.where(lambda row: row["cint"] == 1)
        st = res.to_csv()
        self.assertStartsWith(",cint,cstr\n0,,\n1,1.0,s1", st.replace("\r", ""))
        res = sdf.where(lambda row: row["cint"] == 1)
        st = res.to_csv()
        self.assertStartsWith(",cint,cstr\n0,,\n1,1.0,s1", st.replace("\r", ""))

    def test_dataframe(self):
        sdf = dummy_streaming_dataframe(100)
        df = sdf.to_dataframe()
        self.assertEqual(df.shape, (100, 2))

    def test_sample(self):
        sdf = dummy_streaming_dataframe(100)
        res = sdf.sample(frac=0.1)
        self.assertLesser(res.shape[0], 30)
        self.assertRaise(lambda: sdf.sample(n=5), ValueError)
        res = sdf.sample(frac=0.1)
        self.assertLesser(res.shape[0], 30)
        self.assertRaise(lambda: sdf.sample(n=5), ValueError)

    def test_sample_cache(self):
        sdf = dummy_streaming_dataframe(100)
        res = sdf.sample(frac=0.1, cache=True)
        df1 = res.to_df()
        df2 = res.to_df()
        self.assertEqualDataFrame(df1, df2)
        self.assertTrue(res.is_stable(n=df1.shape[0], do_check=True))
        self.assertTrue(res.is_stable(n=df1.shape[0], do_check=False))
        res = sdf.sample(frac=0.1, cache=False)
        self.assertFalse(res.is_stable(n=df1.shape[0], do_check=False))

    def test_sample_reservoir_cache(self):
        sdf = dummy_streaming_dataframe(100)
        res = sdf.sample(n=10, cache=True, reservoir=True)
        df1 = res.to_df()
        df2 = res.to_df()
        self.assertEqualDataFrame(df1, df2)
        self.assertEqual(df1.shape, (10, res.shape[1]))
        self.assertRaise(
            lambda: sdf.sample(n=10, cache=False, reservoir=True), ValueError
        )
        self.assertRaise(
            lambda: sdf.sample(frac=0.1, cache=True, reservoir=True), ValueError
        )

    def test_apply(self):
        sdf = dummy_streaming_dataframe(100)
        self.assertNotEmpty(list(sdf))
        sdf = sdf.applymap(str)
        self.assertNotEmpty(list(sdf))
        sdf = sdf.apply(lambda row: row[["cint"]] + "r", axis=1)
        self.assertNotEmpty(list(sdf))
        text = sdf.to_csv(header=False)
        self.assertStartsWith("0,0r\n1,1r\n2,2r\n3,3r", text.replace("\r", ""))

    def test_train_test_split(self):
        sdf = dummy_streaming_dataframe(100)
        tr, te = sdf.train_test_split(index=False, streaming=False)
        self.assertRaise(
            lambda: StreamingDataFrame.read_str(tr, chunksize=None), ValueError
        )
        self.assertRaise(
            lambda: StreamingDataFrame.read_str(tr, iterator=False), ValueError
        )
        StreamingDataFrame.read_str(tr.encode("utf-8"))
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
        sdf = dummy_streaming_dataframe(100, asfloat=True)
        trsdf, tesdf = sdf.train_test_split(
            streaming=True, unique_rows=True, partitions=[0.7, 0.3]
        )
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
        df = pandas.DataFrame(data=dict(X=[4.5, 6, 7], Y=["a", "b", "c"]))

        sdf2 = StreamingDataFrame.read_df(pandas.concat([df, df]))
        sdftr, sdfte = sdf2.train_test_split(test_size=0.5)
        df1 = sdfte.head()
        df2 = sdfte.head()
        if df1 is not None or df2 is not None:
            self.assertEqualDataFrame(df1, df2)
        df1 = sdftr.head()
        df2 = sdftr.head()
        if df1 is not None or df2 is not None:
            self.assertEqualDataFrame(df1, df2)
        sdf = StreamingDataFrame.read_df(df)
        sdf2 = sdf.concat(sdf, axis=0)
        sdftr, sdfte = sdf2.train_test_split(test_size=0.5)
        df1 = sdfte.head()
        df2 = sdfte.head()
        if df1 is not None or df2 is not None:
            self.assertEqualDataFrame(df1, df2)
        df1 = sdftr.head()
        df2 = sdftr.head()
        if df1 is not None or df2 is not None:
            self.assertEqualDataFrame(df1, df2)

    def test_train_test_split_streaming_strat(self):
        sdf = dummy_streaming_dataframe(
            100, asfloat=True, tify=["t1" if i % 3 else "t0" for i in range(0, 100)]
        )
        trsdf, tesdf = sdf.train_test_split(
            streaming=True, unique_rows=True, stratify="tify"
        )
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
        self.assertGreater(gr["cfloat"].min(), 4)

    def test_train_test_split_file(self):
        with tempfile.TemporaryDirectory() as temp:
            names = [os.path.join(temp, "train.txt"), os.path.join(temp, "test.txt")]
            sdf = dummy_streaming_dataframe(100)
            sdf.train_test_split(names, index=False, streaming=False)
            trsdf = StreamingDataFrame.read_csv(names[0])
            tesdf = StreamingDataFrame.read_csv(names[1])
            self.assertGreater(trsdf.shape[0], 20)
            self.assertGreater(tesdf.shape[0], 20)
            trdf = trsdf.to_dataframe()
            tedf = tesdf.to_dataframe()
            self.assertGreater(trdf.shape[0], 20)
            self.assertGreater(tedf.shape[0], 20)
            df_exp = sdf.to_dataframe()
            df_val = pandas.concat([trdf, tedf])
            self.assertEqual(df_exp.shape, df_val.shape)
            df_val = df_val.sort_values("cint").reset_index(drop=True)
            self.assertEqualDataFrame(df_val, df_exp)

    def test_train_test_split_file_pattern(self):
        with tempfile.TemporaryDirectory() as temp:
            sdf = dummy_streaming_dataframe(100)
            names = os.path.join(temp, "spl_{0}.txt")
            self.assertRaise(
                lambda: sdf.train_test_split(names, index=False, streaming=False),
                ValueError,
            )
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
        def compares(a, b, how):
            m = a.merge(b, on="cint", indicator=True)
            dm = m.to_dataframe()
            da = a.to_dataframe()
            db = b.to_dataframe()
            exp = da.merge(db, on="cint", indicator=True)
            self.assertEqualDataFrame(
                dm.reset_index(drop=True), exp.reset_index(drop=True)
            )

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

    def test_concatv(self):
        sdf20 = dummy_streaming_dataframe(20)
        sdf30 = dummy_streaming_dataframe(30)
        df20 = sdf20.to_dataframe()
        df30 = sdf30.to_dataframe()
        df = pandas.concat([df20, df30], axis=0)

        m1 = sdf20.concat(sdf30, axis=0)
        self.assertEqualDataFrame(m1.to_dataframe(), df)
        m1 = sdf20.concat(df30, axis=0)
        self.assertEqualDataFrame(m1.to_dataframe(), df)
        m1 = sdf20.concat(map(lambda x: x, [df30]), axis=0)
        self.assertEqualDataFrame(m1.to_dataframe(), df)
        m1 = sdf20.concat(map(lambda x: x, [df30]), axis=0)
        self.assertEqualDataFrame(m1.to_dataframe(), df)

        df30["g"] = 4
        self.assertRaise(
            lambda: sdf20.concat(df30).to_dataframe(),
            ValueError,
            "Frame others[0] do not have the same column names",
        )
        df20["cint"] = df20["cint"].astype(float)
        self.assertRaise(
            lambda: sdf20.concat(df20).to_dataframe(),
            ValueError,
            "Frame others[0] do not have the same column types",
        )

    def test_concath(self):
        sdf20 = dummy_streaming_dataframe(20)
        sdf30 = dummy_streaming_dataframe(20)
        df20 = sdf20.to_dataframe()
        df30 = sdf30.to_dataframe()
        df = pandas.concat([df20, df30], axis=1)

        m1 = sdf20.concat(sdf30, axis=1)
        self.assertEqualDataFrame(m1.to_dataframe(), df)
        sdf22 = dummy_streaming_dataframe(22)
        sdf25 = dummy_streaming_dataframe(25)
        self.assertRaise(
            lambda: sdf22.concat(sdf25, axis=1).to_dataframe(), RuntimeError
        )

    def test_groupby(self):
        df20 = dummy_streaming_dataframe(20).to_dataframe()
        df20["key"] = df20["cint"].apply(lambda i: i % 3 == 0)
        sdf20 = StreamingDataFrame.read_df(df20, chunksize=5)
        gr = sdf20.groupby("key", lambda gr: gr.sum())
        gr2 = df20.groupby("key").sum()
        self.assertEqualDataFrame(gr, gr2)
        self.assertRaise(
            lambda: sdf20.groupby("key", in_memory=False), NotImplementedError
        )

        # Do not replace lambda c:sum(c) by sum or...
        # pandas.core.base.SpecificationError: Function names
        # must be unique, found multiple named sum
        gr2 = (
            df20.drop("cstr", axis=1).groupby("key").agg([numpy.sum, lambda c: sum(c)])
        )
        gr = sdf20.drop("cstr", axis=1).groupby(
            "key", lambda gr: gr.agg([numpy.sum, lambda c: sum(c)])
        )
        self.assertEqualDataFrame(gr, gr2)

        gr = sdf20.groupby("key", lambda gr: gr.count())
        gr2 = df20.groupby("key").count()
        self.assertEqualDataFrame(gr, gr2)

        df = pandas.DataFrame(dict(A=[3, 4, 3], B=[5, 6, 7]))
        sdf = StreamingDataFrame.read_df(df)
        gr = sdf.groupby("A")
        gr2 = df.groupby("A").sum()
        self.assertEqualDataFrame(gr, gr2)

    def test_groupby_cum(self):
        df20 = dummy_streaming_dataframe(20).to_dataframe()
        df20["key"] = df20["cint"].apply(lambda i: i % 3 == 0)
        sdf20 = StreamingDataFrame.read_df(df20, chunksize=5)
        sgr = sdf20.groupby_streaming(
            "key", lambda gr: gr.sum(), strategy="cum", as_index=False
        )
        gr2 = df20.groupby("key", as_index=False).sum()
        lastgr = None
        for gr in sgr:
            self.assertEqual(list(gr.columns), list(gr2.columns))
            lastgr = gr
        self.assertEqualDataFrame(lastgr, gr2)

    def test_groupby_streaming(self):
        df20 = dummy_streaming_dataframe(20).to_dataframe()
        df20["key"] = df20["cint"].apply(lambda i: i % 3 == 0)
        sdf20 = StreamingDataFrame.read_df(df20, chunksize=5)
        sgr = sdf20.groupby_streaming(
            "key", lambda gr: gr.sum(), strategy="streaming", as_index=False
        )
        gr2 = df20.groupby("key", as_index=False).sum()
        grs = list(sgr)
        gr = pandas.concat(grs).groupby("key", as_index=False).sum()
        self.assertEqualDataFrame(gr, gr2)

    def test_groupby_cum_asindex(self):
        df20 = dummy_streaming_dataframe(20).to_dataframe()
        df20["key"] = df20["cint"].apply(lambda i: i % 3 == 0)
        sdf20 = StreamingDataFrame.read_df(df20, chunksize=5)
        sgr = sdf20.groupby_streaming(
            "key", lambda gr: gr.sum(), strategy="cum", as_index=True
        )
        gr2 = df20.groupby("key", as_index=True).sum()
        lastgr = None
        for gr in sgr:
            self.assertEqual(list(gr.columns), list(gr2.columns))
            lastgr = gr
        self.assertEqualDataFrame(lastgr, gr2)

    def test_merge_2(self):
        df = pandas.DataFrame(data=dict(X=[4.5, 6, 7], Y=["a", "b", "c"]))
        df2 = pandas.concat([df, df])
        sdf = StreamingDataFrame.read_df(df)
        sdf2 = sdf.concat(sdf, axis=0)
        self.assertEqualDataFrame(df2, sdf2.to_dataframe())
        self.assertEqualDataFrame(df2, sdf2.to_dataframe())
        m = pandas.DataFrame(dict(Y=["a", "b"], Z=[10, 20]))
        jm = df2.merge(m, left_on="Y", right_on="Y", how="outer")
        sjm = sdf2.merge(m, left_on="Y", right_on="Y", how="outer")
        self.assertEqualDataFrame(
            jm.sort_values(["X", "Y"]).reset_index(drop=True),
            sjm.to_dataframe().sort_values(["X", "Y"]).reset_index(drop=True),
        )

    @ignore_warnings(ResourceWarning)
    def test_schema_consistent(self):
        df = pandas.DataFrame(
            [
                dict(cf=0, cint=0, cstr="0"),
                dict(cf=1, cint=1, cstr="1"),
                dict(cf=2, cint="s2", cstr="2"),
                dict(cf=3, cint=3, cstr="3"),
            ]
        )
        with tempfile.TemporaryDirectory() as temp:
            name = os.path.join(temp, "df.csv")
            stio = StringIO()
            df.to_csv(stio, index=False)
            self.assertNotEmpty(stio.getvalue())
            df.to_csv(name, index=False)
            self.assertEqual(df.shape, (4, 3))
            sdf = StreamingDataFrame.read_csv(name, chunksize=2)
            self.assertRaise(lambda: list(sdf), StreamingDataFrameSchemaError)
            sdf = StreamingDataFrame.read_csv(name, chunksize=2, check_schema=False)
            pieces = list(sdf)
            self.assertEqual(len(pieces), 2)

    def test_getitem(self):
        sdf = dummy_streaming_dataframe(100)
        sdf2 = sdf[["cint"]]
        self.assertEqual(sdf2.shape, (100, 1))
        df1 = sdf.to_df()
        df2 = sdf2.to_df()
        self.assertEqualDataFrame(df1[["cint"]], df2)
        self.assertRaise(lambda: sdf[:, "cint"], NotImplementedError)

    @ignore_warnings(ResourceWarning)
    def test_read_csv_names(self):
        this = os.path.abspath(os.path.dirname(__file__))
        data = os.path.join(this, "data", "buggy_hash2.csv")
        df = pandas.read_csv(data, sep="\t", names=["A", "B", "C"], header=None)
        sdf = StreamingDataFrame.read_csv(
            data, sep="\t", names=["A", "B", "C"], chunksize=2, header=None
        )
        head = sdf.head(n=1)
        self.assertEqualDataFrame(df.head(n=1), head)

    def test_add_column(self):
        df = pandas.DataFrame(data=dict(X=[4.5, 6, 7], Y=["a", "b", "c"]))
        sdf = StreamingDataFrame.read_df(df)
        sdf2 = sdf.add_column("d", lambda row: int(1))
        df2 = sdf2.to_dataframe()
        df["d"] = 1
        self.assertEqualDataFrame(df, df2)

        sdf3 = StreamingDataFrame.read_df(df)
        sdf4 = sdf3.add_column("dd", 2)
        df4 = sdf4.to_dataframe()
        df["dd"] = 2
        self.assertEqualDataFrame(df, df4)

        sdfA = StreamingDataFrame.read_df(df)
        sdfB = sdfA.add_column("dd12", lambda row: row["dd"] + 10)
        dfB = sdfB.to_dataframe()
        df["dd12"] = 12
        self.assertEqualDataFrame(df, dfB)

    def test_fillna(self):
        df = pandas.DataFrame(data=dict(X=[4.5, numpy.nan, 7], Y=["a", "b", numpy.nan]))
        sdf = StreamingDataFrame.read_df(df)

        df2 = pandas.DataFrame(data=dict(X=[4.5, 10.0, 7], Y=["a", "b", "NAN"]))
        na = sdf.fillna(value=dict(X=10.0, Y="NAN"))
        ndf = na.to_df()
        self.assertEqual(ndf, df2)

        df3 = pandas.DataFrame(data=dict(X=[4.5, 10.0, 7], Y=["a", "b", numpy.nan]))
        na = sdf.fillna(value=dict(X=10.0))
        ndf = na.to_df()
        self.assertEqual(ndf, df3)

    def test_describe(self):
        x = numpy.arange(100001).astype(numpy.float64) / 100000 - 0.5
        y = numpy.arange(100001).astype(numpy.int64)
        z = numpy.array([chr(65 + j % 45) for j in y])
        df = pandas.DataFrame(data=dict(X=x, Y=y, Z=z))
        sdf = StreamingDataFrame.read_df(df)

        desc = sdf.describe()
        self.assertEqual(["X", "Y"], list(desc.columns))
        self.assertEqual(desc.loc["min", :].tolist(), [-0.5, 0])
        self.assertEqual(desc.loc["max", :].tolist(), [0.5, 100000])
        self.assertEqualArray(desc.loc["mean", :], numpy.array([0, 50000]), atol=1e-8)
        self.assertEqualArray(desc.loc["25%", :], numpy.array([-0.25, 25000]))
        self.assertEqualArray(desc.loc["50%", :], numpy.array([0.0, 50000]))
        self.assertEqualArray(desc.loc["75%", :], numpy.array([0.25, 75000]))
        self.assertEqualArray(
            desc.loc["std", :], numpy.array([2.886795e-01, 28867.946472]), decimal=4
        )

    def test_set_item(self):
        df = pandas.DataFrame(data=dict(a=[4.5], b=[6], c=[7]))
        self.assertRaise(lambda: StreamingDataFrame(df), TypeError)
        sdf = StreamingDataFrame.read_df(df)

        def f():
            sdf[["a"]] = 10

        self.assertRaise(f, ValueError)

        def g():
            sdf["a"] = [10]

        self.assertRaise(g, NotImplementedError)

        sdf["aa"] = 10
        df = sdf.to_df()
        ddf = pandas.DataFrame(data=dict(a=[4.5], b=[6], c=[7], aa=[10]))
        self.assertEqualDataFrame(df, ddf)
        sdf["bb"] = sdf["b"] + 10
        df = sdf.to_df()
        ddf = ddf = pandas.DataFrame(data=dict(a=[4.5], b=[6], c=[7], aa=[10], bb=[16]))
        self.assertEqualDataFrame(df, ddf)

    def test_set_item_function(self):
        df = pandas.DataFrame(data=dict(a=[4.5], b=[6], c=[7]))
        self.assertRaise(lambda: StreamingDataFrame(df), TypeError)
        sdf = StreamingDataFrame.read_df(df)
        sdf["bb"] = sdf["b"].apply(lambda x: x + 11)
        df = sdf.to_df()
        ddf = ddf = pandas.DataFrame(data=dict(a=[4.5], b=[6], c=[7], bb=[17]))
        self.assertEqualDataFrame(df, ddf)


if __name__ == "__main__":
    unittest.main(verbosity=2)
