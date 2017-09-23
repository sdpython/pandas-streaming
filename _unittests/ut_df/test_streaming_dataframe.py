#-*- coding: utf-8 -*-
"""
@brief      test log(time=3s)
"""

import sys
import os
import unittest


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
from src.pandas_streaming.data import dummy_streaming_dataframe
from src.pandas_streaming.exc import StreamingInefficientException


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
        

if __name__ == "__main__":
    unittest.main()
