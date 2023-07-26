import os
import unittest
import io
import zipfile
import numpy
import pandas
from pandas_streaming.ext_test_case import ExtTestCase
from pandas_streaming.df import to_zip, read_zip


class TestDataFrameIO(ExtTestCase):
    def test_zip_dataframe(self):
        df = pandas.DataFrame(
            [
                dict(a=1, b="eé", c=5.6, ind="a1", ai=1),
                dict(b="f", c=5.7, ind="a2", ai=2),
                dict(a=4, b="g", ind="a3", ai=3),
                dict(a=8, b="h", c=5.9, ai=4),
                dict(a=16, b="i", c=6.2, ind="a5", ai=5),
            ]
        )

        temp = get_temp_folder(__file__, "temp_zip")
        name = os.path.join(temp, "df.zip")
        to_zip(df, name, encoding="utf-8", index=False)
        df2 = read_zip(name, encoding="utf-8")
        self.assertEqualDataFrame(df, df2)

        st = io.BytesIO()
        zp = zipfile.ZipFile(st, "w")
        to_zip(df, zp, encoding="utf-8", index=False)
        zp.close()

        st = io.BytesIO(st.getvalue())
        zp = zipfile.ZipFile(st, "r")
        df3 = read_zip(zp, encoding="utf-8")
        zp.close()
        self.assertEqualDataFrame(df, df3)

    def test_zip_numpy(self):
        df = numpy.zeros((3, 4))
        df[2, 3] = 1

        temp = get_temp_folder(__file__, "temp_zip")
        name = os.path.join(temp, "df.zip")
        to_zip(df, name, "arr.npy")
        df2 = read_zip(name, "arr.npy")
        self.assertEqualArray(df, df2)

        st = io.BytesIO()
        zp = zipfile.ZipFile(st, "w")
        to_zip(df, zp, "arr.npy")
        zp.close()

        st = io.BytesIO(st.getvalue())
        zp = zipfile.ZipFile(st, "r")
        df3 = read_zip(zp, "arr.npy")
        zp.close()
        self.assertEqualArray(df, df3)


if __name__ == "__main__":
    unittest.main()
