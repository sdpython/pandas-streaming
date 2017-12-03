#-*- coding: utf-8 -*-
"""
@file
@brief Saves a dataframe into a zip files.
"""
import io
import os
import pandas
import zipfile


def to_zip(df, zipfilename, zname="df.csv", **kwargs):
    """
    Saves a dataframe into a zip file.
    It can be read by @see fn to_zip.

    @param      df          dataframe
    @param      zipfilename a :epkg:`*py:zipfile:ZipFile` or a filename
    @param      zname       a filename in th zipfile
    @param      kwargs      parameters for :epkg:`pandas:to_csv`
    @return                 zipfilename

    .. exref::
        :title: Saves and read a dataframe in a zip file

        This shows an example on how to save and read a
        dataframe directly into a zip file.

        .. runpython::
            :showcode:

            import pandas
            from pandas_streaming.df import to_zip, read_zip

            df = pandas.DataFrame([dict(a=1, b="e"),
                                   dict(b="f", a=5.7)])

            to_zip(df, name, encoding="utf-8", index=False)
            df2 = read_zip(name, encoding="utf-8")
            print(df2)
    """
    stb = io.StringIO()
    df.to_csv(stb, **kwargs)
    text = stb.getvalue()

    if isinstance(zipfilename, str):
        ext = os.path.splitext(zipfilename)[-1]
        if ext != '.zip':
            raise NotImplementedError(
                "Only zip file are implemented not '{0}'.".format(ext))
        zf = zipfile.ZipFile(zipfilename, 'w')
        close = True
    elif isinstance(zipfilename, zipfile.ZipFile):
        zf = zipfilename
        close = False
    else:
        raise TypeError(
            "No implementation for type '{0}'".format(type(zipfilename)))

    zf.writestr(zname, text)
    if close:
        zf.close()


def read_zip(zipfilename, zname="df.csv", **kwargs):
    """
    Reads a dataframe from a zip file.
    It can be saved by @see fn read_zip.

    @param      zipfilename a :epkg:`*py:zipfile:ZipFile` or a filename
    @param      zname       a filename in th zipfile
    @param      kwargs      parameters for :epkg:`pandas:read_csv`
    @return                 dataframe
    """
    if isinstance(zipfilename, str):
        ext = os.path.splitext(zipfilename)[-1]
        if ext != '.zip':
            raise NotImplementedError(
                "Only zip file are implemented not '{0}'.".format(ext))
        zf = zipfile.ZipFile(zipfilename, 'r')
        close = True
    elif isinstance(zipfilename, zipfile.ZipFile):
        zf = zipfilename
        close = False
    else:
        raise TypeError(
            "No implementation for type '{0}'".format(type(zipfilename)))

    content = zf.read(zname)
    stb = io.BytesIO(content)
    df = pandas.read_csv(stb, **kwargs)

    if close:
        zf.close()

    return df
