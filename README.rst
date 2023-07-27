pandas-streaming: streaming API over pandas
===========================================

.. image:: https://ci.appveyor.com/api/projects/status/4te066r8ne1ymmhy?svg=true
    :target: https://ci.appveyor.com/project/sdpython/pandas-streaming
    :alt: Build Status Windows

.. image:: https://dl.circleci.com/status-badge/img/gh/sdpython/pandas-streaming/tree/main.svg?style=svg
    :target: https://dl.circleci.com/status-badge/redirect/gh/sdpython/pandas-streaming/tree/main

.. image:: https://dev.azure.com/xavierdupre3/pandas_streaming/_apis/build/status/sdpython.pandas_streaming
    :target: https://dev.azure.com/xavierdupre3/pandas_streaming/

.. image:: https://badge.fury.io/py/pandas_streaming.svg
    :target: http://badge.fury.io/py/pandas_streaming

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: https://opensource.org/license/MIT/

.. image:: https://codecov.io/gh/sdpython/pandas-streaming/branch/main/graph/badge.svg?token=0caHX1rhr8 
    :target: https://codecov.io/gh/sdpython/pandas-streaming

.. image:: http://img.shields.io/github/issues/sdpython/pandas_streaming.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/pandas_streaming/issues

.. image:: https://pepy.tech/badge/pandas_streaming/month
    :target: https://pepy.tech/project/pandas_streaming/month
    :alt: Downloads

.. image:: https://img.shields.io/github/forks/sdpython/pandas_streaming.svg
    :target: https://github.com/sdpython/pandas_streaming/
    :alt: Forks

.. image:: https://img.shields.io/github/stars/sdpython/pandas_streaming.svg
    :target: https://github.com/sdpython/pandas_streaming/
    :alt: Stars

.. image:: https://img.shields.io/github/repo-size/sdpython/pandas_streaming
    :target: https://github.com/sdpython/pandas_streaming/
    :alt: size

**pandas_streaming**
aims at processing big files with `pandas <http://pandas.pydata.org/>`_,
too big to hold in memory, too small to be parallelized with a significant gain.
The module replicates a subset of `pandas <http://pandas.pydata.org/>`_ API
and implements other functionalities for machine learning.

::

    from pandas_streaming.df import StreamingDataFrame
    sdf = StreamingDataFrame.read_csv("filename", sep="\t", encoding="utf-8")

    for df in sdf:
        # process this chunk of data
        # df is a dataframe
        print(df)

The module can also stream an existing dataframe.

::

    import pandas
    df = pandas.DataFrame([dict(cf=0, cint=0, cstr="0"),
                           dict(cf=1, cint=1, cstr="1"),
                           dict(cf=3, cint=3, cstr="3")])

    from pandas_streaming.df import StreamingDataFrame
    sdf = StreamingDataFrame.read_df(df)

    for df in sdf:
        # process this chunk of data
        # df is a dataframe
        print(df)

It contains other helpers to split datasets into
train and test with some weird constraints.

* `documentation <https://sdpython.github.io/doc/pandas-streaming/dev/>`_
