pandas-streaming: streaming API over pandas
===========================================

.. image:: https://github.com/sdpython/pandas_streaming/blob/main/_doc/_static/project_ico.png?raw=true
    :target: https://github.com/sdpython/pandas_streaming/

.. image:: https://travis-ci.com/sdpython/pandas_streaming.svg?branch=main
    :target: https://app.travis-ci.com/github/sdpython/pandas_streaming
    :alt: Build status

.. image:: https://ci.appveyor.com/api/projects/status/4te066r8ne1ymmhy?svg=true
    :target: https://ci.appveyor.com/project/sdpython/pandas-streaming
    :alt: Build Status Windows

.. image:: https://circleci.com/gh/sdpython/pandas_streaming/tree/main.svg?style=svg
    :target: https://circleci.com/gh/sdpython/pandas_streaming/tree/main

.. image:: https://dev.azure.com/xavierdupre3/pandas_streaming/_apis/build/status/sdpython.pandas_streaming
    :target: https://dev.azure.com/xavierdupre3/pandas_streaming/

.. image:: https://badge.fury.io/py/pandas_streaming.svg
    :target: http://badge.fury.io/py/pandas_streaming

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: http://opensource.org/licenses/MIT

.. image:: https://codecov.io/github/sdpython/pandas_streaming/coverage.svg?branch=main
    :target: https://codecov.io/github/sdpython/pandas_streaming?branch=main

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

`pandas_streaming <http://www.xavierdupre.fr/app/pandas_streaming/helpsphinx/index.html>`_
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

**Links:**

* `GitHub/pandas_streaming <https://github.com/sdpython/pandas_streaming/>`_
* `documentation <http://www.xavierdupre.fr/app/pandas_streaming/helpsphinx/index.html>`_
* `Blog <http://www.xavierdupre.fr/app/pandas_streaming/helpsphinx/blog/main_0000.html#ap-main-0>`_
