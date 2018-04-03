
.. _l-README:

README
======

.. image:: https://travis-ci.org/sdpython/pandas_streaming.svg?branch=master
    :target: https://travis-ci.org/sdpython/pandas_streaming
    :alt: Build status

.. image:: https://ci.appveyor.com/api/projects/status/4te066r8ne1ymmhy?svg=true
    :target: https://ci.appveyor.com/project/sdpython/pandas-streaming
    :alt: Build Status Windows

.. image:: https://circleci.com/gh/sdpython/pandas_streaming/tree/master.svg?style=svg
    :target: https://circleci.com/gh/sdpython/pandas_streaming/tree/master

.. image:: https://badge.fury.io/py/pandas_streaming.svg
    :target: http://badge.fury.io/py/pandas_streaming

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: http://opensource.org/licenses/MIT

.. image:: https://requires.io/github/sdpython/pandas_streaming/requirements.svg?branch=master
     :target: https://requires.io/github/sdpython/pandas_streaming/requirements/?branch=master
     :alt: Requirements Status

.. image:: https://codecov.io/github/sdpython/pandas_streaming/coverage.svg?branch=master
    :target: https://codecov.io/github/sdpython/pandas_streaming?branch=master

.. image:: http://img.shields.io/github/issues/sdpython/pandas_streaming.png
    :alt: GitHub Issues
    :target: https://github.com/sdpython/pandas_streaming/issues

.. image:: https://badge.waffle.io/sdpython/pandas_streaming.png?label=to%20do&title=to%20do
    :alt: Waffle
    :target: https://waffle.io/sdpython/pandas_streaming

.. image:: http://www.xavierdupre.fr/app/pandas_streaming/helpsphinx/_images/nbcov.png
    :target: http://www.xavierdupre.fr/app/pandas_streaming/helpsphinx/all_notebooks_coverage.html
    :alt: Notebook Coverage

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

**Links:**

* `GitHub/pandas_streaming <https://github.com/sdpython/pandas_streaming/>`_
* `documentation <http://www.xavierdupre.fr/app/pandas_streaming/helpsphinx/index.html>`_
* `Blog <http://www.xavierdupre.fr/app/pandas_streaming/helpsphinx/blog/main_0000.html#ap-main-0>`_
