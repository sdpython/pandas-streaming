
pandas_streaming.df
===================

Streaming
+++++++++

The main class is an interface which mimic
:class:`pandas.DataFrame` interface to offer
a short list of methods which apply on an
iterator of dataframes. This provides somehow
a streaming version of it. As a result, the creation
of an instance is fast as long as the data is not
processed. Iterators can be chained as many map reduce
framework does.

.. toctree::
    :maxdepth: 2

    dataframe

The module implements additional and useful functions
not necessarily for the streaming version of the dataframes.
Many methods have been rewritten to support
streaming. Among them, IO methods:
:meth:`read_csv <pandas_streaming.df.dataframe.StreamingDataFrame.read_csv>`,
:meth:`read_df <pandas_streaming.df.dataframe.StreamingDataFrame.read_df>`,
:meth:`read_json <pandas_streaming.df.dataframe.StreamingDataFrame.read_json>`.

Data Manipulation
+++++++++++++++++

.. autofunction:: pandas_streaming.df.dataframe_helpers.dataframe_hash_columns

.. autofunction:: pandas_streaming.df.connex_split.dataframe_shuffle

.. autofunction:: pandas_streaming.df.dataframe_helpers.dataframe_unfold

.. autofunction:: pandas_streaming.df.dataframe_helpers.pandas_groupby_nan

Complex splits
++++++++++++++

Splitting a database into train and test is usually simple except
if rows are not independant and share some ids. In that case,
the following functions will try to build two partitions keeping
ids separate or separate as much as possible:
:func:`train_test_apart_stratify <pandas_streaming.df.connex_split.train_test_apart_stratify>`,
:func:`train_test_connex_split <pandas_streaming.df.connex_split.train_test_connex_split>`,
:func:`train_test_split_weights <pandas_streaming.df.connex_split.train_test_split_weights>`.

Extensions
++++++++++

.. toctree::
    :maxdepth: 1

    connex_split
    dataframe_io
    dataframe_split
