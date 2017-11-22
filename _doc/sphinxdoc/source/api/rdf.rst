
pandas_streaming.df
===================

.. contents::
    :local:

Streaming
+++++++++

The main class is an interface which mimic
:epkg:`pandas:DataFrame` interface to offer
a short list of methods which apply on an
iterator of dataframes. This provides somehow
a streaming version of it. As a result, the creation
of an instance is fast as long as the data is not
processed. Iterators can be chained as many map reduce
framework does.

.. autosignature:: pandas_streaming.df.dataframe.StreamingDataFrame

The module implements additional and useful functions
not necessarily for the streaming version of the dataframes.

Data Manipulation
+++++++++++++++++

.. autosignature:: pandas_streaming.df.dataframe_helpers.dataframe_hash_columns

.. autosignature:: pandas_streaming.df.connex_split.dataframe_shuffle

.. autosignature:: pandas_streaming.df.dataframe_helpers.dataframe_unfold

Complex splits
++++++++++++++

Splitting a database into train and test is usually simple except
if rows are not independant and share some ids. In that case,
the following functions will try to build two partitions keeping
ids separate or separate as much as possible.

.. autosignature:: pandas_streaming.df.connex_split.train_test_connex_split

.. autosignature:: pandas_streaming.df.connex_split.train_test_split_weights
