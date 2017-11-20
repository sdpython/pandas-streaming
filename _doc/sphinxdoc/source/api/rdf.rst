
pandas_streaming.df
===================

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

.. autosignature:: pandas_streaming.df.connex_split.dataframe_shuffle

.. autosignature:: pandas_streaming.df.connex_split.train_test_connex_split

.. autosignature:: pandas_streaming.df.connex_split.train_test_split_weights
