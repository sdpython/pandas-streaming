
Tutorial
========

The main class :class:`StreamingDataFrame <pandas_streaming.df.dataframe.StreamingDataFrame>`
is basically on iterator on dataframes. Altogether, it is a
single dataframe which does not have to fit in memory.
It implements a subset a functionalities :epkg:`pandas` provides
related to map reduce,
:meth:`concat <pandas_streaming.df.dataframe.StreamingDataFrame.concat>`,
:meth:`join <pandas_streaming.df.dataframe.StreamingDataFrame.concat>`.
Both return a :class:`StreamingDataFrame <pandas_streaming.df.dataframe.StreamingDataFrame>`
as opposed to :meth:`groupby <pandas_streaming.df.dataframe.StreamingDataFrame.concat>`
which does not.

The beginning is always the same, we create such object with one
method :meth:`read_csv <pandas_streaming.df.dataframe.StreamingDataFrame.read_csv>`,
:meth:`read_df <pandas_streaming.df.dataframe.StreamingDataFrame.read_df>`,
:meth:`read_str <pandas_streaming.df.dataframe.StreamingDataFrame.read_str>`.
The module was initially created to easily split a dataset into train/test
when it does not fit into memory.

::

    from pandas_streaming.df import StreamingDataFrame
    sdf = StreamingDataFrame.read_csv("<filename>", sep="\t")
    sdf.train_test_split("dataset_split_{}.txt", sep="\t")

    >>> ['dataset_split_train.txt', 'dataset_split_test.txt']
