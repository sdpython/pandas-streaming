
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

.. contents::
    :local:

Check the schema consistency of a large file
++++++++++++++++++++++++++++++++++++++++++++

Large files usually comes from an export of a database and this
for some reason, this export failed for a couple of lines.
It can be character *end of line* not removed from a comment,
a separator also present in the data. When that happens, :epkg:`pandas`
takes the least strict type as the column type. Sometimes, we prefer to get a
an idea of where we could find the error.

.. runpython::
    :showcode:

    import pandas
    df = pandas.DataFrame([dict(cf=0, cint=0, cstr="0"), dict(cf=1, cint=1, cstr="1"),
                           dict(cf=2, cint="s2", cstr="2"), dict(cf=3, cint=3, cstr="3")])
    name = "temp_df.csv"
    df.to_csv(name, index=False)

    from pandas_streaming.df import StreamingDataFrame
    try:
        sdf = StreamingDataFrame.read_csv(name, chunksize=2)
        for df in sdf:
            print(df.dtypes)
    except Exception as e:
        print(e)

The method :py:meth:`__iter__ <pandas_streaming.df.dataframe.StreamingDataFrame.__iter__>`
checks that the schema does not change between two iterations.
It can be disabled by adding *check_schema=False* when
the constructor is called.
