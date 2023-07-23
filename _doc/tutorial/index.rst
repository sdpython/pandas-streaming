
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

Objectives and Competitors
++++++++++++++++++++++++++

The first objective is speed.
:class:`StreamingDataFrame <pandas_streaming.df.dataframe.StreamingDataFrame>`
is useful when the user needs to process a large data set which does not
hold in memory (*out-of-memory dataset*) or when the user needs to fastly
check an algorithm on the beginning of a big dataset without paying the
cost of loading the data.

The second objective is simplicity. The proposed interface
tries to follow the same syntax as :epkg:`pandas`.
That is one of the direction followed by :epkg:`dask`.

:epkg:`dask` tries to address these two objectives
and also offers parallelization. Based on my experience,
:epkg:`dask` is efficient but tends to be slow for simple things
on medium datasets (a couple of gigabytes). The API is not exactly
the same either. The parser does not behave exactly the same.
:epkg:`pyspark` seems a bit of overhead, more difficult
to install and still slow if it is used locally.
:epkg:`pyarrow` is supposed to be the next :epkg:`pandas` but its
scope is larger (it handles streaming dataset from :epkg:`Hadoop`)
and does not work yet with :epkg:`scikit-learn`.
I expect this module to be live until
:epkg:`scikit-learn` updates its code to handle
a streaming container. This one will probably be
the winner.
:epkg:`streamz` follows a different direction.
It offers parallelisation, relies on :epkg:`tornado` but not
on :epkg:`pandas` meaning using it for machine learning
might hide some unexpected loopholes.
:epkg:`scikit-multiflow` does not only implement streaming
container but also streaming machine learning trainings.

One element of design to remember
+++++++++++++++++++++++++++++++++

The class :class:`StreamingDataFrame <pandas_streaming.df.dataframe.StreamingDataFrame>`
does not hold an iterator but a function which creates an iterator.
Every time the user writes the following loop, the function is called
to create an iterator then used to walk through the data.

.. runpython::
    :showcode:

    import pandas
    df = pandas.DataFrame([dict(cf=0, cint=0, cstr="0"), dict(cf=1, cint=1, cstr="1"),
                           dict(cf=3, cint=3, cstr="3")])

    from pandas_streaming.df import StreamingDataFrame
    sdf = StreamingDataFrame.read_df(df, chunksize=2)

    print("First time:")

    for df in sdf:
        # process this chunk of data
        print(df)

    print("\nSecond time:\n")

    for df in sdf:
        # process this chunk of data a second time
        print(df)

The reason why the class cannot directly use an iterator is because
it is not possible to pickle an iterator. An iterator is meant to
be used only once, a second loop would not be possible and would
be quite surprising to most of users.

A :class:`StreamingDataFrame <pandas_streaming.df.dataframe.StreamingDataFrame>`
is also supposed to be *stable*: the two loops in the previous example
should produce the exact same chunks. However, in some cases, the user can choose
not to abide by this constraint. Drawing a sample is one of the reasons.
A user can either choose to draw the same sample every time he is going
through the data. He could also choose that a different sample should be
drawn each time. The following method indicates which kinds of sample
the :class:`StreamingDataFrame <pandas_streaming.df.dataframe.StreamingDataFrame>`
is producing.

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
        print("ERROR:", e)

The method :meth:`__iter__
<pandas_streaming.df.dataframe.StreamingDataFrame.__iter__>`
checks that the schema does not change between two iterations.
It can be disabled by adding *check_schema=False* when
the constructor is called.
