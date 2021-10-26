# -*- coding: utf-8 -*-
"""
@file
@brief Defines a streaming dataframe.
"""
import pickle
import os
from io import StringIO, BytesIO
from inspect import isfunction
import numpy
import numpy.random as nrandom
import pandas
from pandas.testing import assert_frame_equal
from pandas.io.json import json_normalize
from .dataframe_split import sklearn_train_test_split, sklearn_train_test_split_streaming
from .dataframe_io_helpers import enumerate_json_items, JsonIterator2Stream


class StreamingDataFrameSchemaError(Exception):
    """
    Reveals an issue with inconsistant schemas.
    """
    pass


class StreamingDataFrame:
    """
    Defines a streaming dataframe.
    The goal is to reduce the memory footprint.
    The class takes a function which creates an iterator
    on :epkg:`dataframe`. We assume this function can
    be called multiple time. As a matter of fact, the
    function is called every time the class needs to walk
    through the stream with the following loop:

    ::

        for df in self:  # self is a StreamingDataFrame
            # ...

    The constructor cannot receive an iterator otherwise
    this class would be able to walk through the data
    only once. The main reason is it is impossible to
    :epkg:`*py:pickle` (or :epkg:`dill`)
    an iterator: it cannot be replicated.
    Instead, the class takes a function which generates
    an iterator on :epkg:`DataFrame`.
    Most of the methods returns either a :epkg:`DataFrame`
    either a @see cl StreamingDataFrame. In the second case,
    methods can be chained.

    By default, the object checks that the schema remains
    the same between two chunks. This can be disabled
    by setting *check_schema=False* in the constructor.

    The user should expect the data to remain stable.
    Every loop should produce the same data. However,
    in some situations, it is more efficient not to keep
    that constraints. Draw a random @see me sample
    is one of these cases.

    :param iter_creation: function which creates an iterator or an
        instance of @see cl StreamingDataFrame
    :param check_schema: checks that the schema is the same
        for every :epkg:`dataframe`
    :param stable: indicates if the :epkg:`dataframe` remains the same
        whenever it is walked through
    """

    def __init__(self, iter_creation, check_schema=True, stable=True):
        self._delete_ = []
        if isinstance(iter_creation, (pandas.DataFrame, dict,
                                      numpy.ndarray, str)):
            raise TypeError(
                "Unexpected type %r for iter_creation. It must "
                "be an iterator." % type(iter_creation))
        if isinstance(iter_creation, StreamingDataFrame):
            self.iter_creation = iter_creation.iter_creation
            self.stable = iter_creation.stable
        else:
            self.iter_creation = iter_creation
            self.stable = stable
        self.check_schema = check_schema

    def is_stable(self, do_check=False, n=10):
        """
        Tells if the :epkg:`dataframe` is supposed to be stable.

        @param      do_check    do not trust the value sent to the constructor
        @param      n           number of rows used to check the stability,
                                None for all rows
        @return                 boolean

        *do_check=True* means the methods checks the first
        *n* rows remains the same for two iterations.
        """
        if do_check:
            for i, (a, b) in enumerate(zip(self, self)):
                if n is not None and i >= n:
                    break
                try:
                    assert_frame_equal(a, b)
                except AssertionError:  # pragma: no cover
                    return False
            return True
        else:
            return self.stable

    def get_kwargs(self):
        """
        Returns the parameters used to call the constructor.
        """
        return dict(check_schema=self.check_schema)

    def train_test_split(self, path_or_buf=None, export_method="to_csv",
                         names=None, streaming=True, partitions=None,
                         **kwargs):
        """
        Randomly splits a :epkg:`dataframe` into smaller pieces.
        The function returns streams of file names.
        It chooses one of the options from module
        :mod:`dataframe_split <pandas_streaming.df.dataframe_split>`.

        @param  path_or_buf     a string, a list of strings or buffers, if it is a
                                string, it must contain ``{}`` like ``partition{}.txt``,
                                if None, the function returns strings.
        @param  export_method   method used to store the partitions, by default
                                :epkg:`pandas:DataFrame:to_csv`, additional parameters
                                will be given to that function
        @param  names           partitions names, by default ``('train', 'test')``
        @param  kwargs          parameters for the export function and
                                :epkg:`sklearn:model_selection:train_test_split`.
        @param  streaming       the function switches to a
                                streaming version of the algorithm.
        @param  partitions      splitting partitions
        @return                 outputs of the exports functions or two
                                @see cl StreamingDataFrame if path_or_buf is None.

        The streaming version of this algorithm is implemented by function
        @see fn sklearn_train_test_split_streaming. Its documentation
        indicates the limitation of the streaming version and gives some
        insights about the additional parameters.
        """
        if streaming:
            if partitions is not None:
                if len(partitions) != 2:
                    raise NotImplementedError(  # pragma: no cover
                        "Only train and test split is allowed, *partitions* "
                        "must be of length 2.")
                kwargs = kwargs.copy()
                kwargs['train_size'] = partitions[0]
                kwargs['test_size'] = partitions[1]
            return sklearn_train_test_split_streaming(self, **kwargs)
        return sklearn_train_test_split(self, path_or_buf=path_or_buf,
                                        export_method=export_method,
                                        names=names, **kwargs)

    @staticmethod
    def _process_kwargs(kwargs):
        """
        Filters out parameters for the constructor of this class.
        """
        kw = {}
        for k in ['check_schema']:
            if k in kwargs:
                kw[k] = kwargs[k]
                del kwargs[k]
        return kw

    @staticmethod
    def read_json(*args, chunksize=100000, flatten=False, **kwargs) -> 'StreamingDataFrame':
        """
        Reads a :epkg:`json` file or buffer as an iterator
        on :epkg:`DataFrame`. The signature is the same as
        :epkg:`pandas:read_json`. The important parameter is
        *chunksize* which defines the number
        of rows to parse in a single bloc
        and it must be defined to return an iterator.
        If *lines* is True, the function falls back into
        :epkg:`pandas:read_json`, otherwise it used
        @see fn enumerate_json_items. If *lines* is ``'stream'``,
        *enumerate_json_items* is called with parameter
        ``lines=True``.
        Parameter *flatten* uses the trick described at
        `Flattening JSON objects in Python
        <https://towardsdatascience.com/flattening-json-objects-in-python-f5343c794b10>`_.
        Examples:

        .. runpython::
            :showcode:

            from io import BytesIO
            from pandas_streaming.df import StreamingDataFrame

            data = b'''{"a": 1, "b": 2}
                       {"a": 3, "b": 4}'''
            it = StreamingDataFrame.read_json(BytesIO(data), lines=True)
            dfs = list(it)
            print(dfs)

        .. runpython::
            :showcode:

            from io import BytesIO
            from pandas_streaming.df import StreamingDataFrame

            data = b'''[{"a": 1,
                         "b": 2},
                        {"a": 3,
                         "b": 4}]'''

            it = StreamingDataFrame.read_json(BytesIO(data))
            dfs = list(it)
            print(dfs)

        .. index:: IncompleteJSONError

        The parsed json must have an empty line at the end otherwise
        the following exception is raised:
        `ijson.common.IncompleteJSONError: `
        `parse error: unallowed token at this point in JSON text`.
        """
        if not isinstance(chunksize, int) or chunksize <= 0:
            raise ValueError(  # pragma: no cover
                'chunksize must be a positive integer')
        kwargs_create = StreamingDataFrame._process_kwargs(kwargs)

        if isinstance(args[0], (list, dict)):
            if flatten:
                return StreamingDataFrame.read_df(
                    json_normalize(args[0]), **kwargs_create)
            return StreamingDataFrame.read_df(args[0], **kwargs_create)

        if kwargs.get('lines', None) == 'stream':
            del kwargs['lines']

            def localf(a0=args[0]):
                if hasattr(a0, 'seek'):
                    a0.seek(0)
                return enumerate_json_items(
                    a0, encoding=kwargs.get('encoding', None), lines=True,
                    flatten=flatten)

            st = JsonIterator2Stream(localf)
            args = args[1:]

            if chunksize is None:
                return StreamingDataFrame(
                    lambda: pandas.read_json(
                        st, *args, chunksize=None, lines=True, **kwargs),
                    **kwargs_create)

            def fct1(st=st, args=args, chunksize=chunksize, kw=kwargs.copy()):
                st.seek(0)
                for r in pandas.read_json(
                        st, *args, chunksize=chunksize, nrows=chunksize,
                        lines=True, **kw):
                    yield r

            return StreamingDataFrame(fct1, **kwargs_create)

        if kwargs.get('lines', False):
            if flatten:
                raise NotImplementedError(
                    "flatten==True is implemented with option lines='stream'")
            if chunksize is None:
                return StreamingDataFrame(
                    lambda: pandas.read_json(*args, chunksize=None, **kwargs),
                    **kwargs_create)

            def fct2(args=args, chunksize=chunksize, kw=kwargs.copy()):
                for r in pandas.read_json(
                        *args, chunksize=chunksize, nrows=chunksize, **kw):
                    yield r
            return StreamingDataFrame(fct2, **kwargs_create)

        st = JsonIterator2Stream(
            lambda a0=args[0]: enumerate_json_items(
                a0, encoding=kwargs.get('encoding', None), flatten=flatten))
        args = args[1:]
        if 'lines' in kwargs:
            del kwargs['lines']

        if chunksize is None:
            return StreamingDataFrame(
                lambda: pandas.read_json(
                    st, *args, chunksize=chunksize, lines=True, **kwargs),
                **kwargs_create)

        def fct3(st=st, args=args, chunksize=chunksize, kw=kwargs.copy()):
            st.seek(0)
            for r in pandas.read_json(
                    st, *args, chunksize=chunksize, nrows=chunksize,
                    lines=True, **kw):
                yield r
        return StreamingDataFrame(fct3, **kwargs_create)

    @staticmethod
    def read_csv(*args, **kwargs) -> 'StreamingDataFrame':
        """
        Reads a :epkg:`csv` file or buffer
        as an iterator on :epkg:`DataFrame`.
        The signature is the same as :epkg:`pandas:read_csv`.
        The important parameter is *chunksize* which defines the number
        of rows to parse in a single bloc. If not specified,
        it will be equal to 100000.
        """
        if not kwargs.get('iterator', True):
            raise ValueError("If specified, iterator must be True.")
        if not kwargs.get('chunksize', 100000):
            raise ValueError("If specified, chunksize must not be None.")
        kwargs_create = StreamingDataFrame._process_kwargs(kwargs)
        kwargs['iterator'] = True
        if 'chunksize' not in kwargs:
            kwargs['chunksize'] = 100000
        return StreamingDataFrame(lambda: pandas.read_csv(*args, **kwargs), **kwargs_create)

    @staticmethod
    def read_str(text, **kwargs) -> 'StreamingDataFrame':
        """
        Reads a :epkg:`DataFrame` as an iterator on :epkg:`DataFrame`.
        The signature is the same as :epkg:`pandas:read_csv`.
        The important parameter is *chunksize* which defines the number
        of rows to parse in a single bloc.
        """
        if not kwargs.get('iterator', True):
            raise ValueError("If specified, iterator must be True.")
        if not kwargs.get('chunksize', 100000):
            raise ValueError("If specified, chunksize must not be None.")
        kwargs_create = StreamingDataFrame._process_kwargs(kwargs)
        kwargs['iterator'] = True
        if 'chunksize' not in kwargs:
            kwargs['chunksize'] = 100000
        if isinstance(text, str):
            buffer = StringIO(text)
        else:
            buffer = BytesIO(text)
        return StreamingDataFrame(
            lambda: pandas.read_csv(buffer, **kwargs), **kwargs_create)

    @staticmethod
    def read_df(df, chunksize=None, check_schema=True) -> 'StreamingDataFrame':
        """
        Splits a :epkg:`DataFrame` into small chunks mostly for
        unit testing purposes.

        @param      df              :epkg:`DataFrame`
        @param      chunksize       number rows per chunks (// 10 by default)
        @param      check_schema    check schema between two iterations
        @return                     iterator on @see cl StreamingDataFrame
        """
        if chunksize is None:
            if hasattr(df, 'shape'):
                chunksize = df.shape[0]
            else:
                raise NotImplementedError(
                    "Cannot retrieve size to infer chunksize for type={0}"
                    ".".format(type(df)))

        if hasattr(df, 'shape'):
            size = df.shape[0]
        else:
            raise NotImplementedError(  # pragma: no cover
                "Cannot retrieve size for type={0}.".format(type(df)))

        def local_iterator():
            "local iterator"
            for i in range(0, size, chunksize):
                end = min(size, i + chunksize)
                yield df[i:end].copy()
        return StreamingDataFrame(local_iterator, check_schema=check_schema)

    def __iter__(self):
        """
        Iterator on a large file with a sliding window.
        Each windows is a :epkg:`DataFrame`.
        The method stores a copy of the initial iterator
        and restores it after the end of the iterations.
        If *check_schema* was enabled when calling the constructor,
        the method checks that every :epkg:`DataFrame`
        follows the same schema as the first chunck.

        Even with a big chunk size, it might happen
        that consecutive chunks might detect different type
        for one particular column. An error message shows up
        saying ``Column types are different after row``
        with more information about the column which failed.
        In that case, :epkg:`pandas:DataFrame.read_csv` can overwrite
        the type on one column by specifying
        ``dtype={column_name: new_type}``. It frequently happens
        when a string column has many missing values.
        """
        iters = self.iter_creation()
        sch = None
        rows = 0
        for it in iters:
            if sch is None:
                sch = (list(it.columns), list(it.dtypes))
            elif self.check_schema:
                if list(it.columns) != sch[0]:  # pylint: disable=E1136
                    raise StreamingDataFrameSchemaError(  # pragma: no cover
                        'Column names are different after row {0}\nFirst   chunk: {1}'
                        '\nCurrent chunk: {2}'.format(
                            rows, sch[0], list(it.columns)))  # pylint: disable=E1136
                if list(it.dtypes) != sch[1]:  # pylint: disable=E1136
                    errdf = pandas.DataFrame(
                        dict(names=sch[0], schema1=sch[1],  # pylint: disable=E1136
                             schema2=list(it.dtypes)))  # pylint: disable=E1136
                    tdf = StringIO()
                    errdf['diff'] = errdf['schema2'] != errdf['schema1']
                    errdf = errdf[errdf['diff']]
                    errdf.to_csv(tdf, sep=",", index=False)
                    raise StreamingDataFrameSchemaError(
                        'Column types are different after row {0}. You may use option '
                        'dtype={{"column_name": str}} to force the type on this column.'
                        '\n---\n{1}'.format(rows, tdf.getvalue()))

            rows += it.shape[0]
            yield it

    @property
    def shape(self):
        """
        This is the kind of operations you do not want to do
        when a file is large because it goes through the whole
        stream just to get the number of rows.
        """
        nl, nc = 0, 0
        for it in self:
            nc = max(it.shape[1], nc)
            nl += it.shape[0]
        return nl, nc

    @property
    def columns(self):
        """
        See :epkg:`pandas:DataFrame:columns`.
        """
        for it in self:
            return it.columns
        # The dataframe is empty.
        return []

    @property
    def dtypes(self):
        """
        See :epkg:`pandas:DataFrame:dtypes`.
        """
        for it in self:
            return it.dtypes

    def to_csv(self, path_or_buf=None, **kwargs) -> 'StreamingDataFrame':
        """
        Saves the :epkg:`DataFrame` into string.
        See :epkg:`pandas:DataFrame.to_csv`.
        """
        if path_or_buf is None:
            st = StringIO()
            close = False
        elif isinstance(path_or_buf, str):
            st = open(  # pylint: disable=R1732
                path_or_buf, "w", encoding=kwargs.get('encoding'))
            close = True
        else:
            st = path_or_buf
            close = False

        for df in self:
            df.to_csv(st, **kwargs)
            kwargs['header'] = False

        if close:
            st.close()
        if isinstance(st, StringIO):
            return st.getvalue()
        return path_or_buf

    def to_dataframe(self) -> pandas.DataFrame:
        """
        Converts everything into a single :epkg:`DataFrame`.
        """
        return pandas.concat(self, axis=0)

    def to_df(self) -> pandas.DataFrame:
        """
        Converts everything into a single :epkg:`DataFrame`.
        """
        return self.to_dataframe()

    def iterrows(self):
        """
        See :epkg:`pandas:DataFrame:iterrows`.
        """
        for df in self:
            for it in df.iterrows():
                yield it

    def head(self, n=5) -> pandas.DataFrame:
        """
        Returns the first rows as a :epkg:`DataFrame`.
        """
        st = []
        total = 0
        for df in self:
            h = df.head(n=n)
            total += h.shape[0]
            st.append(h)
            if total >= n:
                break
            n -= h.shape[0]
        if len(st) == 1:
            return st[0]
        if len(st) == 0:
            return None
        return pandas.concat(st, axis=0)

    def tail(self, n=5) -> pandas.DataFrame:
        """
        Returns the last rows as a :epkg:`DataFrame`.
        The size of chunks must be greater than ``n`` to
        get ``n`` lines. This method is not efficient
        because the whole dataset must be walked through.
        """
        for df in self:
            h = df.tail(n=n)
        return h

    def where(self, *args, **kwargs) -> 'StreamingDataFrame':
        """
        Applies :epkg:`pandas:DataFrame:where`.
        *inplace* must be False.
        This function returns a @see cl StreamingDataFrame.
        """
        kwargs['inplace'] = False
        return StreamingDataFrame(
            lambda: map(lambda df: df.where(*args, **kwargs), self),
            **self.get_kwargs())

    def sample(self, reservoir=False, cache=False, **kwargs) -> 'StreamingDataFrame':
        """
        See :epkg:`pandas:DataFrame:sample`.
        Only *frac* is available, otherwise choose
        @see me reservoir_sampling.
        This function returns a @see cl StreamingDataFrame.

        @param      reservoir   use `reservoir sampling <https://en.wikipedia.org/wiki/Reservoir_sampling>`_
        @param      cache       cache the sample
        @param      kwargs      additional parameters for :epkg:`pandas:DataFrame:sample`

        If *cache* is True, the sample is cached (assuming it holds in memory).
        The second time an iterator walks through the
        """
        if reservoir or 'n' in kwargs:
            if 'frac' in kwargs:
                raise ValueError(
                    'frac cannot be specified for reservoir sampling.')
            return self._reservoir_sampling(cache=cache, n=kwargs['n'], random_state=kwargs.get('random_state'))
        if cache:
            sdf = self.sample(cache=False, **kwargs)
            df = sdf.to_df()
            return StreamingDataFrame.read_df(df, chunksize=df.shape[0])
        return StreamingDataFrame(lambda: map(lambda df: df.sample(**kwargs), self), **self.get_kwargs(), stable=False)

    def _reservoir_sampling(self, cache=True, n=1000, random_state=None) -> 'StreamingDataFrame':
        """
        Uses the `reservoir sampling <https://en.wikipedia.org/wiki/Reservoir_sampling>`_
        algorithm to draw a random sample with exactly *n* samples.

        @param      cache           cache the sample
        @param      n               number of observations to keep
        @param      random_state    sets the random_state
        @return                     @see cl StreamingDataFrame

        .. warning::
            The sample is split by chunks of size 1000.
            This parameter is not yet exposed.
        """
        if not cache:
            raise ValueError(
                "cache=False is not available for reservoir sampling.")
        indices = []
        seen = 0
        for i, df in enumerate(self):
            for ir, _ in enumerate(df.iterrows()):
                seen += 1
                if len(indices) < n:
                    indices.append((i, ir))
                else:
                    x = nrandom.random()  # pylint: disable=E1101
                    if x * n < (seen - n):
                        k = nrandom.randint(0, len(indices) - 1)
                        indices[k] = (i, ir)  # pylint: disable=E1126
        indices = set(indices)

        def reservoir_iterate(sdf, indices, chunksize):
            "iterator"
            buffer = []
            for i, df in enumerate(self):
                for ir, row in enumerate(df.iterrows()):
                    if (i, ir) in indices:
                        buffer.append(row)
                        if len(buffer) >= chunksize:
                            yield pandas.DataFrame(buffer)
                            buffer.clear()
            if len(buffer) > 0:
                yield pandas.DataFrame(buffer)

        return StreamingDataFrame(
            lambda: reservoir_iterate(sdf=self, indices=indices, chunksize=1000))

    def apply(self, *args, **kwargs) -> 'StreamingDataFrame':
        """
        Applies :epkg:`pandas:DataFrame:apply`.
        This function returns a @see cl StreamingDataFrame.
        """
        return StreamingDataFrame(
            lambda: map(lambda df: df.apply(*args, **kwargs), self),
            **self.get_kwargs())

    def applymap(self, *args, **kwargs) -> 'StreamingDataFrame':
        """
        Applies :epkg:`pandas:DataFrame:applymap`.
        This function returns a @see cl StreamingDataFrame.
        """
        return StreamingDataFrame(
            lambda: map(lambda df: df.applymap(*args, **kwargs), self),
            **self.get_kwargs())

    def merge(self, right, **kwargs) -> 'StreamingDataFrame':
        """
        Merges two @see cl StreamingDataFrame and returns @see cl StreamingDataFrame.
        *right* can be either a @see cl StreamingDataFrame or simply
        a :epkg:`pandas:DataFrame`. It calls :epkg:`pandas:DataFrame:merge` in
        a double loop, loop on *self*, loop on *right*.
        """
        if isinstance(right, pandas.DataFrame):
            return self.merge(StreamingDataFrame.read_df(right, chunksize=right.shape[0]), **kwargs)

        def iterator_merge(sdf1, sdf2, **kw):
            "iterate on dataframes"
            for df1 in sdf1:
                for df2 in sdf2:
                    df = df1.merge(df2, **kw)
                    yield df

        return StreamingDataFrame(
            lambda: iterator_merge(self, right, **kwargs), **self.get_kwargs())

    def concat(self, others, axis=0) -> 'StreamingDataFrame':
        """
        Concatenates :epkg:`dataframes`. The function ensures all :epkg:`pandas:DataFrame`
        or @see cl StreamingDataFrame share the same columns (name and type).
        Otherwise, the function fails as it cannot guess the schema without
        walking through all :epkg:`dataframes`.

        :param others: list, enumeration, :epkg:`pandas:DataFrame`
        :param axis: concatenate by rows (0) or by columns (1)
        :return: @see cl StreamingDataFrame
        """
        if axis == 1:
            return self._concath(others)
        if axis == 0:
            return self._concatv(others)
        raise ValueError("axis must be 0 or 1")  # pragma: no cover

    def _concath(self, others):
        if not isinstance(others, list):
            others = [others]

        def iterateh(self, others):
            cols = tuple([self] + others)
            for dfs in zip(*cols):
                nrows = [_.shape[0] for _ in dfs]
                if min(nrows) != max(nrows):
                    raise RuntimeError(
                        "StreamingDataFram cannot merge DataFrame with different size or chunksize")
                yield pandas.concat(list(dfs), axis=1)

        return StreamingDataFrame(lambda: iterateh(self, others), **self.get_kwargs())

    def _concatv(self, others):

        def iterator_concat(this, lothers):
            "iterator on dataframes"
            columns = None
            dtypes = None
            for df in this:
                if columns is None:
                    columns = df.columns
                    dtypes = df.dtypes
                yield df
            for obj in lothers:
                check = True
                for i, df in enumerate(obj):
                    if check:
                        if list(columns) != list(df.columns):
                            raise ValueError(
                                "Frame others[{0}] do not have the same column names or the same order.".format(i))
                        if list(dtypes) != list(df.dtypes):
                            raise ValueError(
                                "Frame others[{0}] do not have the same column types.".format(i))
                        check = False
                    yield df

        if isinstance(others, pandas.DataFrame):
            others = [others]
        elif isinstance(others, StreamingDataFrame):
            others = [others]

        def change_type(obj):
            "change column type"
            if isinstance(obj, pandas.DataFrame):
                return StreamingDataFrame.read_df(obj, obj.shape[0])
            else:
                return obj

        others = list(map(change_type, others))
        return StreamingDataFrame(
            lambda: iterator_concat(self, others), **self.get_kwargs())

    def groupby(self, by=None, lambda_agg=None, lambda_agg_agg=None,
                in_memory=True, **kwargs) -> pandas.DataFrame:
        """
        Implements the streaming :epkg:`pandas:DataFrame:groupby`.
        We assume the result holds in memory. The out-of-memory is
        not implemented yet.

        @param      by              see :epkg:`pandas:DataFrame:groupby`
        @param      in_memory       in-memory algorithm
        @param      lambda_agg      aggregation function, *sum* by default
        @param      lambda_agg_agg  to aggregate the aggregations, *sum* by default
        @param      kwargs          additional parameters for :epkg:`pandas:DataFrame:groupby`
        @return                     :epkg:`pandas:DataFrame`

        As the input @see cl StreamingDataFrame does not necessarily hold
        in memory, the aggregation must be done at every iteration.
        There are two levels of aggregation: one to reduce every iterated
        :epkg:`dataframe`, another one to combine all the reduced :epkg:`dataframes`.
        This second one is always a **sum**.
        As a consequence, this function should not compute any *mean* or *count*,
        only *sum* because we do not know the size of each iterated
        :epkg:`dataframe`. To compute an average, sum and weights must be
        aggregated.

        Parameter *lambda_agg* is ``lambda gr: gr.sum()`` by default.
        It could also be ``lambda gr: gr.max()`` or
        ``lambda gr: gr.min()`` but not ``lambda gr: gr.mean()``
        as it would lead to incoherent results.

        .. exref::
            :title: StreamingDataFrame and groupby
            :tag: streaming

            Here is an example which shows how to write a simple *groupby*
            with :epkg:`pandas` and @see cl StreamingDataFrame.

            .. runpython::
                :showcode:

                from pandas import DataFrame
                from pandas_streaming.df import StreamingDataFrame

                df = DataFrame(dict(A=[3, 4, 3], B=[5,6, 7]))
                sdf = StreamingDataFrame.read_df(df)

                # The following:
                print(sdf.groupby("A", lambda gr: gr.sum()))

                # Is equivalent to:
                print(df.groupby("A").sum())
        """
        if not in_memory:
            raise NotImplementedError(
                "Out-of-memory group by is not implemented.")
        if lambda_agg is None:
            def lambda_agg_(gr):
                "sum"
                return gr.sum()
            lambda_agg = lambda_agg_
        if lambda_agg_agg is None:
            def lambda_agg_agg_(gr):
                "sum"
                return gr.sum()
            lambda_agg_agg = lambda_agg_agg_
        ckw = kwargs.copy()
        ckw["as_index"] = False

        agg = []
        for df in self:
            gr = df.groupby(by=by, **ckw)
            agg.append(lambda_agg(gr))
        conc = pandas.concat(agg, sort=False)
        return lambda_agg_agg(conc.groupby(by=by, **kwargs))

    def groupby_streaming(self, by=None, lambda_agg=None, lambda_agg_agg=None, in_memory=True,
                          strategy='cum', **kwargs) -> pandas.DataFrame:
        """
        Implements the streaming :epkg:`pandas:DataFrame:groupby`.
        We assume the result holds in memory. The out-of-memory is
        not implemented yet.

        :param by: see :epkg:`pandas:DataFrame:groupby`
        :param in_memory: in-memory algorithm
        :param lambda_agg: aggregation function, *sum* by default
        :param lambda_agg_agg: to aggregate the aggregations, *sum* by default
        :param kwargs: additional parameters for :epkg:`pandas:DataFrame:groupby`
        :param strategy: ``'cum'``, or ``'streaming'``, see below
        :return: :epkg:`pandas:DataFrame`

        As the input @see cl StreamingDataFrame does not necessarily hold
        in memory, the aggregation must be done at every iteration.
        There are two levels of aggregation: one to reduce every iterated
        :epkg:`dataframe`, another one to combine all the reduced :epkg:`dataframes`.
        This second one is always a **sum**.
        As a consequence, this function should not compute any *mean* or *count*,
        only *sum* because we do not know the size of each iterated
        :epkg:`dataframe`. To compute an average, sum and weights must be
        aggregated.

        Parameter *lambda_agg* is ``lambda gr: gr.sum()`` by default.
        It could also be ``lambda gr: gr.max()`` or
        ``lambda gr: gr.min()`` but not ``lambda gr: gr.mean()``
        as it would lead to incoherent results.

        Parameter *strategy* allows three scenarios.
        First one if ``strategy is None`` goes through
        the whole datasets to produce a final :epkg:`DataFrame`.
        Second if ``strategy=='cum'`` returns a
        @see cl StreamingDataFrame, each iteration produces
        the current status of the *group by*. Last case,
        ``strategy=='streaming'`` produces :epkg:`DataFrame`
        which must be concatenated into a single :epkg:`DataFrame`
        and grouped again to get the results.

        .. exref::
            :title: StreamingDataFrame and groupby
            :tag: streaming

            Here is an example which shows how to write a simple *groupby*
            with :epkg:`pandas` and @see cl StreamingDataFrame.

            .. runpython::
                :showcode:

                from pandas import DataFrame
                from pandas_streaming.df import StreamingDataFrame
                from pandas_streaming.data import dummy_streaming_dataframe

                df20 = dummy_streaming_dataframe(20).to_dataframe()
                df20["key"] = df20["cint"].apply(lambda i: i % 3 == 0)
                sdf20 = StreamingDataFrame.read_df(df20, chunksize=5)
                sgr = sdf20.groupby_streaming("key", lambda gr: gr.sum(),
                                              strategy='cum', as_index=False)
                for gr in sgr:
                    print()
                    print(gr)
        """
        if not in_memory:
            raise NotImplementedError(
                "Out-of-memory group by is not implemented.")
        if lambda_agg is None:
            def lambda_agg_(gr):
                "sum"
                return gr.sum()
            lambda_agg = lambda_agg_
        if lambda_agg_agg is None:
            def lambda_agg_agg_(gr):
                "sum"
                return gr.sum()
            lambda_agg_agg = lambda_agg_agg_
        ckw = kwargs.copy()
        ckw["as_index"] = False

        if strategy == 'cum':
            def iterate_cum():
                agg = None
                for df in self:
                    gr = df.groupby(by=by, **ckw)
                    gragg = lambda_agg(gr)
                    if agg is None:
                        yield lambda_agg_agg(gragg.groupby(by=by, **kwargs))
                        agg = gragg
                    else:
                        lagg = pandas.concat([agg, gragg], sort=False)
                        yield lambda_agg_agg(lagg.groupby(by=by, **kwargs))
                        agg = lagg
            return StreamingDataFrame(lambda: iterate_cum(), **self.get_kwargs())

        if strategy == 'streaming':
            def iterate_streaming():
                for df in self:
                    gr = df.groupby(by=by, **ckw)
                    gragg = lambda_agg(gr)
                    yield lambda_agg(gragg.groupby(by=by, **kwargs))
            return StreamingDataFrame(lambda: iterate_streaming(), **self.get_kwargs())

        raise ValueError(  # pragma: no cover
            "Unknown strategy '{0}'".format(strategy))

    def ensure_dtype(self, df, dtypes):
        """
        Ensures the :epkg:`dataframe` *df* has types indicated in dtypes.
        Changes it if not.

        :param df: dataframe
        :param dtypes: list of types
        :return: updated?
        """
        ch = False
        cols = df.columns
        for i, (has, exp) in enumerate(zip(df.dtypes, dtypes)):
            if has != exp:
                name = cols[i]
                df[name] = df[name].astype(exp)
                ch = True
        return ch

    def __getitem__(self, *args):
        """
        Implements some of the functionalities :epkg:`pandas`
        offers for the operator ``[]``.
        """
        if len(args) != 1:
            raise NotImplementedError("Only a list of columns is supported.")
        cols = args[0]
        if isinstance(cols, str):
            # One column.
            iter_creation = self.iter_creation

            def iterate_col():
                "iterate on one column"
                one_col = [cols]
                for df in iter_creation():
                    yield df[one_col]
            return StreamingSeries(iterate_col, **self.get_kwargs())

        if not isinstance(cols, list):
            raise NotImplementedError("Only a list of columns is supported.")

        def iterate_cols(sdf):
            """Iterate on columns."""
            for df in sdf:
                yield df[cols]

        return StreamingDataFrame(lambda: iterate_cols(self), **self.get_kwargs())

    def __setitem__(self, index, value):
        """
        Limited set of operators are supported.
        """
        if not isinstance(index, str):
            raise ValueError(
                "Only column affected are supported but index=%r." % index)
        if isinstance(value, (int, float, numpy.number, str)):
            # Is is equivalent to add_column.
            iter_creation = self.iter_creation

            def iterate_fct():
                "iterate on rows"
                iters = iter_creation()
                for df in iters:
                    dfc = df.copy()
                    dfc[index] = value
                    yield dfc

            self.iter_creation = iterate_fct

        elif isinstance(value, StreamingSeries):
            iter_creation = self.iter_creation

            def iterate_fct():
                "iterate on rows"
                iters = iter_creation()
                for df, dfs in zip(iters, value):
                    if df.shape[0] != dfs.shape[0]:
                        raise RuntimeError(
                            "Chunksize or shape are different when "
                            "iterating on two StreamDataFrame at the same "
                            "time: %r != %r." % (df.shape[0], dfs.shape[0]))
                    dfc = df.copy()
                    dfc[index] = dfs
                    yield dfc

            self.iter_creation = iterate_fct
        else:
            raise NotImplementedError(
                "Not implemented for type(index)=%r and type(value)=%r." % (
                    type(index), type(value)))

    def add_column(self, col, value):
        """
        Implements some of the functionalities :epkg:`pandas`
        offers for the operator ``[]``.

        @param      col             new column
        @param      value           @see cl StreamingDataFrame or a lambda function
        @return                     @see cl StreamingDataFrame

        ..note::

            If value is a @see cl StreamingDataFrame,
            *chunksize* must be the same for both.

        .. exref::
            :title: Add a new column to a StreamingDataFrame
            :tag: streaming

            .. runpython::
                :showcode:

                from pandas import DataFrame
                from pandas_streaming.df import StreamingDataFrame

                df = DataFrame(data=dict(X=[4.5, 6, 7], Y=["a", "b", "c"]))
                sdf = StreamingDataFrame.read_df(df)
                sdf2 = sdf.add_column("d", lambda row: int(1))
                print(sdf2.to_dataframe())

                sdf2 = sdf.add_column("d", lambda row: int(1))
                print(sdf2.to_dataframe())

        """
        if not isinstance(col, str):
            raise NotImplementedError(
                "Only a column as a string is supported.")

        if isfunction(value):
            def iterate_fct(self, value, col):
                "iterate on rows"
                for df in self:
                    dfc = df.copy()
                    dfc.insert(dfc.shape[1], col, dfc.apply(value, axis=1))
                    yield dfc

            return StreamingDataFrame(lambda: iterate_fct(self, value, col), **self.get_kwargs())

        if isinstance(value, (pandas.Series, pandas.DataFrame, StreamingDataFrame)):
            raise NotImplementedError(
                "Unable set a new column based on a datadframe.")

        def iterate_cst(self, value, col):
            "iterate on rows"
            for df in self:
                dfc = df.copy()
                dfc[col] = value
                yield dfc

        return StreamingDataFrame(
            lambda: iterate_cst(self, value, col), **self.get_kwargs())

    def fillna(self, **kwargs):
        """
        Replaces the missing values, calls
        :epkg:`pandas:DataFrame:fillna`.

        @param      kwargs      see :epkg:`pandas:DataFrame:fillna`
        @return                 @see cl StreamingDataFrame

        .. warning::
            The function does not check what happens at the
            limit of every chunk of data. Anything but a constant value
            will probably have an inconsistent behaviour.
        """

        def iterate_na(self, **kwargs):
            "iterate on rows"
            if kwargs.get('inplace', True):
                kwargs['inplace'] = True
                for df in self:
                    df.fillna(**kwargs)
                    yield df
            else:
                for df in self:
                    yield df.fillna(**kwargs)

        return StreamingDataFrame(
            lambda: iterate_na(self, **kwargs), **self.get_kwargs())

    def describe(self, percentiles=None, include=None, exclude=None,
                 datetime_is_numeric=False):
        """
        Calls :epkg:`pandas:DataFrame:describe` on every piece
        of the datasets. *percentiles* are not really accurate
        but just an indication.

        :param percentiles: see :epkg:`pandas:DataFrame:describe`
        :param include: see :epkg:`pandas:DataFrame:describe`
        :param exclude: see :epkg:`pandas:DataFrame:describe`
        :param datetime_is_numeric: see :epkg:`pandas:DataFrame:describe`
        :return: :epkg:`pandas:DataFrame:describe`
        """
        merged = None
        stack = []
        notper = ['count', 'mean', 'std']
        for df in self:
            desc = df.describe(
                percentiles=percentiles, include=include, exclude=exclude,
                datetime_is_numeric=datetime_is_numeric)
            count = desc.loc['count', :]
            rows = [name for name in desc.index if name not in notper]
            stack.append(desc.loc[rows, :])
            if merged is None:
                merged = desc
                merged.loc['std', :] = (
                    merged.loc['std', :] ** 2 + merged.loc['mean', :] ** 2) * count
                merged.loc['mean', :] *= count
            else:
                merged.loc['count', :] += desc.loc['count', :]
                merged.loc['mean', :] += desc.loc['mean', :] * count
                merged.loc['std', :] += (
                    desc.loc['std', :] ** 2 + desc.loc['mean', :] ** 2) * count
                merged.loc['max', :] = numpy.maximum(
                    merged.loc['max', :], desc.loc['max', :])
                merged.loc['min', :] = numpy.maximum(
                    merged.loc['min', :], desc.loc['min', :])
        merged.loc['mean', :] /= merged.loc['count', :]
        merged.loc['std', :] = (
            merged.loc['std', :] / merged.loc['count', :] -
            merged.loc['mean', :] ** 2) ** 0.5
        values = pandas.concat(stack)
        summary = values.describe(percentiles=percentiles,
                                  datetime_is_numeric=datetime_is_numeric)
        merged = merged.loc[notper, :]
        rows = [name for name in summary.index if name not in notper]
        summary = summary.loc[rows, :]
        return pandas.concat([merged, summary])

    def sort_values(self, by, axis=0, ascending=True, kind='quicksort',
                    na_position='last',
                    temp_file='_pandas_streaming_sort_values_'):
        """
        Sorts the streaming dataframe by values.

        :param by: one column
        :param ascending: order
        :param kind: see :meth:`pandas.DataFrame.sort_values`
        :param na_position: see :meth:`pandas.DataFrame.sort_values`
        :param temp_file: sorting a whole database is impossible
            without storing intermediate results on disk
            unless it can fit into the memory, but in that case,
            it is easier to convert the streaming database into
            a dataframe and sort it
        :return: streaming database
        """
        if not isinstance(by, str):
            raise NotImplementedError(
                "Only one column can be used to sort not %r." % by)
        keys = {}
        nans = []
        indices = []
        with open(temp_file, 'wb') as f:
            for df in self:
                dfs = df.sort_values(by, ascending=ascending, kind=kind,
                                     na_position=na_position)
                for tu in dfs[by]:
                    if isinstance(tu, float) and numpy.isnan(tu):
                        nans.append(len(indices))
                    else:
                        if tu not in keys:
                            keys[tu] = []
                        keys[tu].append(len(indices))
                indices.append(f.tell())
                st = BytesIO()
                pickle.dump(dfs, st)
                f.write(st.getvalue())

            indices.append(f.tell())

        values = list(keys.items())
        values.sort(reverse=not ascending)

        def iterate():

            with open(temp_file, 'rb') as f:

                if na_position == 'first':
                    for p in nans:
                        f.seek(indices[p])
                        length = indices[p + 1] - indices[p]
                        pkl = f.read(length)
                        dfs = pickle.load(BytesIO(pkl))
                        sub = dfs[numpy.isnan(dfs[by])]
                        yield sub

                for key, positions in values:
                    for p in positions:
                        f.seek(indices[p])
                        length = indices[p + 1] - indices[p]
                        pkl = f.read(length)
                        dfs = pickle.load(BytesIO(pkl))
                        sub = dfs[dfs[by] == key]
                        yield sub

                if na_position == 'last':
                    for p in nans:
                        f.seek(indices[p])
                        length = indices[p + 1] - indices[p]
                        pkl = f.read(length)
                        dfs = pickle.load(BytesIO(pkl))
                        sub = dfs[numpy.isnan(dfs[by])]
                        yield sub

        res = StreamingDataFrame(
            lambda: iterate(), **self.get_kwargs())
        res._delete_.append(lambda: os.remove(temp_file))
        return res

    def __del__(self):
        """
        Calls every function in `_delete_`.
        """
        for f in self._delete_:
            f()


class StreamingSeries(StreamingDataFrame):
    """
    Seens as a @see cl StreamingDataFrame of one column.
    """

    def __init__(self, iter_creation, check_schema=True, stable=True):
        StreamingDataFrame.__init__(
            self, iter_creation, check_schema=check_schema, stable=stable)
        if len(self.columns) != 1:
            raise RuntimeError(
                "A series can contain only one column not %r." % len(self.columns))

    def apply(self, *args, **kwargs) -> 'StreamingDataFrame':
        """
        Applies :epkg:`pandas:Series:apply`.
        This function returns a @see cl StreamingSeries.
        """
        return StreamingSeries(
            lambda: map(lambda df: df.apply(*args, **kwargs), self),
            **self.get_kwargs())

    def __add__(self, value):
        """
        Does an addition on every value hoping that has a meaning.

        :param value: any value which makes sense
        :return: a new series
        """
        def iterate():
            for df in self:
                yield df + value

        return StreamingSeries(iterate, **self.get_kwargs())
