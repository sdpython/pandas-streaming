#-*- coding: utf-8 -*-
"""
@file
@brief Defines a streaming dataframe.
"""
import pandas
import numpy.random as random
from io import StringIO
from pandas.testing import assert_frame_equal
from .dataframe_split import sklearn_train_test_split, sklearn_train_test_split_streaming
from ..exc import StreamingInefficientException


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
    on dataframe. We assume this function can be called multiple time.
    As a matter of fact, the function is called every time
    the class needs to walk through the stream with the following
    loop:

    ::

        for df in self:  # self is a StreamingDataFrame
            # ...

    The constructor cannot receive an iterator otherwise
    this class would be able to walk through the data
    only once. The main reason is it is impossible to
    :epkg:`*py:pickle` (or :epkg:`dill`)
    an iterator: it cannot be replicated.
    Instead, the class takes a function which generates
    an iterator on :epkg:`pandas:DataFrame`.
    Most of the methods returns either a :epkg:`pandas:DataFrame`
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
    """

    def __init__(self, iter_creation, check_schema=True, stable=True):
        """
        @param      iter_creation   function which creates an iterator or an instance of
                                    @see cl StreamingDataFrame
        @param      check_schema    checks that the schema is the same for every dataframe
        @param      stable          indicates if the dataframe remains the same whenever
                                    it is walked through
        """
        if isinstance(iter_creation, StreamingDataFrame):
            self.iter_creation = iter_creation.iter_creation
            self.stable = iter_creation.stable
        else:
            self.iter_creation = iter_creation
            self.stable = stable
        self.check_schema = check_schema

    def is_stable(self, do_check=False, n=10):
        """
        Tells if the dataframe is supposed to be stable.

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
                except AssertionError:
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
        Randomly splits a dataframe into smaller pieces.
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
                    raise NotImplementedError(
                        "Only train and test split is allowed, *partitions* must be of length 2.")
                kwargs = kwargs.copy()
                kwargs['train_size'] = partitions[0]
                kwargs['test_size'] = partitions[1]
            return sklearn_train_test_split_streaming(self, **kwargs)
        else:
            return sklearn_train_test_split(self, path_or_buf=path_or_buf,
                                            export_method=export_method,
                                            names=names, **kwargs)

    @staticmethod
    def _process_kwargs(kwargs):
        """
        Filters out parameters for the constructor of this class.
        """
        kw = {}
        for k in {'check_schema'}:
            if k in kwargs:
                kw[k] = kwargs[k]
                del kwargs[k]
        return kw

    @staticmethod
    def read_csv(*args, **kwargs) -> 'StreamingDataFrame':
        """
        Reads a dataframe as an iterator on DataFrame.
        The signature is the same as :epkg:`pandas:read_csv`.
        The important parameter is *chunksize* which defines the number
        of rows to parse in a single bloc.
        """
        if not kwargs.get('iterator', True):
            raise ValueError("If specified, iterator must be True.")
        kwargs_create = StreamingDataFrame._process_kwargs(kwargs)
        kwargs['iterator'] = True
        return StreamingDataFrame(lambda: pandas.read_csv(*args, **kwargs), **kwargs_create)

    @staticmethod
    def read_str(text, **kwargs) -> 'StreamingDataFrame':
        """
        Reads a dataframe as an iterator on DataFrame.
        The signature is the same as :epkg:`pandas:read_csv`.
        The important parameter is *chunksize* which defines the number
        of rows to parse in a single bloc.
        """
        if not kwargs.get('iterator', True):
            raise ValueError("If specified, iterator must be True.")
        kwargs_create = StreamingDataFrame._process_kwargs(kwargs)
        kwargs['iterator'] = True
        buffer = StringIO(text)
        return StreamingDataFrame(lambda: pandas.read_csv(buffer, **kwargs), **kwargs_create)

    @staticmethod
    def read_df(df, chunksize=None, check_schema=True) -> 'StreamingDataFrame':
        """
        Splits a dataframe into small chunks mostly for
        unit testing purposes.

        @param      df              :epkg:`pandas:DataFrame`
        @param      chunksize       number rows per chunks (// 10 by default)
        @param      check_schema    check schema between two iterations
        @return                     iterator on @see cl StreamingDataFrame
        """
        if chunksize is None:
            chunksize = df.shape[0]

        def local_iterator():
            for i in range(0, df.shape[0], chunksize):
                end = min(df.shape[0], i + chunksize)
                yield df[i:end].copy()
        return StreamingDataFrame(local_iterator, check_schema=check_schema)

    def __iter__(self):
        """
        Iterator on a large file with a sliding window.
        Each windows is a :epkg:`pandas:DataFrame`.
        The method stores a copy of the initial iterator
        and restores it after the end of the iterations.
        If *check_schema* was enabled when calling the constructor,
        the method checks that every dataframe follows the same schema
        as the first chunck.
        """
        iter = self.iter_creation()
        sch = None
        rows = 0
        for it in iter:
            if sch is None:
                sch = (list(it.columns), list(it.dtypes))
            elif self.check_schema:
                if list(it.columns) != sch[0]:
                    raise StreamingDataFrameSchemaError(
                        'Column names are different after row {0}\nFirst   chunk: {1}\nCurrent chunk: {2}'.format(rows, sch[0], list(it.columns)))
                if list(it.dtypes) != sch[1]:
                    raise StreamingDataFrameSchemaError(
                        'Column types are different after row {0}\nFirst   chunk: {1}\nCurrent chunk: {2}'.format(rows, sch[1], list(it.dtypes)))
            rows += it.shape[0]
            yield it

    def sort_values(self, *args, **kwargs):
        """
        Not implemented.
        """
        raise StreamingInefficientException(StreamingDataFrame.sort_values)

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

    @property
    def dtypes(self):
        """
        See :epkg:`pandas:DataFrame:dtypes`.
        """
        for it in self:
            return it.dtypes

    def to_csv(self, path_or_buf=None, **kwargs) -> 'StreamingDataFrame':
        """
        Saves the dataframe into string.
        See :epkg:`pandas:DataFrame.to_csv`.
        """
        if path_or_buf is None:
            st = StringIO()
            close = False
        elif isinstance(path_or_buf, str):
            st = open(path_or_buf, "w", encoding=kwargs.get('encoding'))
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
        else:
            return path_or_buf

    def to_dataframe(self) -> pandas.DataFrame:
        """
        Converts everything into a single dataframe.
        """
        return pandas.concat(self, axis=0)

    def to_df(self) -> pandas.DataFrame:
        """
        Converts everything into a single dataframe.
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
        Returns the first rows as a DataFrame.
        """
        st = []
        for df in self:
            h = df.head(n=n)
            st.append(h)
            if h.shape[0] >= n:
                break
            n -= h.shape[0]
        if len(st) == 1:
            return st[0]
        elif len(st) == 0:
            return None
        else:
            return pandas.concat(st, axis=0)

    def tail(self, n=5) -> pandas.DataFrame:
        """
        Returns the last rows as a DataFrame.
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
        return StreamingDataFrame(lambda: map(lambda df: df.where(*args, **kwargs), self), **self.get_kwargs())

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
        else:
            if cache:
                sdf = self.sample(cache=False, **kwargs)
                df = sdf.to_df()
                return StreamingDataFrame.read_df(df, chunksize=df.shape[0])
            else:
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
            for ir, row in enumerate(df.iterrows()):
                seen += 1
                if len(indices) < n:
                    indices.append((i, ir))
                else:
                    x = random.random()
                    if x * n < (seen - n):
                        k = random.randint(0, len(indices) - 1)
                        indices[k] = (i, ir)
        indices = set(indices)

        def reservoir_iterate(sdf, indices, chunksize):
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

        return StreamingDataFrame(lambda: reservoir_iterate(sdf=self, indices=indices, chunksize=1000))

    def apply(self, *args, **kwargs) -> 'StreamingDataFrame':
        """
        Applies :epkg:`pandas:DataFrame:apply`.
        This function returns a @see cl StreamingDataFrame.
        """
        return StreamingDataFrame(lambda: map(lambda df: df.apply(*args, **kwargs), self), **self.get_kwargs())

    def applymap(self, *args, **kwargs) -> 'StreamingDataFrame':
        """
        Applies :epkg:`pandas:DataFrame:applymap`.
        This function returns a @see cl StreamingDataFrame.
        """
        return StreamingDataFrame(lambda: map(lambda df: df.applymap(*args, **kwargs), self), **self.get_kwargs())

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
            for df1 in sdf1:
                for df2 in sdf2:
                    df = df1.merge(df2, **kw)
                    yield df

        return StreamingDataFrame(lambda: iterator_merge(self, right, **kwargs), **self.get_kwargs())

    def concat(self, others) -> 'StreamingDataFrame':
        """
        Concatenates dataframes. The function ensures all :epkg:`pandas:DataFrame`
        or @see cl StreamingDataFrame share the same columns (name and type).
        Otherwise, the function fails as it cannot guess the schema without
        walking through all dataframes.

        @param  others      list, enumeration, :epkg:`pandas:DataFrame`
        @return             @see cl StreamingDataFrame
        """

        def iterator_concat(this, lothers):
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
            if isinstance(obj, pandas.DataFrame):
                return StreamingDataFrame.read_df(obj, obj.shape[0])
            else:
                return obj

        others = list(map(change_type, others))
        return StreamingDataFrame(lambda: iterator_concat(self, others), **self.get_kwargs())

    def groupby(self, by=None, lambda_agg=None, in_memory=True, **kwargs) -> pandas.DataFrame:
        """
        Implements the streaming :epkg:`pandas:DataFrame:groupby`.
        We assume the result holds in memory. The out-of-memory is
        not implemented yet.

        @param      by          see :epkg:`pandas:DataFrame:groupby`
        @param      in_memory   in-memory algorithm
        @param      lambda_agg  aggregation function, *sum* by default
        @param      kwargs      additional parameters for :epkg:`pandas:DataFrame:groupby`
        @return                 :epkg:`pandas:DataFrame`

        As the input @see cl StreamingDataFrame does not necessarily hold
        in memory, the aggregation must be done at every iteration.
        There are two levels of aggregation: one to reduce every iterated
        dataframe, another one to combine all the reduced dataframes.
        This second one is always a **sum**.
        As a consequence, this function should not compute any *mean* or *count*,
        only *sum* because we do not know the size of each iterated
        dataframe. To compute an average, sum and weights must be
        aggregated.

        .. exref::
            :title: StreamingDataFrame and groupby

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
                return gr.sum()
            lambda_agg = lambda_agg_
        ckw = kwargs.copy()
        ckw["as_index"] = False
        agg = []
        for df in self:
            gr = df.groupby(by=by, **ckw)
            agg.append(lambda_agg(gr))
        conc = pandas.concat(agg)
        return conc.groupby(by=by, **kwargs).sum()

    def ensure_dtype(self, df, dtypes):
        """
        Ensures the dataframe *df* has types indicated in dtypes.
        Changes it if not.

        @param      df      dataframe
        @param      dtypes  list of types
        @return             updated?
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
        if not isinstance(cols, list):
            raise NotImplementedError("Only a list of columns is supported.")

        def iterate_cols(sdf):
            for df in sdf:
                yield df[cols]

        return StreamingDataFrame(lambda: iterate_cols(self), **self.get_kwargs())
