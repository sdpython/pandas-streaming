#-*- coding: utf-8 -*-
"""
@file
@brief Defines a streming dataframe.
"""
import warnings
import pandas
from io import StringIO
from ..exc import StreamingInefficientException


class StreamingDataFrame:
    """
    Defines a streaming dataframe.

    The constructor cannot receive an iterator otherwise
    this class would be able to walk through the data
    only once. Instead, it takes a function which generates
    an iterator on :epkg:`pandas:DataFrame`.
    Most of the functions returns either :epkg:`pandas:DataFrame`
    or a @see cl StreamingDataFrame. In the second case,
    methods can be chained.
    """

    def __init__(self, iter_creation):
        """
        Wraps a iterator on dataframe.

        @param      iter_creation   code which creates an iterator.
        """
        self.iter_creation = iter_creation

    @staticmethod
    def read_csv(*args, **kwargs):
        """
        Reads a dataframe as an iterator on DataFrame.
        The signature is the same as :epkg:`pandas:read_csv`.
        The important parameter is *chunksize* which defines the number
        of rows to parse in a single bloc.
        """
        if not kwargs.get('iterator', True):
            raise ValueError("If specified, iterator must be True.")
        kwargs['iterator'] = True
        return StreamingDataFrame(lambda: pandas.read_csv(*args, **kwargs))

    @staticmethod
    def read_str(text, **kwargs):
        """
        Reads a dataframe as an iterator on DataFrame.
        The signature is the same as :epkg:`pandas:read_csv`.
        The important parameter is *chunksize* which defines the number
        of rows to parse in a single bloc.
        """
        if not kwargs.get('iterator', True):
            raise ValueError("If specified, iterator must be True.")
        kwargs['iterator'] = True
        buffer = StringIO(text)
        return StreamingDataFrame(lambda: pandas.read_csv(buffer, **kwargs))

    @staticmethod
    def read_df(df, chunk_size=None):
        """
        Splits a dataframe into small chunks mostly for
        unit testing purposes.

        @param      df          :epkg:`pandas:DataFrame`
        @param      chunk_size  number rows per chunks (// 10 by default)
        @return                 iterator on @see cl StreamingDataFrame
        """
        if chunk_size is None:
            chunk_size = max(1, len(df) // 10)

        def local_iterator():
            for i in range(0, df.shape[0], chunk_size):
                end = min(df.shape[0], i + chunk_size)
                yield df[i:end].copy()
        return StreamingDataFrame(local_iterator)

    def __iter__(self):
        """
        Iterator on a large file with a sliding window.
        Each windows is a :epkg:`pandas:DataFrame`.
        The method stores a copy of the initial iterator
        and restores it after the end of the iterations.
        """
        iter = self.iter_creation()
        for it in iter:
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

    def to_csv(self, path_or_buf=None, **kwargs):
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

    def to_dataframe(self):
        """
        Converts everything into a single dataframe.
        """
        return pandas.concat(self, axis=0)

    def iterrows(self):
        """
        See :epkg:`pandas:DataFrame:iterrows`.
        """
        for df in self:
            for it in df.iterrows():
                yield it

    def head(self, n=5):
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
        else:
            return pandas.concat(st, axis=0)

    def tail(self, n=5):
        """
        Returns the last rows as a DataFrame.
        The size of chunks must be greater than ``n`` to
        get ``n`` lines. This method is not efficient
        because the whole dataset must be walked through.
        """
        for df in self:
            h = df.tail(n=n)
        return h

    def where(self, *args, **kwargs):
        """
        Applies :epkg:`pandas:DataFrame:where`.
        *inplace* must be False.
        This function returns a @see cl StreamingDataFrame.
        """
        kwargs['inplace'] = False
        return StreamingDataFrame(lambda: map(lambda df: df.where(*args, **kwargs), self))

    def sample(self, **kwargs):
        """
        See :epkg:`pandas:DataFrame:sample`.
        Only *frac* is available, otherwise choose
        @see me reservoir_sampling.
        This function returns a @see cl StreamingDataFrame.
        """
        if 'n' in kwargs:
            raise ValueError('Only frac is implemented.')
        return StreamingDataFrame(lambda: map(lambda df: df.sample(**kwargs), self))

    def apply(self, *args, **kwargs):
        """
        Applies :epkg:`pandas:DataFrame:apply`.
        This function returns a @see cl StreamingDataFrame.
        """
        return StreamingDataFrame(lambda: map(lambda df: df.apply(*args, **kwargs), self))

    def applymap(self, *args, **kwargs):
        """
        Applies :epkg:`pandas:DataFrame:applymap`.
        This function returns a @see cl StreamingDataFrame.
        """
        return StreamingDataFrame(lambda: map(lambda df: df.applymap(*args, **kwargs), self))

    def train_test_split(self, path_or_buf=None, export_method="to_csv",
                         names=None, **kwargs):
        """
        Randomly splits a dataframe into smaller pieces.
        The function returns streams of file names.
        The function relies on :epkg:`sklearn:model_selection:train_test_split`.

        @param  partitions      splitting partitions
        @param  path_or_bug     a string, a list of strings or buffers, if it is a
                                string, it must contain ``{}`` like ``partition{}.txt``
        @param  export_method   method used to store the partitions, by default
                                :epkg:`pandas:DataFrame:to_csv`
        @param  names           partitions names, by default ``('train', 'test')``
        @param  kwargs          parameters for the export function and
                                :epkg:`sklearn:model_selection:train_test_split`.
        @return                 outputs of the exports functions

        The function cannot return two iterators or two
        @see cl StreamingDataFrame because running through one
        means running through the other. We can assume both
        splits do not hold in memory and we cannot run through
        the same iterator again as random draws would be different.
        We need to store the results into files or buffers.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ImportWarning)
            from sklearn.model_selection import train_test_split

        opts = ['test_size', 'train_size',
                'random_state', 'shuffle', 'stratify']
        split_ops = {}
        for o in opts:
            if o in kwargs:
                split_ops[o] = kwargs[o]
                del kwargs[o]

        exportf = getattr(pandas.DataFrame, export_method)

        if isinstance(path_or_buf, str):
            if "{}" not in path_or_buf:
                raise ValueError(
                    "path_or_buf must contain {} to insert the partition name")
            if names is None:
                names = ['train', 'test']
            elif len(names) != len(path_or_buf):
                raise ValueError(
                    'names and path_or_buf must have the same length')
            path_or_buf = [path_or_buf.format(n) for n in names]
        elif path_or_buf is None:
            path_or_buf = [None, None]
        else:
            if not isinstance(path_or_buf, list):
                raise TypeError('path_or_buf must be a list or a string')

        bufs = []
        close = []
        for p in path_or_buf:
            if p is None:
                st = StringIO()
                cl = False
            elif isinstance(p, str):
                st = open(p, "w", encoding=kwargs.get('encoding'))
                cl = True
            else:
                st = p
                cl = False
            bufs.append(st)
            close.append(cl)

        for df in self:
            train, test = train_test_split(df, **split_ops)
            exportf(train, bufs[0], **kwargs)
            exportf(test, bufs[1], **kwargs)
            kwargs['header'] = False

        for b, c in zip(bufs, close):
            if c:
                b.close()
        return [st.getvalue() if isinstance(st, StringIO) else p for st, p in zip(bufs, path_or_buf)]