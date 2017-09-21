#-*- coding: utf-8 -*-
"""
@file
@brief Defines a streming dataframe.
"""
import pandas
from ..exc import StreamingInefficientException


class StreamingDataFrame:
    """
    Defines a streaming dataframe.
    """

    def __init__(self, iter_creation):
        """
        Wraps a iterator on dataframe.

        @param      iter_creation   code which creates an iterator.

        The constructor cannot receive an iterator otherwise
        this class would be able to walk through the data
        only once. Instead, it takes a function which generates
        an iterator on :epkg:`pandas:DataFrame`.
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
        when a file is large.
        """
        nl, nc = 0, 0
        for it in self:
            nc = max(it.shape[1], nc)
            nl += it.shape[0]
        return nl, nc
