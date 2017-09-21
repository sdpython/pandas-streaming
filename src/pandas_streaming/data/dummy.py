#-*- coding: utf-8 -*-
"""
@file
@brief Dummy datasets.
"""
from pandas import DataFrame
from ..df import StreamingDataFrame


def dummy_streaming_dataframe(n, chunk_size=10):
    """
    Returns a dummy streaming dataframe
    mostly for unit test purposes.

    @param      n           number of rows
    @param      chunk_size  chunk size
    @return                 a @see cl StreamingDataFrame
    """
    df = DataFrame(dict(cint=list(range(0, n)), cstr=[
                   "s{0}".format(i) for i in range(0, n)]))
    return StreamingDataFrame.read_df(df, chunk_size=chunk_size)
