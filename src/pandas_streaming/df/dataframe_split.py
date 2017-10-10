#-*- coding: utf-8 -*-
"""
@file
@brief Implements different methods to split a dataframe.
"""
import warnings
import pandas
from io import StringIO


def sklearn_train_test_split(self, path_or_buf=None, export_method="to_csv",
                             names=None, **kwargs):
    """
    Randomly splits a dataframe into smaller pieces.
    The function returns streams of file names.
    The function relies on :epkg:`sklearn:model_selection:train_test_split`.
    It does not handle stratified version of it.

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
    if kwargs.get("stratify") is not None:
        raise NotImplementedError(
            "No implementation yet for the stratified version.")
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
