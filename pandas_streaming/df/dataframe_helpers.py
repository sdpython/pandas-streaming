# -*- coding: utf-8 -*-
"""
@file
@brief Helpers for dataframes.
"""
import hashlib
import struct
import warnings
import numpy
from pandas import DataFrame, Index


def numpy_types():
    """
    Returns the list of :epkg:`numpy` available types.

    :return: list of types
    """

    return [numpy.bool_,
            numpy.int_,
            numpy.intc,
            numpy.intp,
            numpy.int8,
            numpy.int16,
            numpy.int32,
            numpy.int64,
            numpy.uint8,
            numpy.uint16,
            numpy.uint32,
            numpy.uint64,
            numpy.float_,
            numpy.float16,
            numpy.float32,
            numpy.float64,
            numpy.complex_,
            numpy.complex64,
            numpy.complex128]


def hash_str(c, hash_length):
    """
    Hashes a string.

    @param      c               value to hash
    @param      hash_length     hash_length
    @return                     string
    """
    if isinstance(c, float):
        if numpy.isnan(c):
            return c
        raise ValueError("numpy.nan expected, not {0}".format(c))
    m = hashlib.sha256()
    m.update(c.encode("utf-8"))
    r = m.hexdigest()
    if len(r) >= hash_length:
        return r[:hash_length]
    return r


def hash_int(c, hash_length):
    """
    Hashes an integer into an integer.

    @param      c               value to hash
    @param      hash_length     hash_length
    @return                     int
    """
    if isinstance(c, float):
        if numpy.isnan(c):
            return c
        else:
            raise ValueError("numpy.nan expected, not {0}".format(c))
    else:
        b = struct.pack("i", c)
        m = hashlib.sha256()
        m.update(b)
        r = m.hexdigest()
        if len(r) >= hash_length:
            r = r[:hash_length]
        return int(r, 16) % (10 ** 8)


def hash_float(c, hash_length):
    """
    Hashes a float into a float.

    @param      c               value to hash
    @param      hash_length     hash_length
    @return                     int
    """
    if numpy.isnan(c):
        return c
    else:
        b = struct.pack("d", c)
        m = hashlib.sha256()
        m.update(b)
        r = m.hexdigest()
        if len(r) >= hash_length:
            r = r[:hash_length]
        i = int(r, 16) % (2 ** 53)
        return float(i)


def dataframe_hash_columns(df, cols=None, hash_length=10, inplace=False):
    """
    Hashes a set of columns in a dataframe.
    Keeps the same type. Skips missing values.

    @param      df          dataframe
    @param      cols        columns to hash or None for alls.
    @param      hash_length for strings only, length of the hash
    @param      inplace     modifies inplace
    @return                 new dataframe

    This might be useful to anonimized data before
    making it public.

    .. exref::
        :title: Hashes a set of columns in a dataframe
        :tag: dataframe

        .. runpython::
            :showcode:

            import pandas
            from pandas_streaming.df import dataframe_hash_columns
            df = pandas.DataFrame([dict(a=1, b="e", c=5.6, ind="a1", ai=1),
                                   dict(b="f", c=5.7, ind="a2", ai=2),
                                   dict(a=4, b="g", ind="a3", ai=3),
                                   dict(a=8, b="h", c=5.9, ai=4),
                                   dict(a=16, b="i", c=6.2, ind="a5", ai=5)])
            print(df)
            print('--------------')
            df2 = dataframe_hash_columns(df)
            print(df2)
    """
    if cols is None:
        cols = list(df.columns)

    if not inplace:
        df = df.copy()

    def hash_intl(c):
        "hash int"
        return hash_int(c, hash_length)

    def hash_strl(c):
        "hash string"
        return hash_str(c, hash_length)

    def hash_floatl(c):
        "hash float"
        return hash_float(c, hash_length)

    coltype = {n: t for n, t in zip(  # pylint: disable=R1721
        df.columns, df.dtypes)}  # pylint: disable=R1721
    for c in cols:
        t = coltype[c]
        if t == int:
            df[c] = df[c].apply(hash_intl)
        elif t == numpy.int64:
            df[c] = df[c].apply(lambda x: numpy.int64(hash_intl(x)))
        elif t == float:
            df[c] = df[c].apply(hash_floatl)
        elif t == object:
            df[c] = df[c].apply(hash_strl)
        else:
            raise NotImplementedError(
                "Conversion of type {0} in column '{1}' is not implemented".format(t, c))

    return df


def dataframe_unfold(df, col, new_col=None, sep=","):
    """
    One column may contain concatenated values.
    This function splits these values and multiplies the
    rows for each split value.

    @param      df      dataframe
    @param      col     column with the concatenated values (strings)
    @param      new_col new column name, if None, use default value.
    @param      sep     separator
    @return             a new dataframe

    .. exref::
        :title: Unfolds a column of a dataframe.
        :tag: dataframe

        .. runpython::
            :showcode:

            import pandas
            import numpy
            from pandas_streaming.df import dataframe_unfold

            df = pandas.DataFrame([dict(a=1, b="e,f"),
                                   dict(a=2, b="g"),
                                   dict(a=3)])
            print(df)
            df2 = dataframe_unfold(df, "b")
            print('----------')
            print(df2)

            # To fold:
            folded = df2.groupby('a').apply(lambda row: ','.join(row['b_unfold'].dropna()) \\
                                            if len(row['b_unfold'].dropna()) > 0 else numpy.nan)
            print('----------')
            print(folded)
    """
    if new_col is None:
        col_name = col + "_unfold"
    else:
        col_name = new_col
    temp_col = '__index__'
    while temp_col in df.columns:
        temp_col += "_"
    rows = []
    for i, v in enumerate(df[col]):
        if isinstance(v, str):
            spl = v.split(sep)
            for vs in spl:
                rows.append({col: v, col_name: vs, temp_col: i})
        else:
            rows.append({col: v, col_name: v, temp_col: i})
    df = df.copy()
    df[temp_col] = list(range(df.shape[0]))
    dfj = DataFrame(rows)
    res = df.merge(dfj, on=[col, temp_col])
    return res.drop(temp_col, axis=1).copy()


def dataframe_shuffle(df, random_state=None):
    """
    Shuffles a dataframe.

    :param df: :epkg:`pandas:DataFrame`
    :param random_state: seed
    :return: new :epkg:`pandas:DataFrame`

    .. exref::
        :title: Shuffles the rows of a dataframe
        :tag: dataframe

        .. runpython::
            :showcode:

            import pandas
            from pandas_streaming.df import dataframe_shuffle

            df = pandas.DataFrame([dict(a=1, b="e", c=5.6, ind="a1"),
                                   dict(a=2, b="f", c=5.7, ind="a2"),
                                   dict(a=4, b="g", c=5.8, ind="a3"),
                                   dict(a=8, b="h", c=5.9, ind="a4"),
                                   dict(a=16, b="i", c=6.2, ind="a5")])
            print(df)
            print('----------')

            shuffled = dataframe_shuffle(df, random_state=0)
            print(shuffled)
    """
    if random_state is not None:
        state = numpy.random.RandomState(random_state)
        permutation = state.permutation
    else:
        permutation = numpy.random.permutation
    ori_cols = list(df.columns)
    scols = set(ori_cols)

    no_index = df.reset_index(drop=False)
    keep_cols = [_ for _ in no_index.columns if _ not in scols]
    index = no_index.index
    index = permutation(index)
    shuffled = no_index.iloc[index, :]
    res = shuffled.set_index(keep_cols)[ori_cols]
    res.index.names = df.index.names
    return res


def pandas_fillna(df, by, hasna=None, suffix=None):
    """
    Replaces the :epkg:`nan` values for something not :epkg:`nan`.
    Mostly used by @see fn pandas_groupby_nan.

    :param df: dataframe
    :param by: list of columns for which we need to replace nan
    :param hasna: None or list of columns for which we need to replace NaN
    :param suffix: use a prefix for the NaN value
    :return: list of values chosen for each column, new dataframe (new copy)
    """
    suffix = suffix if suffix else "²nan"
    df = df.copy()
    rep = {}
    for c in by:
        if hasna is not None and c not in hasna:
            continue
        if df[c].dtype in (str, bytes, object):
            se = set(df[c].dropna())
            val = se.pop()
            if isinstance(val, str):
                cst = suffix
                val = ""
            elif isinstance(val, bytes):
                cst = b"_"
            else:
                raise TypeError(  # pragma: no cover
                    "Unable to determine a constant for type='{0}' dtype='{1}'".format(
                        val, df[c].dtype))
            val += cst
            while val in se:
                val += suffix
            df[c].fillna(val, inplace=True)
            rep[c] = val
        else:
            dr = df[c].dropna()
            mi = abs(dr.min())
            ma = abs(dr.max())
            val = ma + mi
            if val == ma and not isinstance(val, str):
                val += ma + 1.
            if val <= ma:
                raise ValueError(  # pragma: no cover
                    "Unable to find a different value for column '{}' v='{}: "
                    "min={} max={}".format(c, val, mi, ma))
            df[c].fillna(val, inplace=True)
            rep[c] = val
    return rep, df


def pandas_groupby_nan(df, by, axis=0, as_index=False, suffix=None, nanback=True, **kwargs):
    """
    Does a *groupby* including keeping missing values (:epkg:`nan`).

    :param df: dataframe
    :param by: column or list of columns
    :param axis: only 0 is allowed
    :param as_index: should be False
    :param suffix: None or a string
    :param nanback: put :epkg:`nan` back in the index,
        otherwise it leaves a replacement for :epkg:`nan`.
        (does not work when grouping by multiple columns)
    :param kwargs: other parameters sent to
        `groupby <http://pandas.pydata.org/pandas-docs/stable/
        generated/pandas.DataFrame.groupby.html>`_
    :return: groupby results

    See `groupby and missing values <http://pandas-docs.github.io/
    pandas-docs-travis/groupby.html#na-and-nat-group-handling>`_.
    If no :epkg:`nan` is detected, the function falls back in regular
    :epkg:`pandas:DataFrame:groupby` which has the following
    behavior.

    .. exref::
        :title: Group a dataframe by one column including nan values
        :tag: dataframe

        The regular :epkg:`pandas:dataframe:GroupBy` of a
        :epkg:`pandas:DataFrame` removes every :epkg:`nan`
        values from the index.

        .. runpython::
            :showcode:

            from pandas import DataFrame

            data = [dict(a=2, ind="a", n=1),
                    dict(a=2, ind="a"),
                    dict(a=3, ind="b"),
                    dict(a=30)]
            df = DataFrame(data)
            print(df)
            gr = df.groupby(["ind"]).sum()
            print(gr)

        Function @see fn pandas_groupby_nan modifies keeps them.

        .. runpython::
            :showcode:

            from pandas import DataFrame
            from pandas_streaming.df import pandas_groupby_nan

            data = [dict(a=2, ind="a", n=1),
                    dict(a=2, ind="a"),
                    dict(a=3, ind="b"),
                    dict(a=30)]
            df = DataFrame(data)
            gr2 = pandas_groupby_nan(df, ["ind"]).sum()
            print(gr2)
    """
    if axis != 0:
        raise NotImplementedError("axis should be 0")
    if as_index:
        raise NotImplementedError("as_index must be False")
    if isinstance(by, tuple):
        raise TypeError("by should be of list not tuple")
    if not isinstance(by, list):
        by = [by]
    hasna = {}
    for b in by:
        h = df[b].isnull().values.any()
        if h:
            hasna[b] = True
    if len(hasna) > 0:
        rep, df_copy = pandas_fillna(df, by, hasna, suffix=suffix)
        res = df_copy.groupby(by, axis=axis, as_index=as_index, **kwargs)
        if len(by) == 1:
            if not nanback:
                dummy = DataFrame([{"a": "a"}])
                do = dummy.dtypes[0]
                typ = {c: t for c, t in zip(  # pylint: disable=R1721
                    df.columns, df.dtypes)}  # pylint: disable=R1721
                if typ[by[0]] != do:
                    warnings.warn(  # pragma: no cover
                        "[pandas_groupby_nan] NaN value: {0}".format(rep))
                return res
            for b in by:
                fnan = rep[b]
                if fnan in res.grouper.groups:
                    res.grouper.groups[numpy.nan] = res.grouper.groups[fnan]
                    del res.grouper.groups[fnan]
                new_val = list((numpy.nan if b == fnan else b)
                               for b in res.grouper.result_index)
                res.grouper.groupings[0]._group_index = Index(new_val)
                res.grouper.groupings[0].obj[b].replace(
                    fnan, numpy.nan, inplace=True)
                if hasattr(res.grouper, 'grouping'):
                    if isinstance(res.grouper.groupings[0].grouper, numpy.ndarray):
                        arr = numpy.array(new_val)
                        res.grouper.groupings[0].grouper = arr
                        if (hasattr(res.grouper.groupings[0], '_cache') and
                                'result_index' in res.grouper.groupings[0]._cache):
                            del res.grouper.groupings[0]._cache['result_index']
                    else:
                        raise NotImplementedError("Not implemented for type: {0}".format(
                            type(res.grouper.groupings[0].grouper)))
                else:
                    grouper = res.grouper._get_grouper()
                    if isinstance(grouper, numpy.ndarray):
                        arr = numpy.array(new_val)
                        res.grouper.groupings[0].grouping_vector = arr
                        if (hasattr(res.grouper.groupings[0], '_cache') and
                                'result_index' in res.grouper.groupings[0]._cache):
                            index = res.grouper.groupings[0]._cache['result_index']
                            if len(rep) == 1:
                                key = list(rep.values())[0]
                                new_index = numpy.array(index)
                                for i in range(0, len(new_index)):  # pylint: disable=C0200
                                    if new_index[i] == key:
                                        new_index[i] = numpy.nan
                                res.grouper.groupings[0]._cache['result_index'] = (
                                    index.__class__(new_index))
                            else:
                                raise NotImplementedError(
                                    "NaN values not implemented for multiindex.")
                    else:
                        raise NotImplementedError(
                            "Not implemented for type: {0}".format(
                                type(res.grouper.groupings[0].grouper)))
                res.grouper._cache['result_index'] = res.grouper.groupings[0]._group_index
        else:
            if not nanback:
                dummy = DataFrame([{"a": "a"}])
                do = dummy.dtypes[0]
                typ = {c: t for c, t in zip(  # pylint: disable=R1721
                    df.columns, df.dtypes)}  # pylint: disable=R1721
                for b in by:
                    if typ[b] != do:
                        warnings.warn(  # pragma: no cover
                            "[pandas_groupby_nan] NaN values: {0}".format(rep))
                        break
                return res
            raise NotImplementedError(
                "Not yet implemented. Replacing pseudo nan values by real nan "
                "values is not as easy as it looks. Use nanback=False")

            # keys = list(res.grouper.groups.keys())
            # didit = False
            # mapping = {}
            # for key in keys:
            #     new_key = list(key)
            #     mod = False
            #     for k, b in enumerate(by):
            #         if b not in rep:
            #             continue
            #         fnan = rep[b]
            #         if key[k] == fnan:
            #             new_key[k] = numpy.nan
            #             mod = True
            #             didit = True
            #             mapping[fnan] = numpy.nan
            #     if mod:
            #         new_key = tuple(new_key)
            #         mapping[key] = new_key
            #         res.grouper.groups[new_key] = res.grouper.groups[key]
            #         del res.grouper.groups[key]
            # if didit:
            #     # this code deos not work
            #     vnan = numpy.nan
            #     new_index = list(mapping.get(v, v)
            #                      for v in res.grouper.result_index)
            #     names = res.grouper.result_index.names
            #     # index = MultiIndex.from_tuples(tuples=new_index, names=names)
            #     # res.grouper.result_index = index  # does not work cannot set
            #     # values for [result_index]
            #     for k in range(len(res.grouper.groupings)):
            #         grou = res.grouper.groupings[k]
            #         new_val = list(mapping.get(v, v) for v in grou)
            #         grou._group_index = Index(new_val)
            #         b = names[k]
            #         if b in rep:
            #             vv = rep[b]
            #             grou.obj[b].replace(vv, vnan, inplace=True)
            #         if isinstance(grou.grouper, numpy.ndarray):
            #             grou.grouper = numpy.array(new_val)
            #         else:
            #             raise NotImplementedError(
            #                 "Not implemented for type: {0}".format(type(grou.grouper)))
            #     del res.grouper._cache
        return res
    else:
        return df.groupby(by, axis=axis, **kwargs)
