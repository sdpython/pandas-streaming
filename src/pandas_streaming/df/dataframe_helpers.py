# -*- coding: utf-8 -*-
"""
@file
@brief Helpers for dataframes.
"""
import hashlib
import struct
import pandas
import numpy


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
        else:
            raise ValueError("numpy.nan expected, not {0}".format(c))
    else:
        m = hashlib.sha256()
        m.update(c.encode("utf-8"))
        r = m.hexdigest()
        if len(r) >= hash_length:
            return r[:hash_length]
        else:
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
        return hash_int(c, hash_length)

    def hash_strl(c):
        return hash_str(c, hash_length)

    def hash_floatl(c):
        return hash_float(c, hash_length)

    coltype = {n: t for n, t in zip(df.columns, df.dtypes)}
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
    dfj = pandas.DataFrame(rows)
    res = df.merge(dfj, on=[col, temp_col])
    return res.drop(temp_col, axis=1).copy()


def dataframe_shuffle(df, random_state=None):
    """
    Shuffles a dataframe.

    @param      df              :epkg:`pandas:DataFrame`
    @param      random_state    seed
    @return                     new :epkg:`pandas:DataFrame`

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
