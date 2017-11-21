#-*- coding: utf-8 -*-
"""
@file
@brief Helpers for dataframes.
"""
import hashlib
import struct
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
    Keep the same type. Skips missing values.

    @param      df          dataframe
    @param      cols        columns to hash or None for alls.
    @param      hask_length for strings only, length of the hash
    @param      inplace     modifies inplace
    @return                 new dataframe

    This might be useful to anonimized data before
    making it public.
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
        elif t == float:
            df[c] = df[c].apply(hash_floatl)
        elif t == object:
            df[c] = df[c].apply(hash_strl)
        else:
            raise NotImplementedError(
                "Conversion of type {0} in column '{1}' is not implemented".format(t, c))

    return df
