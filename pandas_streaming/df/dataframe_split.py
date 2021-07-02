# -*- coding: utf-8 -*-
"""
@file
@brief Implements different methods to split a dataframe.
"""
import hashlib
import pickle
import random
import warnings
from io import StringIO
import pandas


def sklearn_train_test_split(self, path_or_buf=None, export_method="to_csv",
                             names=None, **kwargs):
    """
    Randomly splits a dataframe into smaller pieces.
    The function returns streams of file names.
    The function relies on :epkg:`sklearn:model_selection:train_test_split`.
    It does not handle stratified version of it.

    @param  self            @see cl StreamingDataFrame
    @param  path_or_buf     a string, a list of strings or buffers, if it is a
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

    .. warning::
        The method *export_method* must write the data in
        mode *append* and allows stream.
    """
    if kwargs.get("stratify") is not None:
        raise NotImplementedError(
            "No implementation yet for the stratified version.")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ImportWarning)
        from sklearn.model_selection import train_test_split  # pylint: disable=C0415

    opts = ['test_size', 'train_size',
            'random_state', 'shuffle', 'stratify']
    split_ops = {}
    for o in opts:
        if o in kwargs:
            split_ops[o] = kwargs[o]
            del kwargs[o]

    exportf_ = getattr(pandas.DataFrame, export_method)
    if export_method == 'to_csv' and 'mode' not in kwargs:
        exportf = lambda *a, **kw: exportf_(*a, mode='a', **kw)
    else:
        exportf = exportf_

    if isinstance(path_or_buf, str):
        if "{}" not in path_or_buf:
            raise ValueError(
                "path_or_buf must contain {} to insert the partition name")
        if names is None:
            names = ['train', 'test']
        elif len(names) != len(path_or_buf):
            raise ValueError(  # pragma: no cover
                'names and path_or_buf must have the same length')
        path_or_buf = [path_or_buf.format(n) for n in names]
    elif path_or_buf is None:
        path_or_buf = [None, None]
    else:
        if not isinstance(path_or_buf, list):
            raise TypeError(  # pragma: no cover
                'path_or_buf must be a list or a string')

    bufs = []
    close = []
    for p in path_or_buf:
        if p is None:
            st = StringIO()
            cl = False
        elif isinstance(p, str):
            st = open(  # pylint: disable=R1732
                p, "w", encoding=kwargs.get('encoding'))
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
    return [st.getvalue() if isinstance(st, StringIO) else p
            for st, p in zip(bufs, path_or_buf)]


def sklearn_train_test_split_streaming(self, test_size=0.25, train_size=None,
                                       stratify=None, hash_size=9, unique_rows=False):
    """
    Randomly splits a dataframe into smaller pieces.
    The function returns streams of file names.
    The function relies on :epkg:`sklearn:model_selection:train_test_split`.
    It handles the stratified version of it.

    @param  self            @see cl StreamingDataFrame
    @param  test_size       ratio for the test partition (if *train_size* is not specified)
    @param  train_size      ratio for the train partition
    @param  stratify        column holding the stratification
    @param  hash_size       size of the hash to cache information about partition
    @param  unique_rows     ensures that rows are unique
    @return                 Two @see cl StreamingDataFrame, one
                            for train, one for test.

    The function returns two iterators or two
    @see cl StreamingDataFrame. It
    tries to do everything without writing anything on disk
    but it requires to store the repartition somehow.
    This function hashes every row and maps the hash with a part
    (train or test). This cache must hold in memory otherwise the
    function fails. The two returned iterators must not be used
    for the first time in the same time. The first time is used to
    build the cache. The function changes the order of rows if
    the parameter *stratify* is not null. The cache has a side effect:
    every exact same row will be put in the same partition.
    If that is not what you want, you should add an index column
    or a random one.
    """
    p = (1 - test_size) if test_size else None
    if train_size is not None:
        p = train_size
    n = 2 * max(1 / p, 1 / (1 - p))  # changement

    static_schema = []

    def iterator_rows():
        "iterates on rows"
        counts = {}
        memory = {}
        pos_col = None
        for df in self:
            if pos_col is None:
                static_schema.append(list(df.columns))
                static_schema.append(list(df.dtypes))
                static_schema.append(df.shape[0])
                if stratify is not None:
                    pos_col = list(df.columns).index(stratify)
                else:
                    pos_col = -1

            for obs in df.itertuples(index=False, name=None):
                strat = 0 if stratify is None else obs[pos_col]
                if strat not in memory:
                    memory[strat] = []
                memory[strat].append(obs)

                for k, v in memory.items():
                    if len(v) >= n + random.randint(0, 10):  # changement
                        vr = list(range(len(v)))
                        # on permute aléatoirement
                        random.shuffle(vr)
                        if (0, k) in counts:
                            tt = counts[1, k] + counts[0, k]
                            delta = - int(counts[0, k] - tt * p + 0.5)
                        else:
                            delta = 0
                        i = int(len(v) * p + 0.5)
                        i += delta
                        i = max(0, min(len(v), i))
                        one = set(vr[:i])
                        for d, obs_ in enumerate(v):
                            yield obs_, 0 if d in one else 1
                        if (0, k) not in counts:
                            counts[0, k] = i
                            counts[1, k] = len(v) - i
                        else:
                            counts[0, k] += i
                            counts[1, k] += len(v) - i
                        # on efface de la mémoire les informations produites
                        v.clear()

        # Lorsqu'on a fini, il faut tout de même répartir les
        # observations stockées.
        for k, v in memory.items():
            vr = list(range(len(v)))
            # on permute aléatoirement
            random.shuffle(vr)
            if (0, k) in counts:
                tt = counts[1, k] + counts[0, k]
                delta = - int(counts[0, k] - tt * p + 0.5)
            else:
                delta = 0
            i = int(len(v) * p + 0.5)
            i += delta
            i = max(0, min(len(v), i))
            one = set(vr[:i])
            for d, obs in enumerate(v):
                yield obs, 0 if d in one else 1
            if (0, k) not in counts:
                counts[0, k] = i
                counts[1, k] = len(v) - i
            else:
                counts[0, k] += i
                counts[1, k] += len(v) - i

    def h11(w):
        "pickle and hash"
        b = pickle.dumps(w)
        return hashlib.md5(b).hexdigest()[:hash_size]

    # We store the repartition in a cache.
    cache = {}

    def iterator_internal(part_requested):
        "internal iterator on dataframes"
        iy = 0
        accumul = []
        if len(cache) == 0:
            for obs, part in iterator_rows():
                h = h11(obs)
                if unique_rows and h in cache:
                    raise ValueError(  # pragma: no cover
                        "A row or at least its hash is already cached. "
                        "Increase hash_size or check for duplicates "
                        "('{0}')\n{1}.".format(h, obs))
                if h not in cache:
                    cache[h] = part
                else:
                    part = cache[h]
                if part == part_requested:
                    accumul.append(obs)
                    if len(accumul) >= static_schema[2]:
                        dfo = pandas.DataFrame(
                            accumul, columns=static_schema[0])
                        self.ensure_dtype(dfo, static_schema[1])
                        iy += dfo.shape[0]
                        accumul.clear()
                        yield dfo
        else:
            for df in self:
                for obs in df.itertuples(index=False, name=None):
                    h = h11(obs)
                    part = cache.get(h)
                    if part is None:
                        raise ValueError(  # pragma: no cover
                            "Second iteration. A row was never met in the first one\n{0}".format(obs))
                    if part == part_requested:
                        accumul.append(obs)
                        if len(accumul) >= static_schema[2]:
                            dfo = pandas.DataFrame(
                                accumul, columns=static_schema[0])
                            self.ensure_dtype(dfo, static_schema[1])
                            iy += dfo.shape[0]
                            accumul.clear()
                            yield dfo
        if len(accumul) > 0:
            dfo = pandas.DataFrame(accumul, columns=static_schema[0])
            self.ensure_dtype(dfo, static_schema[1])
            iy += dfo.shape[0]
            yield dfo

    return (self.__class__(lambda: iterator_internal(0)),
            self.__class__(lambda: iterator_internal(1)))
