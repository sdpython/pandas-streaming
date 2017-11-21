#-*- coding: utf-8 -*-
"""
@file
@brief Implements a connex split between train and test.
"""
import pandas
import random
import numpy
from collections import Counter
from sklearn.model_selection import train_test_split


class ImbalancedSplitException(Exception):
    """
    Raised when an imbalanced split is detected.
    """
    pass


def dataframe_shuffle(df, seed=None):
    """
    Shuffles a dataframe.

    @param      df      :epkg:`pandas:DataFrame`
    @param      seed    seed
    @return             new :epkg:`pandas:DataFrame`
    """
    if seed is not None:
        random.seed(seed)
    ori_cols = list(df.columns)
    scols = set(ori_cols)

    no_index = df.reset_index(drop=False)
    keep_cols = [_ for _ in no_index.columns if _ not in scols]
    index = list(no_index.index)
    random.shuffle(index)
    shuffled = no_index.iloc[index, :]
    res = shuffled.set_index(keep_cols)[ori_cols]
    res.index.names = df.index.names
    return res


def train_test_split_weights(df, weights=None, test_size=0.25, train_size=None,
                             shuffle=True, fail_imbalanced=0.05):
    """
    Splits a database in train/test given, every row
    can have a different weight.

    @param  df              :epkg:`pandas:DataFrame` or @see cl StreamingDataFrame
    @param  weights         None or weights or weights column name
    @param  test_size       ratio for the test partition (if *train_size* is not specified)
    @param  train_size      ratio for the train partition
    @param  shuffle         shuffles before the split
    @param  fail_imbalanced raises an exception if relative weights difference is higher than this value
    @return                 train and test :epkg:`pandas:DataFrame`

    If the dataframe is not shuffled first, the function
    will produce two datasets which are unlikely to be randomized
    as the function tries to keep equal weights among both paths
    without using randomness.
    """
    if hasattr(df, 'iter_creation'):
        raise NotImplementedError(
            'Not implemented yet for StreamingDataFrame.')
    if isinstance(df, numpy.ndarray):
        raise NotImplementedError("Not implemented on numpy arrays.")
    if shuffle:
        df = dataframe_shuffle(df)
    if weights is None:
        return train_test_split(df, test_size=test_size, train_size=train_size)

    if isinstance(weights, pandas.Series):
        weights = list(weights)
    elif isinstance(weights, str):
        weights = list(df[weights])
    if len(weights) != df.shape[0]:
        raise ValueError("Dimension mismatch between weights and dataframe {0} != {1}".format(
            df.shape[0], len(weights)))

    p = (1 - test_size) if test_size else None
    if train_size is not None:
        p = train_size
    test_size = 1 - p
    if min(test_size, p) <= 0:
        raise ValueError(
            "test_size={0} or train_size={1} cannot be null".format(test_size, train_size))
    ratio = test_size / p

    balance = 0
    train_ids = []
    test_ids = []
    test_weights = 0
    train_weights = 0
    for i in range(0, df.shape[0]):
        w = weights[i]
        if balance == 0:
            h = random.randint(0, 1)
            totest = h == 0
        else:
            totest = balance < 0
        if totest:
            test_ids.append(i)
            balance += w
            test_weights += w
        else:
            train_ids.append(i)
            balance -= w * ratio
            train_weights += w * ratio

    r = abs(train_weights - test_weights) / \
        (1.0 * (train_weights + test_weights))
    if r >= fail_imbalanced:
        raise ImbalancedSplitException(
            "Split is imbalanced: train_weights={0} test_weights={1} r={2}".format(train_weights, test_weights, r))

    return df.iloc[train_ids, :], df.iloc[test_ids, :]


def train_test_connex_split(df, groups, test_size=0.25, train_size=None,
                            stratify=None, hash_size=9, unique_rows=False,
                            shuffle=True, fail_imbalanced=0.05, keep_balance=None,
                            stop_if_bigger=None, return_cnx=False, fLOG=None):
    """
    This split is for a specific case where data is linked
    in many ways. Let's assume we have three ids as we have
    for online sales: *(product id, user id, card id)*.
    As we may need to compute aggregated features,
    we need every id not to be present in both train and
    test set.

    @param  df              :epkg:`pandas:DataFrame`
    @param  groups          columns name for the ids
    @param  test_size       ratio for the test partition (if *train_size* is not specified)
    @param  train_size      ratio for the train partition
    @param  stratify        column holding the stratification
    @param  hash_size       size of the hash to cache information about partition
    @param  unique_rows     ensures that rows are unique
    @param  shuffle         shuffles before the split
    @param  fail_imbalanced raises an exception if relative weights difference is higher than this value
    @param  stop_if_bigger  (float) stops a connected components from being
                            bigger than this ratio of elements, this should not be used
                            unless a big components emerges, the algorithm stops merging
                            but does not guarantee it returns the best cut,
                            the value should be close to 0
    @param  keep_balance    (float), if not None, does not merge connected components
                            if their relative sizes are too different, the value should be
                            close to 1
    @param  return_cnx      returns connected components as a third results
    @param  fLOG            logging function
    @return                 Two @see cl StreamingDataFrame, one
                            for train, one for test.

    The list of ids must hold in memory.
    There is no streaming implementation for the ids.

    .. exref::
        :title: Split a dataframe, keep ids in separate partitions

        In some data science problems, rows are not independant
        and share common value, most of the time ids. In some
        specific case, multiple ids from different columns are
        connected and must appear in the same partition.
        Testing that each id column is evenly split and do not
        appear in both sets in not enough. Connected components
        are needed.

        .. runpython::
            :showcode:

            from pandas import DataFrame
            from pandas_streaming.df import train_test_connex_split

            df = DataFrame([dict(user="UA", prod="PAA", card="C1"),
                            dict(user="UA", prod="PB", card="C1"),
                            dict(user="UB", prod="PC", card="C2"),
                            dict(user="UB", prod="PD", card="C2"),
                            dict(user="UC", prod="PAA", card="C3"),
                            dict(user="UC", prod="PF", card="C4"),
                            dict(user="UD", prod="PG", card="C5"),
                            ])

            train, test = train_test_connex_split(df, test_size=0.5,
                                                  groups=['user', 'prod', 'card'],
                                                  fail_imbalanced=0.4)
            print(train)
            print(test)

    If *return_cnx* is True, the third results contains:

    * connected components for each id
    * the dataframe with connected components as a new column

    .. runpython::
        :showcode:

        from pandas import DataFrame
        from pandas_streaming.df import train_test_connex_split

        df = DataFrame([dict(user="UA", prod="PAA", card="C1"),
                        dict(user="UA", prod="PB", card="C1"),
                        dict(user="UB", prod="PC", card="C2"),
                        dict(user="UB", prod="PD", card="C2"),
                        dict(user="UC", prod="PAA", card="C3"),
                        dict(user="UC", prod="PF", card="C4"),
                        dict(user="UD", prod="PG", card="C5"),
                        ])

        train, test, cnx = train_test_connex_split(df, test_size=0.5,
                                              groups=['user', 'prod', 'card'],
                                              fail_imbalanced=0.4, return_cnx=True)

        print(cnx[0])
        print(cnx[1])
    """
    if groups is None or len(groups) == 0:
        raise ValueError("groups is empty. Use regular train_test_split.")
    if hasattr(df, 'iter_creation'):
        raise NotImplementedError(
            'Not implemented yet for StreamingDataFrame.')
    if isinstance(df, numpy.ndarray):
        raise NotImplementedError("Not implemented on numpy arrays.")
    if shuffle:
        df = dataframe_shuffle(df)

    dfids = df[groups].copy()

    name = "connex"
    while name in dfids.columns:
        name += "_"
    one = "weight"
    while one in dfids.columns:
        one += "_"

    # Connected components.
    elements = list(range(dfids.shape[0]))
    counts_cnx = {i: {i} for i in elements}
    connex = {}
    avoids_merge = {}
    modif = 1
    iter = 0

    while modif > 0 and iter < len(elements):
        if fLOG and df.shape[0] > 10000:
            fLOG("[train_test_connex_split] iteration={0}-#nb connect={1} - modif={2}".format(
                iter, len(set(elements)), modif))
        modif = 0
        iter += 1
        for i, row in enumerate(dfids.itertuples(index=False, name=None)):
            vals = [val for val in zip(groups, row) if not isinstance(
                val[1], float) or not numpy.isnan(val[1])]

            c = elements[i]
            if i not in counts_cnx[c]:
                raise RuntimeError("Element not found in its own component: {0}\n{1}".format(
                    i, list(sorted(counts_cnx[c]))))

            for val in vals:
                if val not in connex:
                    connex[val] = c
                    modif += 1

            set_c = set(connex[val] for val in vals)
            set_c.add(c)
            new_c = min(set_c)

            add_pair_c = []
            for c in set_c:
                if c == new_c or (new_c, c) in avoids_merge:
                    continue
                if keep_balance is not None:
                    maxi = min(len(counts_cnx[new_c]), len(counts_cnx[c]))
                    if maxi > 5:
                        diff = len(counts_cnx[new_c]) + \
                            len(counts_cnx[c]) - maxi
                        r = diff / float(maxi)
                        if r > keep_balance:
                            if fLOG:
                                fLOG('[train_test_connex_split]    balance r={0:0.00000}>{1:0.00}, #[{2}]={3}, #[{4}]={5}'.format(
                                    r, keep_balance, new_c, len(counts_cnx[new_c]), c, len(counts_cnx[c])))
                            continue

                if stop_if_bigger is not None:
                    r = (len(counts_cnx[new_c]) +
                         len(counts_cnx[c])) / float(len(elements))
                    if r > stop_if_bigger:
                        if fLOG:
                            fLOG('[train_test_connex_split]    no merge r={0:0.00000}>{1:0.00}, #[{2}]={3}, #[{4}]={5}'.format(
                                r, stop_if_bigger, new_c, len(counts_cnx[new_c]), c, len(counts_cnx[c])))
                        avoids_merge[new_c, c] = i
                        continue

                add_pair_c.append(c)

            if len(add_pair_c) > 0:
                for c in add_pair_c:
                    modif += len(counts_cnx[c])
                    for i in counts_cnx[c]:
                        elements[i] = new_c
                    counts_cnx[new_c] = counts_cnx[new_c].union(counts_cnx[c])
                    counts_cnx[c] = set()

                    keys = list(vals)
                    for val in keys:
                        if connex[val] == c:
                            connex[val] = new_c
                            modif += 1

    # final
    dfids[name] = elements
    dfids[one] = 1
    grsum = dfids[[name, one]].groupby(name, as_index=False).sum()
    if fLOG:
        for g in groups:
            fLOG("[train_test_connex_split]     #nb in '{0}': {1}".format(
                g, len(set(dfids[g]))))
        fLOG(
            "[train_test_connex_split] #connex {0}/{1}".format(grsum.shape[0], dfids.shape[0]))
    if grsum.shape[0] <= 1:
        raise ValueError("Every element is in the same connected components.")

    # Statistics: top connected components
    if fLOG:
        # Global statistics
        counts = Counter(elements)
        cl = [(v, k) for k, v in counts.items()]
        cum = 0
        maxc = None
        fLOG("[train_test_connex_split] number of connected components: {0}".format(
            len(set(elements))))
        for i, (v, k) in enumerate(sorted(cl, reverse=True)):
            if i == 0:
                maxc = k, v
            if i >= 10:
                break
            cum += v
            fLOG("[train_test_connex_split]     c={0} #elements={1} cumulated={2}/{3}".format(
                k, v, cum, len(elements)))

        # Most important component
        fLOG(
            '[train_test_connex_split] first row of the biggest component {0}'.format(maxc))
        tdf = dfids[dfids[name] == maxc[0]]
        fLOG('[train_test_connex_split] \n{0}'.format(tdf.head(n=10)))

    # Splits.
    train, test = train_test_split_weights(grsum, weights=one, test_size=test_size,
                                           train_size=train_size, shuffle=shuffle,
                                           fail_imbalanced=fail_imbalanced)
    train.drop(one, inplace=True, axis=1)
    test.drop(one, inplace=True, axis=1)

    # We compute the final dataframe.
    def double_merge(d):
        merge1 = dfids.merge(d, left_on=name, right_on=name)
        merge2 = df.merge(merge1, left_on=groups, right_on=groups)
        return merge2

    train_f = double_merge(train)
    test_f = double_merge(test)
    if return_cnx:
        return train_f, test_f, (connex, dfids)
    else:
        return train_f, test_f
