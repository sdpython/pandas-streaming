#-*- coding: utf-8 -*-
"""
@file
@brief Implements a connex split between train and test.
"""
import pandas
import random
import numpy
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
                            shuffle=True, fail_imbalanced=0.05, fLOG=None):
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
    @param  fLOG            logging function
    @return                 Two @see cl StreamingDataFrame, one
                            for train, one for test.

    The list of ids must hold in memory.
    There is no streaming implementation for the ids.
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
    connex = {}
    modif = 1
    iter = 0
    while modif > 0:
        modif = 0
        iter += 1
        for i, row in enumerate(dfids.itertuples(index=False, name=None)):
            c = elements[i]
            new_c = c
            for val in zip(groups, row):
                if val in connex:
                    new_c = min(c, connex[val])
            if new_c != c:
                modif += 1
                elements[i] = new_c
            for val in zip(groups, row):
                connex[val] = new_c

    if fLOG:
        fLOG(
            "[train_test_connex_split] number of iterations (connex): {0}".format(iter))

    # final
    dfids[name] = elements
    dfids[one] = 1
    grsum = dfids[[name, one]].groupby(name, as_index=False).sum()
    if fLOG:
        for g in groups:
            fLOG("[train_test_connex_split] #nb in '{0}': {1}".format(
                g, len(set(dfids[g]))))
        fLOG(
            "[train_test_connex_split] #connex {0}/{1}".format(grsum.shape[0], dfids.shape[0]))
    if grsum.shape[0] <= 1:
        raise ValueError("Every element is in the same connected components.")

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
    return train_f, test_f
