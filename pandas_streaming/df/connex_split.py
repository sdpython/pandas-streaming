# -*- coding: utf-8 -*-
"""
@file
@brief Implements a connex split between train and test.
"""
from collections import Counter
import pandas
import numpy
from sklearn.model_selection import train_test_split
from .dataframe_helpers import dataframe_shuffle


class ImbalancedSplitException(Exception):
    """
    Raised when an imbalanced split is detected.
    """
    pass


def train_test_split_weights(df, weights=None, test_size=0.25, train_size=None,
                             shuffle=True, fail_imbalanced=0.05, random_state=None):
    """
    Splits a database in train/test given, every row
    can have a different weight.

    @param  df              :epkg:`pandas:DataFrame` or @see cl StreamingDataFrame
    @param  weights         None or weights or weights column name
    @param  test_size       ratio for the test partition (if *train_size* is not specified)
    @param  train_size      ratio for the train partition
    @param  shuffle         shuffles before the split
    @param  fail_imbalanced raises an exception if relative weights difference is higher than this value
    @param  random_state    seed for random generators
    @return                 train and test :epkg:`pandas:DataFrame`

    If the dataframe is not shuffled first, the function
    will produce two datasets which are unlikely to be randomized
    as the function tries to keep equal weights among both paths
    without using randomness.
    """
    if hasattr(df, 'iter_creation'):
        raise NotImplementedError(  # pragma: no cover
            'Not implemented yet for StreamingDataFrame.')
    if isinstance(df, numpy.ndarray):
        raise NotImplementedError(  # pragma: no cover
            "Not implemented on numpy arrays.")
    if shuffle:
        df = dataframe_shuffle(df, random_state=random_state)
    if weights is None:
        if test_size == 0 or train_size == 0:
            raise ValueError(
                "test_size={0} or train_size={1} cannot be null (1)."
                "".format(test_size, train_size))
        return train_test_split(df, test_size=test_size,
                                train_size=train_size,
                                random_state=random_state)

    if isinstance(weights, pandas.Series):
        weights = list(weights)
    elif isinstance(weights, str):
        weights = list(df[weights])
    if len(weights) != df.shape[0]:
        raise ValueError(
            "Dimension mismatch between weights and dataframe "
            "{0} != {1}".format(df.shape[0], len(weights)))

    p = (1 - test_size) if test_size else None
    if train_size is not None:
        p = train_size
        test_size = 1 - p
    if p is None or min(test_size, p) <= 0:
        raise ValueError(
            "test_size={0} or train_size={1} cannot be null (2)."
            "".format(test_size, train_size))
    ratio = test_size / p

    if random_state is None:
        randint = numpy.random.randint
    else:
        state = numpy.random.RandomState(random_state)
        randint = state.randint

    balance = 0
    train_ids = []
    test_ids = []
    test_weights = 0
    train_weights = 0
    for i in range(0, df.shape[0]):
        w = weights[i]
        if balance == 0:
            h = randint(0, 1)
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
        raise ImbalancedSplitException(  # pragma: no cover
            "Split is imbalanced: train_weights={0} test_weights={1} r={2}."
            "".format(train_weights, test_weights, r))

    return df.iloc[train_ids, :], df.iloc[test_ids, :]


def train_test_connex_split(df, groups, test_size=0.25, train_size=None,
                            stratify=None, hash_size=9, unique_rows=False,
                            shuffle=True, fail_imbalanced=0.05, keep_balance=None,
                            stop_if_bigger=None, return_cnx=False,
                            must_groups=None, random_state=None, fLOG=None):
    """
    This split is for a specific case where data is linked
    in many ways. Let's assume we have three ids as we have
    for online sales: *(product id, user id, card id)*.
    As we may need to compute aggregated features,
    we need every id not to be present in both train and
    test set. The function computes the connected components
    and breaks each of them in two parts for train and test.

    @param  df              :epkg:`pandas:DataFrame`
    @param  groups          columns name for the ids
    @param  test_size       ratio for the test partition (if *train_size* is not specified)
    @param  train_size      ratio for the train partition
    @param  stratify        column holding the stratification
    @param  hash_size       size of the hash to cache information about partition
    @param  unique_rows     ensures that rows are unique
    @param  shuffle         shuffles before the split
    @param  fail_imbalanced raises an exception if relative weights difference
                            is higher than this value
    @param  stop_if_bigger  (float) stops a connected components from being
                            bigger than this ratio of elements, this should not be used
                            unless a big components emerges, the algorithm stops merging
                            but does not guarantee it returns the best cut,
                            the value should be close to 0
    @param  keep_balance    (float), if not None, does not merge connected components
                            if their relative sizes are too different, the value should be
                            close to 1
    @param  return_cnx      returns connected components as a third results
    @param  must_groups     column name for ids which must not be shared by
                            train/test partitions
    @param  random_state    seed for random generator
    @param  fLOG            logging function
    @return                 Two @see cl StreamingDataFrame, one
                            for train, one for test.

    The list of ids must hold in memory.
    There is no streaming implementation for the ids.

    .. exref::
        :title: Splits a dataframe, keep ids in separate partitions
        :tag: dataframe

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

            train, test = train_test_connex_split(
                df, test_size=0.5, groups=['user', 'prod', 'card'],
                fail_imbalanced=0.6)

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

        train, test, cnx = train_test_connex_split(
            df, test_size=0.5, groups=['user', 'prod', 'card'],
            fail_imbalanced=0.6, return_cnx=True)

        print(cnx[0])
        print(cnx[1])
    """
    if stratify is not None:
        raise NotImplementedError(  # pragma: no cover
            "Option stratify is not implemented.")
    if groups is None or len(groups) == 0:
        raise ValueError(  # pragma: no cover
            "groups is empty. Use regular train_test_split.")
    if hasattr(df, 'iter_creation'):
        raise NotImplementedError(  # pragma: no cover
            'Not implemented yet for StreamingDataFrame.')
    if isinstance(df, numpy.ndarray):
        raise NotImplementedError(  # pragma: no cover
            "Not implemented on numpy arrays.")
    if shuffle:
        df = dataframe_shuffle(df, random_state=random_state)

    dfids = df[groups].copy()
    if must_groups is not None:
        dfids_must = df[must_groups].copy()

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

    def do_connex_components(dfrows, local_groups, kb, sib):
        "run connected components algorithms"
        itern = 0
        modif = 1

        while modif > 0 and itern < len(elements):
            if fLOG and df.shape[0] > 10000:
                fLOG("[train_test_connex_split] iteration={0}-#nb connect={1} - "
                     "modif={2}".format(iter, len(set(elements)), modif))
            modif = 0
            itern += 1
            for i, row in enumerate(dfrows.itertuples(index=False, name=None)):
                vals = [val for val in zip(local_groups, row) if not isinstance(
                    val[1], float) or not numpy.isnan(val[1])]

                c = elements[i]

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
                    if kb is not None:
                        maxi = min(len(counts_cnx[new_c]), len(counts_cnx[c]))
                        if maxi > 5:
                            diff = len(counts_cnx[new_c]) + \
                                len(counts_cnx[c]) - maxi
                            r = diff / float(maxi)
                            if r > kb:
                                if fLOG:  # pragma: no cover
                                    fLOG('[train_test_connex_split]    balance '
                                         'r={0:0.00000}>{1:0.00}, #[{2}]={3}, '
                                         '#[{4}]={5}'.format(r, kb, new_c,
                                                             len(counts_cnx[new_c]),
                                                             c, len(counts_cnx[c])))
                                continue

                    if sib is not None:
                        r = (len(counts_cnx[new_c]) +
                             len(counts_cnx[c])) / float(len(elements))
                        if r > sib:
                            if fLOG:  # pragma: no cover
                                fLOG('[train_test_connex_split]    no merge '
                                     'r={0:0.00000}>{1:0.00}, #[{2}]={3}, #[{4}]={5}'
                                     ''.format(r, sib, new_c, len(counts_cnx[new_c]),
                                               c, len(counts_cnx[c])))
                            avoids_merge[new_c, c] = i
                            continue

                    add_pair_c.append(c)

                if len(add_pair_c) > 0:
                    for c in add_pair_c:
                        modif += len(counts_cnx[c])
                        for ii in counts_cnx[c]:
                            elements[ii] = new_c
                        counts_cnx[new_c] = counts_cnx[new_c].union(
                            counts_cnx[c])
                        counts_cnx[c] = set()

                        keys = list(vals)
                        for val in keys:
                            if connex[val] == c:
                                connex[val] = new_c
                                modif += 1

    if must_groups:
        do_connex_components(dfids_must, must_groups, None, None)
    do_connex_components(dfids, groups, keep_balance, stop_if_bigger)

    # final
    dfids[name] = elements
    dfids[one] = 1
    grsum = dfids[[name, one]].groupby(name, as_index=False).sum()
    if fLOG:
        for g in groups:
            fLOG("[train_test_connex_split]     #nb in '{0}': {1}".format(
                g, len(set(dfids[g]))))
        fLOG(
            "[train_test_connex_split] #connex {0}/{1}".format(
                grsum.shape[0], dfids.shape[0]))
    if grsum.shape[0] <= 1:
        raise ValueError(  # pragma: no cover
            "Every element is in the same connected components.")

    # Statistics: top connected components
    if fLOG:
        # Global statistics
        counts = Counter(elements)
        cl = [(v, k) for k, v in counts.items()]
        cum = 0
        maxc = None
        fLOG("[train_test_connex_split] number of connected components: {0}"
             "".format(len(set(elements))))
        for i, (v, k) in enumerate(sorted(cl, reverse=True)):
            if i == 0:
                maxc = k, v
            if i >= 10:
                break
            cum += v
            fLOG("[train_test_connex_split]     c={0} #elements={1} cumulated"
                 "={2}/{3}".format(k, v, cum, len(elements)))

        # Most important component
        fLOG('[train_test_connex_split] first row of the biggest component '
             '{0}'.format(maxc))
        tdf = dfids[dfids[name] == maxc[0]]
        fLOG('[train_test_connex_split] \n{0}'.format(tdf.head(n=10)))

    # Splits.
    train, test = train_test_split_weights(
        grsum, weights=one, test_size=test_size, train_size=train_size,
        shuffle=shuffle, fail_imbalanced=fail_imbalanced,
        random_state=random_state)
    train.drop(one, inplace=True, axis=1)
    test.drop(one, inplace=True, axis=1)

    # We compute the final dataframe.
    def double_merge(d):
        "merge twice"
        merge1 = dfids.merge(d, left_on=name, right_on=name)
        merge2 = df.merge(merge1, left_on=groups, right_on=groups)
        return merge2

    train_f = double_merge(train)
    test_f = double_merge(test)
    if return_cnx:
        return train_f, test_f, (connex, dfids)
    else:
        return train_f, test_f


def train_test_apart_stratify(df, group, test_size=0.25, train_size=None,
                              stratify=None, force=False, random_state=None,
                              fLOG=None):
    """
    This split is for a specific case where data is linked
    in one way. Let's assume we have two ids as we have
    for online sales: *(product id, category id)*.
    A product can have multiple categories. We need to have
    distinct products on train and test but common categories
    on both sides.

    @param  df              :epkg:`pandas:DataFrame`
    @param  group           columns name for the ids
    @param  test_size       ratio for the test partition
                            (if *train_size* is not specified)
    @param  train_size      ratio for the train partition
    @param  stratify        column holding the stratification
    @param  force           if True, tries to get at least one example on the test side
                            for each value of the column *stratify*
    @param  random_state    seed for random generators
    @param  fLOG            logging function
    @return                 Two @see cl StreamingDataFrame, one
                            for train, one for test.

    .. index:: multi-label

    The list of ids must hold in memory.
    There is no streaming implementation for the ids.
    This split was implemented for a case of a multi-label
    classification. A category (*stratify*) is not exclusive
    and an observation can be assigned to multiple
    categories. In that particular case, the method
    `train_test_split <http://scikit-learn.org/stable/modules/generated/
    sklearn.model_selection.train_test_split.html>`_
    can not directly be used.

    .. runpython::
        :showcode:

        import pandas
        from pandas_streaming.df import train_test_apart_stratify

        df = pandas.DataFrame([dict(a=1, b="e"),
                               dict(a=1, b="f"),
                               dict(a=2, b="e"),
                               dict(a=2, b="f")])

        train, test = train_test_apart_stratify(
            df, group="a", stratify="b", test_size=0.5)
        print(train)
        print('-----------')
        print(test)
    """
    if stratify is None:
        raise ValueError(  # pragma: no cover
            "stratify must be specified.")
    if group is None:
        raise ValueError(  # pragma: no cover
            "group must be specified.")
    if hasattr(df, 'iter_creation'):
        raise NotImplementedError(
            'Not implemented yet for StreamingDataFrame.')
    if isinstance(df, numpy.ndarray):
        raise NotImplementedError("Not implemented on numpy arrays.")

    p = (1 - test_size) if test_size else None
    if train_size is not None:
        p = train_size
    test_size = 1 - p
    if p is None or min(test_size, p) <= 0:
        raise ValueError(  # pragma: no cover
            "test_size={0} or train_size={1} cannot be null".format(
                test_size, train_size))

    couples = df[[group, stratify]].itertuples(name=None, index=False)
    hist = Counter(df[stratify])
    sorted_hist = [(v, k) for k, v in hist.items()]
    sorted_hist.sort()
    ids = {c: set() for c in hist}

    for g, s in couples:
        ids[s].add(g)

    if random_state is None:
        permutation = numpy.random.permutation
    else:
        state = numpy.random.RandomState(random_state)
        permutation = state.permutation

    split = {}
    for _, k in sorted_hist:
        not_assigned = [c for c in ids[k] if c not in split]
        if len(not_assigned) == 0:
            continue
        assigned = [c for c in ids[k] if c in split]
        nb_test = sum(split[c] for c in assigned)
        expected = min(len(ids[k]), int(
            test_size * len(ids[k]) + 0.5)) - nb_test
        if force and expected == 0 and nb_test == 0:
            nb_train = len(assigned) - nb_test
            if nb_train > 0 or len(not_assigned) > 1:
                expected = min(1, len(not_assigned))
        if expected > 0:
            permutation(not_assigned)
            for e in not_assigned[:expected]:
                split[e] = 1
            for e in not_assigned[expected:]:
                split[e] = 0
        else:
            for c in not_assigned:
                split[c] = 0

    train_set = set(k for k, v in split.items() if v == 0)
    test_set = set(k for k, v in split.items() if v == 1)
    train_df = df[df[group].isin(train_set)]
    test_df = df[df[group].isin(test_set)]
    return train_df, test_df
