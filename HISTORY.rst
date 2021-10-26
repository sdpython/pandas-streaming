
.. _l-HISTORY:

=======
History
=======

current - 2021-10-26 - 0.00Mb
=============================

* #27: Fixes json parser when input is a stream (2021-10-26)
* #26: Fixes bug while reading json (iterator failed to be created twice) (2021-10-26)
* #25: Fixes documentation (2021-10-18)
* #24: Implements a first version of sort_values. (2021-10-18)
* #23: First version of operator __setitem__ (2021-10-16)
* #22: Fixes nan values after pandas update, add documentation example to the unit tests (2021-07-11)
* #21: Fixes grouping by nan values after update pandas to 1.3.0 (2021-07-10)
* #17: Implements method describe (2021-04-08)

0.2.175 - 2020-08-06 - 0.03Mb
=============================

* #16: Unit tests failing with pandas 1.1.0. (2020-08-06)
* #15: implements parameter lines, flatten for read_json (2018-11-21)
* #14: implements fillna (2018-10-29)
* #13: implement concat for axis=0,1 (2018-10-26)
* #12: add groupby_streaming (2018-10-26)
* #11: add method add_column (2018-10-26)
* #10: plan B to bypass a bug in pandas about read_csv when iterator=True --> closed, pandas has a weird behaviour when names is too small compare to the number of columns (2018-10-26)
* #9: head is very slow (2018-10-26)
* #8: fix pandas_streaming for pandas 0.23.1 (2018-07-31)
* #7: implement read_json (2018-05-17)
* #6: add pandas_groupby_nan from pyensae (2018-05-17)
* #5: add random_state parameter to splitting functions (2018-02-04)
* #2: add method sample, resevoir sampling (2017-11-05)
* #3: method train_test_split for out-of-memory datasets (2017-10-21)
* #1: Excited for your project (2017-10-10)
