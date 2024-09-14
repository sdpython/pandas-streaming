
Change Logs
===========

0.5.1
+++++

* :pr:`43`: improves reproducibility of function train_test_apart_stratify

0.5.0
+++++

* :pr:`33`: removes pyquickhelper dependency
* :pr:`30`: fix compatiblity with pandas 2.0

0.3.239
+++++++

* :pr:`27`: Fixes json parser when input is a stream (2021-10-26)
* :pr:`26`: Fixes bug while reading json (iterator failed to be created twice) (2021-10-26)
* :pr:`25`: Fixes documentation (2021-10-18)
* :pr:`24`: Implements a first version of sort_values. (2021-10-18)
* :pr:`23`: First version of operator __setitem__ (2021-10-16)
* :pr:`22`: Fixes nan values after pandas update, add documentation example to the unit tests (2021-07-11)
* :pr:`21`: Fixes grouping by nan values after update pandas to 1.3.0 (2021-07-10)
* :pr:`17`: Implements method describe (2021-04-08)

0.2.175
+++++++

* :pr:`16`: Unit tests failing with pandas 1.1.0. (2020-08-06)
* :pr:`15`: implements parameter lines, flatten for read_json (2018-11-21)
* :pr:`14`: implements fillna (2018-10-29)
* :pr:`13`: implement concat for axis=0,1 (2018-10-26)
* :pr:`12`: add groupby_streaming (2018-10-26)
* :pr:`11`: add method add_column (2018-10-26)
* :pr:`10`: plan B to bypass a bug in pandas about read_csv when iterator=True --> closed, pandas has a weird behaviour when names is too small compare to the number of columns (2018-10-26)
* :pr:`9`: head is very slow (2018-10-26)
* :pr:`8`: fix pandas_streaming for pandas 0.23.1 (2018-07-31)
* :pr:`7`: implement read_json (2018-05-17)
* :pr:`6`: add pandas_groupby_nan from pyensae (2018-05-17)
* :pr:`5`: add random_state parameter to splitting functions (2018-02-04)
* :pr:`2`: add method sample, resevoir sampling (2017-11-05)
* :pr:`3`: method train_test_split for out-of-memory datasets (2017-10-21)
* :pr:`1`: Excited for your project (2017-10-10)
