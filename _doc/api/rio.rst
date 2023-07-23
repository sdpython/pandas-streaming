
Inputs / Outputs
================

.. contents::
    :local:

Dataframes / Numpy arrays
+++++++++++++++++++++++++

`HDF5 <https://pandas.pydata.org/pandas-docs/stable/io.html#hdf5-pytables>`_
is easy to manipulate in the :epkg:`Python` world but difficult
to exchange with other people and other environments.
The two following functions makes it easier to collapse many dataframes
or numpy arrays into one single file. The data can be unzipped afterwards.

.. autofunction:: pandas_streaming.df.dataframe_io.read_zip

.. autofunction:: pandas_streaming.df.dataframe_io.to_zip
