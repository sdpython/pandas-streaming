import os
import sys
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any, Callable, List, Optional

import numpy
from numpy.testing import assert_allclose


def unit_test_going():
    """
    Enables a flag telling the script is running while testing it.
    Avois unit tests to be very long.
    """
    going = int(os.environ.get("UNITTEST_GOING", 0))
    return going == 1


def ignore_warnings(warns: List[Warning]) -> Callable:
    """
    Catches warnings.

    :param warns:   warnings to ignore
    """

    def wrapper(fct):
        if warns is None:
            raise AssertionError(f"warns cannot be None for '{fct}'.")

        def call_f(self):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warns)
                return fct(self)

        return call_f

    return wrapper


class sys_path_append:
    """
    Stores the content of :epkg:`*py:sys:path` and
    restores it afterwards.
    """

    def __init__(self, paths, position=-1):
        """
        :param paths: paths to add
        :param position: where to add it
        """
        self.to_add = paths if isinstance(paths, list) else [paths]
        self.position = position

    def __enter__(self):
        """
        Modifies ``sys.path``.
        """
        self.store = sys.path.copy()
        if self.position == -1:
            sys.path.extend(self.to_add)
        else:
            for p in reversed(self.to_add):
                sys.path.insert(self.position, p)

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Restores``sys.path``.
        """
        sys.path = self.store


class ExtTestCase(unittest.TestCase):
    _warns = []

    def assertExists(self, name):
        if not os.path.exists(name):
            raise AssertionError(f"File or folder {name!r} does not exists.")

    def assertEqualArray(
        self,
        expected: numpy.ndarray,
        value: numpy.ndarray,
        atol: float = 0,
        rtol: float = 0,
    ):
        self.assertEqual(expected.dtype, value.dtype)
        self.assertEqual(expected.shape, value.shape)
        assert_allclose(expected, value, atol=atol, rtol=rtol)

    def assertEqualDataFrame(self, d1, d2, **kwargs):
        """
        Checks that two dataframes are equal.
        Calls :func:`pandas.testing.assert_frame_equal`.
        """
        from pandas.testing import assert_frame_equal

        assert_frame_equal(d1, d2, **kwargs)

    def assertAlmostEqual(
        self,
        expected: numpy.ndarray,
        value: numpy.ndarray,
        atol: float = 0,
        rtol: float = 0,
    ):
        if not isinstance(expected, numpy.ndarray):
            expected = numpy.array(expected)
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value).astype(expected.dtype)
        self.assertEqualArray(expected, value, atol=atol, rtol=rtol)

    def assertRaise(
        self, fct: Callable, exc_type: Exception, msg: Optional[str] = None
    ):
        try:
            fct()
        except exc_type as e:
            if not isinstance(e, exc_type):
                raise AssertionError(f"Unexpected exception {type(e)!r}.") from e
            if msg is None:
                return
            if msg not in str(e):
                raise AssertionError(f"Unexpected error message {e!r}.") from e
            return
        raise AssertionError("No exception was raised.")

    def assertEmpty(self, value: Any):
        if value is None:
            return
        if len(value) == 0:
            return
        raise AssertionError(f"value is not empty: {value!r}.")

    def assertNotEmpty(self, value: Any):
        if value is None:
            raise AssertionError(f"value is empty: {value!r}.")
        if isinstance(value, (list, dict, tuple, set)):
            if len(value) == 0:
                raise AssertionError(f"value is empty: {value!r}.")

    def assertStartsWith(self, prefix: str, full: str):
        if not full.startswith(prefix):
            raise AssertionError(f"prefix={prefix!r} does not start string  {full!r}.")

    def assertLesser(self, x, y, strict=False):
        """
        Checks that ``x <= y``.
        """
        if x > y or (strict and x == y):
            raise AssertionError(
                "x >{2} y with x={0} and y={1}".format(  # noqa: UP030
                    ExtTestCase._format_str(x),
                    ExtTestCase._format_str(y),
                    "" if strict else "=",
                )
            )

    @staticmethod
    def abs_path_join(filename: str, *args: List[str]):
        """
        Returns an absolute and normalized path from this location.

        :param filename: filename, the folder which contains it
            is used as the base
        :param args: list of subpaths to the previous path
        :return: absolute and normalized path
        """
        dirname = os.path.join(os.path.dirname(filename), *args)
        return os.path.normpath(os.path.abspath(dirname))

    @classmethod
    def tearDownClass(cls):
        for name, line, w in cls._warns:
            warnings.warn(f"\n{name}:{line}: {type(w)}\n  {str(w)}", stacklevel=0)

    def capture(self, fct: Callable):
        """
        Runs a function and capture standard output and error.

        :param fct: function to run
        :return: result of *fct*, output, error
        """
        sout = StringIO()
        serr = StringIO()
        with redirect_stdout(sout), redirect_stderr(serr):
            res = fct()
        return res, sout.getvalue(), serr.getvalue()
