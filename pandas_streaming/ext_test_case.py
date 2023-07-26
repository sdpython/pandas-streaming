import os
import sys
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any, Callable, List

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

    def assertRaise(self, fct: Callable, exc_type: Exception):
        try:
            fct()
        except exc_type as e:
            if not isinstance(e, exc_type):
                raise AssertionError(f"Unexpected exception {type(e)!r}.")
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

    @classmethod
    def tearDownClass(cls):
        for name, line, w in cls._warns:
            warnings.warn(f"\n{name}:{line}: {type(w)}\n  {str(w)}")

    def capture(self, fct: Callable):
        """
        Runs a function and capture standard output and error.

        :param fct: function to run
        :return: result of *fct*, output, error
        """
        sout = StringIO()
        serr = StringIO()
        with redirect_stdout(sout):
            with redirect_stderr(serr):
                res = fct()
        return res, sout.getvalue(), serr.getvalue()
