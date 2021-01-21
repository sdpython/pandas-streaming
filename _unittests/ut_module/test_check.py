"""
@brief      test log(time=0s)
"""
import io
import unittest
from contextlib import redirect_stdout
from pyquickhelper.pycode import ExtTestCase
from pandas_streaming import check, _setup_hook


class TestCheck(ExtTestCase):
    """Test style."""

    def test_check(self):
        self.assertTrue(check())

    def test_setup_hook(self):
        f = io.StringIO()
        with redirect_stdout(f):
            _setup_hook(True)
        out = f.getvalue()
        self.assertIn('Success:', out)


if __name__ == "__main__":
    unittest.main()
