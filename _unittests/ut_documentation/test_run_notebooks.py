# -*- coding: utf-8 -*-
"""
@brief      test log(time=33s)
"""

import sys
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import get_temp_folder, ExtTestCase
from pyquickhelper.ipythonhelper import execute_notebook_list, execute_notebook_list_finalize_ut
from pyquickhelper.pycode import is_travis_or_appveyor
from pyquickhelper.ipythonhelper import install_python_kernel_for_unittest

try:
    import src
except ImportError:
    path = os.path.normpath(
        os.path.abspath(
            os.path.join(
                os.path.split(__file__)[0],
                "..",
                "..")))
    if path not in sys.path:
        sys.path.append(path)
    import src

import src.pandas_streaming


class TestRunNotebooksPython(ExtTestCase):

    def test_src(self):
        "for pylint"
        self.assertFalse(src is None)

    def test_run_notebook(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        kernel_name = None if is_travis_or_appveyor() else install_python_kernel_for_unittest(
            "pandas_streaming")

        temp = get_temp_folder(__file__, "temp_run_notebooks")

        # selection of notebooks
        fnb = os.path.normpath(os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "..", "..", "_doc", "notebooks"))
        keepnote = []
        if not os.path.exists(fnb):
            return
        for f in os.listdir(fnb):
            if os.path.splitext(f)[-1] == ".ipynb" and "_long" not in f:
                keepnote.append(os.path.join(fnb, f))
        self.assertGreater(len(keepnote), 0)

        # function to tell that a can be run
        def valid(cell):
            return True

        # additionnal path to add
        addpaths = [os.path.normpath(os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "..", "..", "src")),
            os.path.normpath(os.path.join(
                os.path.abspath(os.path.dirname(__file__)), "..", "..", "..", "jyquickhelper", "src")),
            os.path.normpath(os.path.join(
                os.path.abspath(os.path.dirname(__file__)), "..", "..", "..", "pyquickhelper", "src"))
        ]

        # run the notebooks
        res = execute_notebook_list(
            temp, keepnote, fLOG=fLOG, valid=valid, additional_path=addpaths, kernel_name=kernel_name)
        execute_notebook_list_finalize_ut(
            res, fLOG=fLOG, dump=src.pandas_streaming)


if __name__ == "__main__":
    unittest.main()
