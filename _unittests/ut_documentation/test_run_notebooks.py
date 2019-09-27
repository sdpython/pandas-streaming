# -*- coding: utf-8 -*-
"""
@brief      test log(time=33s)
"""
import os
import unittest
from pyquickhelper.loghelper import fLOG
from pyquickhelper.pycode import ExtTestCase
from pyquickhelper.ipythonhelper import test_notebook_execution_coverage
import pandas_streaming


class TestRunNotebooksPython(ExtTestCase):

    def setUp(self):
        import jyquickhelper  # pylint: disable=C0415
        self.assertTrue(jyquickhelper is not None)

    def test_notebook_artificiel(self):
        fLOG(
            __file__,
            self._testMethodName,
            OutputPrint=__name__ == "__main__")

        self.assertTrue(pandas_streaming is not None)
        folder = os.path.join(os.path.dirname(__file__),
                              "..", "..", "_doc", "notebooks")
        test_notebook_execution_coverage(
            __file__, "first_steps", folder, 'pandas_streaming', copy_files=[], fLOG=fLOG)


if __name__ == "__main__":
    unittest.main()
