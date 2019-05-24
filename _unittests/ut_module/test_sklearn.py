# -*- coding: utf-8 -*-
"""
@brief      test log(time=2s)
"""
import unittest
import warnings
import numpy
import pandas
from sklearn.linear_model import LogisticRegression
from pyquickhelper.pycode import ExtTestCase


class TestScikitLearn(ExtTestCase):

    def test_logistic_regression_check(self):
        X = pandas.DataFrame(numpy.array([[0.1, 0.2], [0.2, 0.3]]))
        Y = numpy.array([0, 1])
        clq = LogisticRegression(fit_intercept=False)
        clq.fit(X, Y)
        pred2 = clq.predict(X)
        if pred2[0] != 0 or pred2[1] != 1:
            warnings.warn("test_logistic_regression_check FAILS {0}".format(
                clq.predict_proba(X)))
        else:
            warnings.warn("test_logistic_regression_check SUCCEEDS {0}".format(
                clq.predict_proba(X)))


if __name__ == "__main__":
    unittest.main()
