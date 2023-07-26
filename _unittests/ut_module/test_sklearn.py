import unittest
import numpy
import pandas
from sklearn.linear_model import LogisticRegression
from pandas_streaming.ext_test_case import ExtTestCase


class TestScikitLearn(ExtTestCase):
    def test_logistic_regression_check(self):
        X = pandas.DataFrame(numpy.array([[0.1, 0.2], [-0.2, 0.3]]))
        Y = numpy.array([0, 1])
        clq = LogisticRegression(
            fit_intercept=False, solver="liblinear", random_state=42
        )
        clq.fit(X, Y)
        pred2 = clq.predict(X)
        self.assertEqual(numpy.array([0, 1]), pred2)


if __name__ == "__main__":
    unittest.main()
