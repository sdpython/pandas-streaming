# -*- coding: utf-8 -*-
"""
@file
@brief Defines a streming dataframe.
"""


class StreamingInefficientException(Exception):
    """
    Kind of operations doable with a :epkg:`pandas:DataFrame`
    but which should not be done in streaming mode.
    """

    def __init__(self, meth):
        """
        This method is inefficient in streaming mode
        and not implemented.

        :param meth: inefficient method
        """
        Exception.__init__(
            self, "{0} should not be done in streaming mode.".format(meth))
