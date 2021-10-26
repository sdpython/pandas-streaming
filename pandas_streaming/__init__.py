# -*- coding: utf-8 -*-
"""
@file
@brief Module *pandas_streaming*.
Processes large datasets with :epkg:`pandas` by
reimplementing streeaming versions of
:epkg:`pandas` functionalites.
"""

__version__ = "0.3.218"
__author__ = "Xavier Dupré"
__github__ = "https://github.com/sdpython/pandas_streaming"
__url__ = "http://www.xavierdupre.fr/app/pandas_streaming/helpsphinx/index.html"
__license__ = "MIT License"
__blog__ = """
<?xml version="1.0" encoding="UTF-8"?>
<opml version="1.0">
    <head>
        <title>blog</title>
    </head>
    <body>
        <outline text="pandas_streaming"
            title="pandas_streaming"
            type="rss"
            xmlUrl="http://www.xavierdupre.fr/app/pandas_streaming/helpsphinx/_downloads/rss.xml"
            htmlUrl="http://www.xavierdupre.fr/app/pandas_streaming/helpsphinx/blog/main_0000.html" />
    </body>
</opml>
"""


def check(log=False):
    """
    Checks the library is working.
    It raises an exception.
    If you want to disable the logs:

    :param log: if True, display information, otherwise none
    :return: 0 or exception
    """
    return True


def _setup_hook(use_print=False):
    """
    if this function is added to the module,
    the help automation and unit tests call it first before
    anything goes on as an initialization step.
    """
    # we can check many things, needed module
    # any others things before unit tests are started
    if use_print:
        print("Success: _setup_hook")
