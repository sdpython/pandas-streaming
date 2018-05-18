# -*- coding: utf-8 -*-
"""
@brief      test log(time=4s)
"""

import sys
import os
import unittest
from io import StringIO
from json import loads
import pandas
from pyquickhelper.pycode import ExtTestCase


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


from src.pandas_streaming.df.dataframe_io_helpers import enumerate_json_items
from src.pandas_streaming.df import StreamingDataFrame


class TestDataFrameIOHelpers(ExtTestCase):

    text_json = '''
        [
        {
            "glossary": {
                "title": "example glossary",
                "GlossDiv": {
                    "title": "S",
                    "GlossList": [{
                        "GlossEntry": {
                            "ID": "SGML",
                            "SortAs": "SGML",
                            "GlossTerm": "Standard Generalized Markup Language",
                            "Acronym": "SGML",
                            "Abbrev": "ISO 8879:1986",
                            "GlossDef": {
                                "para": "A meta-markup language, used to create markup languages such as DocBook.",
                                "GlossSeeAlso": ["GML", "XML"]
                            },
                            "GlossSee": "markup"
                        }
                    }]
                }
            }
        },
        {
            "glossary": {
                "title": "example glossary",
                "GlossDiv": {
                    "title": "X",
                    "GlossList": {
                        "GlossEntry": [{
                            "ID": "SGML",
                            "SortAs": "SGML",
                            "GlossTerm": "Standard Generalized Markup Language",
                            "Acronym": "SGML",
                            "Abbrev": "ISO 8879:1986",
                            "GlossDef": {
                                "para": "A meta-markup language, used to create markup languages such as DocBook.",
                                "GlossSeeAlso": ["GML", "XML"]
                            },
                            "GlossSee": "markup"
                        }]
                    }
                }
            }
        }
        ]
    '''
    text_json_exp = [
        {
            "glossary": {
                "title": "example glossary",
                "GlossDiv": {
                    "title": "S",
                    "GlossList": [{
                        "GlossEntry": {
                            "ID": "SGML",
                            "SortAs": "SGML",
                            "GlossTerm": "Standard Generalized Markup Language",
                            "Acronym": "SGML",
                            "Abbrev": "ISO 8879:1986",
                            "GlossDef": {
                                "para": "A meta-markup language, used to create markup languages such as DocBook.",
                                "GlossSeeAlso": ["GML", "XML"]
                            },
                            "GlossSee": "markup"
                        }
                    }]
                }
            }
        },
        {
            "glossary": {
                "title": "example glossary",
                "GlossDiv": {
                    "title": "X",
                    "GlossList": {
                        "GlossEntry": [{
                            "ID": "SGML",
                            "SortAs": "SGML",
                            "GlossTerm": "Standard Generalized Markup Language",
                            "Acronym": "SGML",
                            "Abbrev": "ISO 8879:1986",
                            "GlossDef": {
                                "para": "A meta-markup language, used to create markup languages such as DocBook.",
                                "GlossSeeAlso": ["GML", "XML"]
                            },
                            "GlossSee": "markup"
                        }]
                    }
                }
            }
        }
    ]

    def test_src(self):
        "for pylint"
        self.assertFalse(src is None)

    def test_enumerate_json_items(self):
        items = list(enumerate_json_items(TestDataFrameIOHelpers.text_json))
        self.assertEqual(TestDataFrameIOHelpers.text_json_exp, items)
        items = list(enumerate_json_items(
            StringIO(TestDataFrameIOHelpers.text_json)))
        self.assertEqual(TestDataFrameIOHelpers.text_json_exp, items)

    def test_read_json_raw(self):
        data = [{'id': 1, 'name': {'first': 'Coleen', 'last': 'Volk'}},
                {'name': {'given': 'Mose', 'family': 'Regner'}},
                {'id': 2, 'name': 'Faye Raker'}]
        exp = """[{"id":1.0,"name":null,"name.family":null,"name.first":"Coleen","name.given":null,"name.last":"Volk"},
                {"id":null,"name":null,"name.family":"Regner","name.first":null,"name.given":"Mose","name.last":null},
                {"id":2.0,"name":"Faye Raker","name.family":null,"name.first":null,
                "name.given":null,"name.last":null}]""".replace(" ", "").replace("\n", "")
        it = StreamingDataFrame.read_json(data)
        dfs = list(it)
        self.assertEqual(len(dfs), 1)
        js = dfs[0].to_json(orient='records')
        self.assertEqual(js.replace(" ", ""), exp)

    def test_pandas_json_chunksize(self):
        jsonl = '''{"a": 1, "b": 2}
                   {"a": 3, "b": 4}'''
        df = pandas.read_json(jsonl, lines=True)
        idf = pandas.read_json(jsonl, lines=True, chunksize=2)
        ldf = list(idf)
        self.assertEqualDataFrame(df, ldf[0])

    def test_read_json_rows(self):
        data = '''{"a": 1, "b": 2}
                  {"a": 3, "b": 4}'''
        it = StreamingDataFrame.read_json(StringIO(data), lines=True)
        dfs = list(it)
        self.assertEqual(len(dfs), 1)
        js = dfs[0].to_json(orient='records')
        self.assertEqual(js, '[{"a":1,"b":2},{"a":3,"b":4}]')

    def test_read_json_ijson(self):
        it = StreamingDataFrame.read_json(
            StringIO(TestDataFrameIOHelpers.text_json))
        dfs = list(it)
        self.assertEqual(len(dfs), 1)
        js = dfs[0].to_json(orient='records', lines=True)
        jsjson = loads('[' + js.replace("\n", ",") + ']')
        self.assertEqual(jsjson, TestDataFrameIOHelpers.text_json_exp)


if __name__ == "__main__":
    unittest.main()
