# -*- coding: utf-8 -*-
"""
@brief      test log(time=4s)
"""
import unittest
from io import StringIO
from json import loads
import pandas
from pyquickhelper.pycode import ExtTestCase
from pandas_streaming.df.dataframe_io_helpers import enumerate_json_items, JsonPerRowsStream
from pandas_streaming.df import StreamingDataFrame


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
        self.assertRaise(lambda: StreamingDataFrame.read_json(
            data), NotImplementedError)
        it = StreamingDataFrame.read_json(data, flatten=True)
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

    def test_read_json_rows2(self):
        data = '''{"a": 1, "b": 2}
                  {"a": 3, "b": 4}'''
        it = StreamingDataFrame.read_json(StringIO(data), lines="stream")
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

    def test_read_json_stream(self):
        text = """{'a': 1}
        {'b': 1, 'a', 'r'}"""
        st = JsonPerRowsStream(StringIO(text))
        val = st.getvalue().replace(" ", "").replace("\n", "")
        exp = "[{'a':1},{'b':1,'a','r'}]"
        self.assertEqual(val, exp)

        st = JsonPerRowsStream(StringIO(text))
        t = st.read(0)
        t = st.read(1)
        c = ""
        while t:
            c += t
            t = st.read(1)
        val = c.replace(" ", "").replace("\n", "")
        self.assertEqual(val, exp)

    def test_enumerate_json_items_lines(self):
        data = '''{"a": 1, "b": 2}
                  {"a": 3, "b": 4}'''
        items = list(enumerate_json_items(data, lines=True))
        self.assertEqual(items, [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])

    def test_read_json_file2(self):
        data = '''{"a": {"c": 1}, "b": [2, 3]}
                  {"a": {"a": 3}, "b": [4, 5, "r"]}'''

        obj1 = list(enumerate_json_items(
            StringIO(data), flatten=False, lines=True))
        obj2 = list(enumerate_json_items(
            StringIO(data), flatten=True, lines=True))
        self.assertNotEqual(obj1, obj2)
        self.assertEqual(obj2, [{'a_c': 1, 'b_0': 2, 'b_1': 3},
                                {'a_a': 3, 'b_0': 4, 'b_1': 5, 'b_2': 'r'}])

        it = StreamingDataFrame.read_json(
            StringIO(data), lines="stream", flatten=True)
        dfs = list(it)
        self.assertEqual(list(dfs[0].columns), [
                         'a_a', 'a_c', 'b_0', 'b_1', 'b_2'])
        self.assertEqual(len(dfs), 1)
        js = dfs[0].to_json(orient='records', lines=True)
        jsjson = loads('[' + js.replace("\n", ",") + ']')
        exp = [{'a_a': None, 'a_c': 1.0, 'b_0': 2, 'b_1': 3, 'b_2': None},
               {'a_a': 3.0, 'a_c': None, 'b_0': 4, 'b_1': 5, 'b_2': 'r'}]
        self.assertEqual(jsjson, exp)

    def test_read_json_item(self):
        text = TestDataFrameIOHelpers.text_json
        st = JsonPerRowsStream(StringIO(text))
        res = []
        while True:
            n = st.read()
            if not n:
                break
            res.append(n)
        self.assertGreater(len(res), 1)


if __name__ == "__main__":
    unittest.main()
