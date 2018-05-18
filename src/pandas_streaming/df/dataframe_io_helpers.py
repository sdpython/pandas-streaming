# -*- coding: utf-8 -*-
"""
@file
@brief Saves and reads a :epkg:`dataframe` into a :epkg:`zip` file.
"""
import os
from io import StringIO
try:
    from ujson import dumps
except ImportError:
    from json import dumps
import ijson


def enumerate_json_items(filename, encoding=None, fLOG=None):
    """
    Enumerates items from a :epkg:`JSON` file or string.

    @param      filename        filename or string or stream to parse
    @param      encoding        encoding
    @param      fLOG            logging function
    @return                     iterator on records at first level.

    It assumes the syntax follows the format: ``[ {"id":1, ...}, {"id": 2, ...}, ...]``.

    .. exref::
        :title: Processes a json file by streaming.

        The module :epkg:`ijson` can read a :epkg:`JSON` file by streaming.
        This module is needed because a record can be written on multiple lines.
        This function leverages it produces the following results.

        .. runpython::
            :showcode:

            from pandas_streaming.df.dataframe_io_helpers import enumerate_json_items

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
                            "title": "S",
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

            for item in enumerate_json_items(text_json):
                print(item)
    """
    if isinstance(filename, str):
        if "{" not in filename and os.path.exists(filename):
            with open(filename, "r", encoding=encoding) as f:
                for el in enumerate_json_items(f, encoding=encoding, fLOG=fLOG):
                    yield el
        else:
            st = StringIO(filename)
            for el in enumerate_json_items(st, encoding=encoding, fLOG=fLOG):
                yield el
    else:
        parser = ijson.parse(filename)
        current = None
        curkey = None
        stack = []
        nbyield = 0
        for i, (_, event, value) in enumerate(parser):
            if i % 1000000 == 0 and fLOG is not None:
                fLOG("[enumerate_json_items] i={0} yielded={1}".format(
                    i, nbyield))
            if event == "start_array":
                if curkey is None:
                    current = []
                else:
                    if not isinstance(current, dict):
                        raise RuntimeError(
                            "Type issue {0}".format(type(current)))
                    c = []
                    current[curkey] = c
                    current = c
                curkey = None
                stack.append(current)
            elif event == "end_array":
                stack.pop()
                if len(stack) == 0:
                    # We should be done.
                    current = None
                else:
                    current = stack[-1]
            elif event == "start_map":
                c = {}
                if curkey is None:
                    current.append(c)
                else:
                    current[curkey] = c
                stack.append(c)
                current = c
                curkey = None
            elif event == "end_map":
                stack.pop()
                current = stack[-1]
                if len(stack) == 1:
                    nbyield += 1
                    yield current[-1]
                    # We clear the memory.
                    current.clear()
            elif event == "map_key":
                curkey = value
            elif event in {"string", "number", "boolean"}:
                if curkey is None:
                    current.append(value)
                else:
                    current[curkey] = value
                    curkey = None
            elif event == "null":
                if curkey is None:
                    current.append(None)
                else:
                    current[curkey] = None
                    curkey = None
            else:
                raise ValueError("Unknown event '{0}'".format(event))


class JsonIterator2Stream:
    """
    Transforms an iterator on :epkg:`JSON` items
    into a stream which returns an items as a string every time
    method *read* is called.
    The iterator could be one returned by @see fn enumerate_json_items.

    .. exref::
        :title: Reshape a json file

        The function @see fn enumerate_json_items reads any
        :epkg:`json` even if every record is split over
        multiple lines. Class @see cl JsonIterator2Stream
        mocks this iterator as a stream. Each row is a single item.

        .. runpython::
            :showcode:

            from pandas_streaming.df.dataframe_io_helpers import enumerate_json_items

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
                            "title": "S",
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

            for item in JsonIterator2Stream(enumerate_json_items(text_json)):
                print(item)
    """

    def __init__(self, it, **kwargs):
        """
        @param      it      iterator
        @param      kwargs  arguments to :epkg:`*py:json:dumps`
        """
        self.it = it
        self.kwargs = kwargs

    def write(self):
        """
        The class does not write.
        """
        raise NotImplementedError()

    def read(self):
        """
        Reads the next item and returns it as a string.
        """
        try:
            value = next(self.it)
            return dumps(value, **self.kwargs)
        except StopIteration:
            return None

    def __iter__(self):
        """
        Iterate on each row.
        """
        for value in self.it:
            yield dumps(value, **self.kwargs)
