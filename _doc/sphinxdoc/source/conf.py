# -*- coding: utf-8 -*-
import sys
import os
import sphinx_readable_theme
from pyquickhelper.helpgen.default_conf import set_sphinx_variables, get_default_stylesheet


sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0])))

local_template = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), "phdoc_templates")

set_sphinx_variables(__file__, "pandas_streaming", "Xavier Dupré", 2020,
                     "readable", sphinx_readable_theme.get_html_theme_path(),
                     locals(), extlinks=dict(
                         issue=('https://github.com/sdpython/pandas_streaming/issues/%s', 'issue')),
                     title="Streaming functionalities for pandas", book=True)

blog_root = "http://www.xavierdupre.fr/app/pandas_streaming/helpsphinx/"

html_context = {
    'css_files': get_default_stylesheet() + ['_static/my-styles.css'],
}

html_logo = "phdoc_static/project_ico.png"

html_sidebars = {}

language = "en"
custom_preamble = """\n
\\newcommand{\\vecteur}[2]{\\pa{#1,\\dots,#2}}
\\newcommand{\\N}[0]{\\mathbb{N}}
\\newcommand{\\indicatrice}[1]{\\mathbf{1\\!\\!1}_{\\acc{#1}}}
\\usepackage[all]{xy}
\\newcommand{\\infegal}[0]{\\leqslant}
\\newcommand{\\supegal}[0]{\\geqslant}
\\newcommand{\\ensemble}[2]{\\acc{#1,\\dots,#2}}
\\newcommand{\\fleche}[1]{\\overrightarrow{ #1 }}
\\newcommand{\\intervalle}[2]{\\left\\{#1,\\cdots,#2\\right\\}}
\\newcommand{\\loinormale}[2]{{\\cal N}\\pa{#1,#2}}
\\newcommand{\\independant}[0]{\\;\\makebox[3ex]{\\makebox[0ex]{\\rule[-0.2ex]{3ex}{.1ex}}\\!\\!\\!\\!\\makebox[.5ex][l]{\\rule[-.2ex]{.1ex}{2ex}}\\makebox[.5ex][l]{\\rule[-.2ex]{.1ex}{2ex}}} \\,\\,}
\\newcommand{\\esp}{\\mathbb{E}}
\\newcommand{\\var}{\\mathbb{V}}
\\newcommand{\\pr}[1]{\\mathbb{P}\\pa{#1}}
\\newcommand{\\loi}[0]{{\\cal L}}
\\newcommand{\\vecteurno}[2]{#1,\\dots,#2}
\\newcommand{\\norm}[1]{\\left\\Vert#1\\right\\Vert}
\\newcommand{\\dans}[0]{\\rightarrow}
\\newcommand{\\partialfrac}[2]{\\frac{\\partial #1}{\\partial #2}}
\\newcommand{\\partialdfrac}[2]{\\dfrac{\\partial #1}{\\partial #2}}
\\newcommand{\\loimultinomiale}[1]{{\\cal M}\\pa{#1}}
\\newcommand{\\trace}[1]{tr\\pa{#1}}
\\newcommand{\\abs}[1]{\\left|#1\\right|}
"""
# \\usepackage{eepic}

imgmath_latex_preamble += custom_preamble
latex_elements['preamble'] += custom_preamble
mathdef_link_only = True

epkg_dictionary.update({
    'csv': 'https://en.wikipedia.org/wiki/Comma-separated_values',
    'dask': 'https://dask.pydata.org/en/latest/',
    'dataframe': 'https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html',
    'Dataframe': 'https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html',
    'DataFrame': 'https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html',
    'dataframes': 'https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html',
    'pandas': ('http://pandas.pydata.org/pandas-docs/stable/',
               ('http://pandas.pydata.org/pandas-docs/stable/generated/pandas.{0}.html', 1),
               ('http://pandas.pydata.org/pandas-docs/stable/generated/pandas.{0}.{1}.html', 2)),
    'sklearn': ('http://scikit-learn.org/stable/',
                ('http://scikit-learn.org/stable/modules/generated/{0}.html', 1),
                ('http://scikit-learn.org/stable/modules/generated/{0}.{1}.html', 2)),
    'Hadoop': 'http://hadoop.apache.org/',
    'pyarrow': 'https://arrow.apache.org/docs/python/',
    'pyspark': 'http://spark.apache.org/docs/2.1.1/api/python/index.html',
    'scikit-multiflow': 'https://scikit-multiflow.github.io/',
    'streamz': 'https://streamz.readthedocs.io/en/latest/index.html',
    'tornado': 'https://www.tornadoweb.org/en/stable/',
})
