# -*- coding: utf-8 -*-
import sys
import os
from sphinx_runpython.github_link import make_linkcode_resolve
from sphinx_runpython.conf_helper import has_dvipng, has_dvisvgm
from pandas_streaming import __version__


extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.ifconfig",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx_gallery.gen_gallery",
    "sphinx_issues",
    "sphinx_runpython.blocdefs.sphinx_exref_extension",
    "sphinx_runpython.blocdefs.sphinx_mathdef_extension",
    "sphinx_runpython.epkg",
    "sphinx_runpython.gdot",
    "sphinx_runpython.runpython",
    "matplotlib.sphinxext.plot_directive",
]

if has_dvisvgm():
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
elif has_dvipng():
    extensions.append("sphinx.ext.pngmath")
    imgmath_image_format = "png"
else:
    extensions.append("sphinx.ext.mathjax")

templates_path = ["_templates"]
html_logo = "_static/project_ico.png"
source_suffix = ".rst"
master_doc = "index"
project = "pandas-streaming"
copyright = "2016-2023, Xavier Dupré"
author = "Xavier Dupré"
version = __version__
release = __version__
language = "en"
exclude_patterns = ["auto_examples/*.ipynb"]
pygments_style = "sphinx"
todo_include_todos = True
nbsphinx_execute = "never"

html_theme = "furo"
html_theme_path = ["_static"]
html_theme_options = {}
html_sourcelink_suffix = ""
html_static_path = ["_static"]

issues_github_path = "sdpython/pandas-streaming"

# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "pandas_streaming",
    (
        "https://github.com/sdpython/pandas-streaming/"
        "blob/{revision}/{package}/"
        "{path}#L{lineno}"
    ),
)

latex_elements = {
    "papersize": "a4",
    "pointsize": "10pt",
    "title": project,
}

mathjax3_config = {"chtml": {"displayAlign": "left"}}

intersphinx_mapping = {
    "onnx": ("https://onnx.ai/onnx/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "sklearn-onnx": ("https://onnx.ai/sklearn-onnx/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Check intersphinx reference targets exist
nitpicky = True
# See also scikit-learn/scikit-learn#26761
nitpick_ignore = [
    ("py:class", "False"),
    ("py:class", "True"),
    ("py:class", "pipeline.Pipeline"),
    ("py:class", "default=sklearn.utils.metadata_routing.UNCHANGED"),
]

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": os.path.join(os.path.dirname(__file__), "examples"),
    # path where to save gallery generated examples
    "gallery_dirs": "auto_examples",
}

# next

preamble = """
\\usepackage{etex}
\\usepackage{fixltx2e} % LaTeX patches, \\textsubscript
\\usepackage{cmap} % fix search and cut-and-paste in Acrobat
\\usepackage[raccourcis]{fast-diagram}
\\usepackage{titlesec}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{amsfonts}
\\usepackage{graphics}
\\usepackage{epic}
\\usepackage{eepic}
%\\usepackage{pict2e}
%%% Redefined titleformat
\\setlength{\\parindent}{0cm}
\\setlength{\\parskip}{1ex plus 0.5ex minus 0.2ex}
\\newcommand{\\hsp}{\\hspace{20pt}}
\\newcommand{\\acc}[1]{\\left\\{#1\\right\\}}
\\newcommand{\\cro}[1]{\\left[#1\\right]}
\\newcommand{\\pa}[1]{\\left(#1\\right)}
\\newcommand{\\R}{\\mathbb{R}}
\\newcommand{\\HRule}{\\rule{\\linewidth}{0.5mm}}
%\\titleformat{\\chapter}[hang]{\\Huge\\bfseries\\sffamily}{\\thechapter\\hsp}{0pt}{\\Huge\\bfseries\\sffamily}

\\usepackage[all]{xy}
\\newcommand{\\vecteur}[2]{\\pa{#1,\\dots,#2}}
\\newcommand{\\N}[0]{\\mathbb{N}}
\\newcommand{\\indicatrice}[1]{ {1\\!\\!1}_{\\acc{#1}} }
\\newcommand{\\infegal}[0]{\\leqslant}
\\newcommand{\\supegal}[0]{\\geqslant}
\\newcommand{\\ensemble}[2]{\\acc{#1,\\dots,#2}}
\\newcommand{\\fleche}[1]{\\overrightarrow{ #1 }}
\\newcommand{\\intervalle}[2]{\\left\\{#1,\\cdots,#2\\right\\}}
\\newcommand{\\independant}[0]{\\perp \\!\\!\\! \\perp}
\\newcommand{\\esp}{\\mathbb{E}}
\\newcommand{\\espf}[2]{\\mathbb{E}_{#1}\\pa{#2}}
\\newcommand{\\var}{\\mathbb{V}}
\\newcommand{\\pr}[1]{\\mathbb{P}\\pa{#1}}
\\newcommand{\\loi}[0]{{\\cal L}}
\\newcommand{\\vecteurno}[2]{#1,\\dots,#2}
\\newcommand{\\norm}[1]{\\left\\Vert#1\\right\\Vert}
\\newcommand{\\norme}[1]{\\left\\Vert#1\\right\\Vert}
\\newcommand{\\scal}[2]{\\left<#1,#2\\right>}
\\newcommand{\\dans}[0]{\\rightarrow}
\\newcommand{\\partialfrac}[2]{\\frac{\\partial #1}{\\partial #2}}
\\newcommand{\\partialdfrac}[2]{\\dfrac{\\partial #1}{\\partial #2}}
\\newcommand{\\trace}[1]{tr\\pa{#1}}
\\newcommand{\\sac}[0]{|}
\\newcommand{\\abs}[1]{\\left|#1\\right|}
\\newcommand{\\loinormale}[2]{{\\cal N} \\pa{#1,#2}}
\\newcommand{\\loibinomialea}[1]{{\\cal B} \\pa{#1}}
\\newcommand{\\loibinomiale}[2]{{\\cal B} \\pa{#1,#2}}
\\newcommand{\\loimultinomiale}[1]{{\\cal M} \\pa{#1}}
\\newcommand{\\variance}[1]{\\mathbb{V}\\pa{#1}}
\\newcommand{\\intf}[1]{\\left\\lfloor #1 \\right\\rfloor}
"""

imgmath_latex_preamble = preamble
latex_elements["preamble"] = imgmath_latex_preamble


epkg_dictionary = {
    "csv": "https://en.wikipedia.org/wiki/Comma-separated_values",
    "dask": "https://dask.pydata.org/en/latest/",
    "dataframe": "https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html",
    "Dataframe": "https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html",
    "DataFrame": "https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html",
    "dataframes": "https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html",
    "dill": "https://dill.readthedocs.io/en/latest/dill.html",
    "groupby and missing values": "https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html",
    "Hadoop": "http://hadoop.apache.org/",
    "ijson": "https://github.com/ICRAR/ijson",
    "nan": "https://numpy.org/doc/stable/reference/constants.html#numpy.NAN",
    "pandas": (
        "http://pandas.pydata.org/pandas-docs/stable/",
        (
            "http://pandas.pydata.org/pandas-docs/stable/generated/pandas.{0}.html",
            1,
        ),
        (
            "http://pandas.pydata.org/pandas-docs/stable/generated/pandas.{0}.{1}.html",
            2,
        ),
    ),
    "pyarrow": "https://arrow.apache.org/docs/python/",
    "pyspark": "http://spark.apache.org/docs/2.1.1/api/python/index.html",
    "scikit-multiflow": "https://scikit-multiflow.github.io/",
    "sklearn": (
        "http://scikit-learn.org/stable/",
        ("https://scikit-learn.org/stable/modules/generated/{0}.html", 1),
        ("https://scikit-learn.org/stable/modules/generated/{0}.{1}.html", 2),
    ),
    "streamz": "https://streamz.readthedocs.io/en/latest/index.html",
    "tornado": "https://www.tornadoweb.org/en/stable/",
}
