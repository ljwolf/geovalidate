import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

import geovalidate

project = "geovalidate"
copyright = "2024-, Levi John Wolf"
author = "Levi John Wolf"
release = geovalidate.__version__

language = "en"
html_title = project

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_immaterial",
]

bibtex_bibfiles = ["_static/references.bib"]
bibtex_reference_style = "author_year"

master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "geopandas": ("https://geopandas.org/en/latest", None),
    "libpysal": ("https://pysal.org/libpysal/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": True,
}
autodoc_typehints = "none"
suppress_warnings = ["ref.ref"]

html_theme = "sphinx_immaterial"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "icon": {"repo": "fontawesome/brands/github"},
    "site_url": "https://ljwolf.github.io/geovalidate",
    "repo_url": "https://github.com/ljwolf/geovalidate/",
    "edit_uri": "blob/main/docs/source",
    "repo_name": "ljwolf/geovalidate",
    "features": [
        "navigation.sections",
        "navigation.top",
        "search.share",
        "search.suggest",
        "toc.follow",
        "toc.sticky",
        "content.code.copy",
        "content.action.edit",
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme)",
            "toggle": {
                "icon": "material/brightness-auto",
                "name": "Switch to light mode",
            },
        },
        {
            "media": "(prefers-color-scheme: light)",
            "scheme": "default",
            "primary": "teal",
            "accent": "teal",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "scheme": "slate",
            "primary": "teal",
            "accent": "teal",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to system preference",
            },
        },
    ],
}

nb_execution_mode = "off"
nb_execution_timeout = 120
nb_kernel_name = "geovalidate"
nb_merge_streams = True
nb_execution_raise_on_error = True


def linkcode_resolve(domain, info):
    def find_source():
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(geovalidate.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "src/geovalidate/%s#L%d-L%d" % find_source()
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"
    tag = "main" if "dev" in release else ("v" + release)
    return f"https://github.com/ljwolf/geovalidate/blob/{tag}/{filename}"
