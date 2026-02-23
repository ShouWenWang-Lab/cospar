"""CoSpar - dynamic inference by integrating transcriptome and lineage information"""

__version__ = "0.5.0"

# Lazy loading for submodules
def __getattr__(name):
    """Lazily import submodules on first access."""
    _submodules = {
        "datasets",
        "hf",
        "logging",
        "pl",
        "pp",
        "settings",
        "simulate",
        "tl",
        "tmap",
    }

    if name in _submodules:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return [
        "__version__",
        "datasets",
        "hf",
        "logging",
        "pl",
        "pp",
        "settings",
        "simulate",
        "tl",
        "tmap",
    ]
