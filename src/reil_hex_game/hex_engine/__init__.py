from .hex_engine import hexPosition

import importlib.util, sys, warnings, re

# Ensure sitecustomize hook exists even if running in embedded interpreter
def _suppress_pygame_pkg_resources_warning():
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API",
        category=UserWarning,
        module=r"pygame\.pkgdata",
    )

_suppress_pygame_pkg_resources_warning()