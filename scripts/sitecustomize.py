import warnings, re
warnings.filterwarnings(
    action="ignore",
    message=r"pkg_resources is deprecated as an API",
    category=UserWarning,
    module=r"pygame\.pkgdata",        # ← only silence that specific module
)