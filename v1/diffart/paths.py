"""External inputs are supplied by the caller; no bundled evaluation assets."""
import os
from pathlib import Path
def required_path(name):
    value=os.environ.get(name)
    if not value: raise RuntimeError(f"Set {name} to your separately obtained input path")
    return Path(value).expanduser()
def input_path(name, default):
    return Path(os.environ.get(name,default)).expanduser()
