"""Runtime discovery for the vendored ABC loader.

The vendored files are copied from https://github.com/amazon-far/abc and are
kept under ``third_party/abc_minimal`` so this repository is standalone.
"""

import importlib.util
import os
from pathlib import Path


def resolve_abc_root(abc_root=None):
    """Find the vendored ABC package or an explicitly supplied override."""
    candidates = []
    vendored_root = Path(__file__).resolve().parents[1] / "third_party"
    candidates.append(vendored_root)
    if abc_root:
        candidates.append(Path(abc_root))
    env_root = os.environ.get("ABC_REPO_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    spec = importlib.util.find_spec("abc_minimal")
    if spec is not None and spec.submodule_search_locations:
        package_dir = Path(next(iter(spec.submodule_search_locations)))
        candidates.append(package_dir.parent)

    for candidate in candidates:
        if (candidate / "abc_minimal").is_dir():
            return candidate

    raise FileNotFoundError("vendored ABC loader package was not found")
