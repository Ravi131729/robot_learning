"""Runtime discovery for the upstream ABC checkout.

Upstream project: https://github.com/amazon-far/abc
"""

import importlib.util
import os
from pathlib import Path


def resolve_abc_root(abc_root=None):
    """Find the local checkout without hard-coding a user-specific path."""
    candidates = []
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

    raise FileNotFoundError(
        "ABC checkout not found. Clone https://github.com/amazon-far/abc "
        "and set ABC_REPO_ROOT to its local path."
    )
