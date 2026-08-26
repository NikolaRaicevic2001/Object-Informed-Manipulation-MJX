import os
from pathlib import Path

import jax

# package root
ROOT = str(Path(__file__).parent.absolute())

# Set XLA flags for better performance
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=true "



def _compilation_cache_dir() -> str | None:
    """A writable persistent JAX cache directory, or None to disable it.

    Per-user, not a fixed `/tmp/jax_cache`: that path is shared, and on a
    multi-user server the first account to run creates it and every other
    account then fails every write with `PermissionError: [Errno 13]`,
    once per compile. JAX only warns, so the sweep keeps going -- and
    recompiles from scratch in every one of its cells, since each cell is
    its own subprocess. The failure is quiet and costs hours.

    `$OIM_JAX_CACHE_DIR` overrides, for a node with a fast scratch disk or
    a home directory under quota. Under `$XDG_CACHE_HOME`/`~/.cache`
    otherwise, which survives `/tmp` cleanup between sweep cells.

    Returns:
        The directory, created and write-tested, or None if it cannot be
        used -- better to compile every time than to warn every time.
    """
    override = os.environ.get("OIM_JAX_CACHE_DIR")
    base = override or os.path.join(
        os.environ.get("XDG_CACHE_HOME")
        or os.path.join(os.path.expanduser("~"), ".cache"),
        "oim",
        "jax",
    )
    try:
        os.makedirs(base, exist_ok=True)
        probe = os.path.join(base, f".write-test-{os.getpid()}")
        with open(probe, "w") as f:
            f.write("")
        os.remove(probe)
    except OSError:
        return None
    return base


# Enable persistent compilation cache
_CACHE_DIR = _compilation_cache_dir()
if _CACHE_DIR is not None:
    jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
