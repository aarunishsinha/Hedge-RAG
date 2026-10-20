import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

PACKAGES = ["numpy", "bm25s", "PyStemmer", "sentence-transformers", "transformers", "torch", "ranx"]


def package_versions(names: list[str] = PACKAGES) -> dict[str, str]:
    # Reads installed package metadata only; doesn't import the packages
    return {name: version(name) for name in names}


def cpu_model() -> str | None:
    """The CPU's model name, e.g. "Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz"."""
    cpuinfo = Path("/proc/cpuinfo")              # Linux only, which includes our container
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def torch_threads() -> int | None:
    # Only report torch's thread count if this run already imported torch (dense runs).
    # Importing it just to ask would add ~1s to BM25 runs that never use it.
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    return torch.get_num_threads()


def environment() -> dict:
    """Where a run happened. Latency numbers are only comparable on the same setup.

    Inside Docker on a Mac, "platform" and "cpu_count" describe Docker's Linux
    VM (its kernel and the CPUs it is allowed), not macOS itself. The CPU model
    is the host's real CPU, since the VM runs on it directly.
    """
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch_threads(),
        "packages": package_versions(),
    }
