"""Facts about the machine and code that produced a results file."""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

PACKAGES = (
    "numpy",
    "onnxruntime",
    "umap-learn",
    "numba",
    "tokenizers",
    "textual",
    "pydantic",
)


def _cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
            return str(winreg.QueryValueEx(handle, "ProcessorNameString")[0]).strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _physical_cores() -> int | None:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor).NumberOfCores",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout
            return sum(int(x) for x in out.split())
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cores = {
            line.split(":", 1)[1].strip()
            for line in cpuinfo.read_text().splitlines()
            if line.startswith(("core id", "physical id"))
        }
        return len(cores) or None
    return None


def _ram_bytes() -> int | None:
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return None


def machine_info() -> dict[str, Any]:
    ram = _ram_bytes()
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "cpu": _cpu_name(),
        "physical_cores": _physical_cores(),
        "logical_cores": os.cpu_count(),
        "ram_gb": round(ram / 2**30, 1) if ram else None,
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": versions,
        "gpu": None,
    }


def git_state(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, timeout=30, check=True
        ).stdout.strip()

    try:
        sha = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": None, "git_dirty": None}
    return {"git_sha": sha, "git_dirty": dirty}


def local_date() -> str:
    """The machine's local calendar date, which is what the README quotes."""
    return datetime.now().astimezone().date().isoformat()


def local_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
