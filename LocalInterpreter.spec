# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Local Interpreter.

Produces a self-contained onedir build:

    LocalInterpreter.exe        the GUI (windowed)
    LocalInterpreterFetch.exe   console model downloader, used by the installer

Everything - Python, PyQt6, CTranslate2, the CUDA runtime and cuDNN - is
copied next to the executables (contents_directory='.'), because cuDNN loads
its own sub-libraries by bare name and only the application directory is
searched reliably for those.
"""

import glob
import os
import site
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

PROJECT = Path(os.path.abspath(SPECPATH))
ICON = PROJECT / "assets" / "app.ico"

datas, binaries, hiddenimports = [], [], []

# --- python packages -------------------------------------------------------
for pkg in (
    "ctranslate2",
    "faster_whisper",
    "onnxruntime",
    "soundcard",
    "av",
    "tokenizers",
    "huggingface_hub",
    "tqdm",
):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += ["numpy", "cffi", "_cffi_backend"]


# --- CUDA runtime ----------------------------------------------------------
# CTranslate2 needs cuBLAS 12 and cuDNN 9 at runtime but ships neither. They
# are taken from whichever pip package provides them on the build machine
# (torch, or the nvidia-* wheels).
CUDA_DLLS = [
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_heuristic64_9.dll",
]

search_dirs = []
roots = list(site.getsitepackages())
try:
    roots.append(site.getusersitepackages())
except Exception:
    pass
for root in roots:
    search_dirs.append(Path(root) / "torch" / "lib")
    nvidia = Path(root) / "nvidia"
    if nvidia.is_dir():
        for sub in nvidia.iterdir():
            search_dirs += [sub / "bin", sub / "lib"]

found, missing = {}, []
for name in CUDA_DLLS:
    for d in search_dirs:
        p = d / name
        if p.is_file():
            found[name] = str(p)
            break
    else:
        missing.append(name)

for name, path in found.items():
    binaries.append((path, "."))

print(f"[spec] CUDA libraries bundled: {len(found)}/{len(CUDA_DLLS)}")
if missing:
    print(f"[spec] WARNING - not found, GPU mode may fall back to CPU: {missing}")

# --- MSVC runtime ----------------------------------------------------------
# Guarantees the app starts on a machine that never had a redistributable.
for name in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll",
             "msvcp140_1.dll", "msvcp140_2.dll", "concrt140.dll"):
    p = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / name
    if p.is_file():
        binaries.append((str(p), "."))


a = Analysis(
    [str(PROJECT / "local_interpreter.py")],
    pathex=[str(PROJECT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch", "torchvision", "torchaudio",
        "tkinter", "matplotlib", "scipy", "pandas", "PIL", "pyarrow",
        "IPython", "notebook", "jupyter", "pytest",
        "PySide6", "PyQt5", "transformers", "sympy", "networkx",
    ],
    noarchive=False,
    optimize=0,
)

# PyInstaller resolves ctranslate2.dll's CUDA dependencies to wherever the
# providing pip package happens to keep them - torch\lib, or nvidia\*\bin for
# the standalone wheels - and preserves that path. Each one is then a second
# copy of a DLL already placed at the top level, which is close to a gigabyte
# of pure duplication. Keep only the top-level copies.
_cuda_names = {name.lower() for name in CUDA_DLLS}


def _drop_nested_cuda(entries):
    kept, dropped = [], 0
    for entry in entries:
        dest = entry[0].replace("/", "\\")
        if "\\" in dest and os.path.basename(dest).lower() in _cuda_names:
            dropped += os.path.getsize(entry[1]) if os.path.isfile(entry[1]) else 0
            continue
        kept.append(entry)
    return kept, dropped


a.binaries, _dropped_bytes = _drop_nested_cuda(a.binaries)
a.datas, _more = _drop_nested_cuda(a.datas)
print(f"[spec] dropped {(_dropped_bytes + _more) / 1e6:,.0f} MB of duplicated CUDA libraries")

pyz = PYZ(a.pure)

gui = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LocalInterpreter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=str(ICON) if ICON.exists() else None,
    contents_directory=".",
)

fetch = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LocalInterpreterFetch",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=str(ICON) if ICON.exists() else None,
    contents_directory=".",
)

coll = COLLECT(
    gui,
    fetch,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LocalInterpreter",
)
