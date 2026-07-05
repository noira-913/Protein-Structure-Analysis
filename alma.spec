# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for a portable, single-file ALMA.exe.
#
# Build (Windows, from repo root, after `python setup.py build_ext --inplace`
# has produced protein_physics*.pyd for the interpreter running PyInstaller):
#
#   pyinstaller alma.spec --noconfirm
#
# Output: dist/ALMA.exe — copy it anywhere and run it; no installer, no
# Python required on the target machine. A `data/` folder is created next
# to the .exe on first run to cache downloaded PDB structures.

import glob
import os

repo_root = os.path.abspath(SPECPATH)

# Bundle whichever compiled physics extensions match the build interpreter.
# The CUDA variant is optional — only picked up if present (e.g. built
# locally with the CUDA toolkit installed); CI builds are CPU-only.
extension_binaries = [
    (path, ".")
    for pattern in ("protein_physics*.pyd", "protein_physics_cuda*.pyd")
    for path in glob.glob(os.path.join(repo_root, pattern))
]

a = Analysis(
    ["python/gui_main.py"],
    pathex=[repo_root, os.path.join(repo_root, "python")],
    binaries=extension_binaries,
    datas=[],
    hiddenimports=["amber_params", "iupred"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ALMA",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
)
