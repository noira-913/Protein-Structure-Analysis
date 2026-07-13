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
import sys

repo_root = os.path.abspath(SPECPATH)

# Bundle whichever compiled physics extensions match the build interpreter.
# The CUDA variant is optional — only picked up if present (e.g. built
# locally with the CUDA toolkit installed); CI builds are CPU-only.
# protein_analysis (2026-07-13, IMPROVEMENTS.md item #7 performance follow-up)
# was missing from this list until merge-readiness audit found it: gui_main.py
# imports it for accelerated ensemble-metrics/knot-classification code and
# falls back to pure Python silently if absent, so the gap never crashed —
# it would have just silently shipped every portable .exe build without the
# 14x-1628x speedup this extension exists for.
extension_binaries = [
    (path, ".")
    for pattern in ("protein_physics*.pyd", "protein_physics_cuda*.pyd",
                     "protein_analysis*.pyd")
    for path in glob.glob(os.path.join(repo_root, pattern))
]

# OpenSSL DLLs (2026-07-09): PyInstaller's automatic dependency walker finds
# _ssl.pyd (it shows up in the build's own xref graph) but does NOT detect or
# bundle libssl-3-x64.dll / libcrypto-3-x64.dll, the two OpenSSL DLLs _ssl.pyd
# actually links against -- confirmed missing from both the xref graph and
# warn-alma.txt after a real build. Without them, ssl.py's `import _ssl`
# fails at runtime inside the frozen exe with "Can't connect to HTTPS URL
# because the SSL module is not available" -- breaking every RCSB/AlphaFold/
# SWISS-MODEL fetch (requests/urllib3 both need ssl for HTTPS), even though
# the exact same code works from the unfrozen venv. A known category of gap
# with python-build-standalone-style DLL layouts (this project's .venv is
# built via `uv venv`, which uses python-build-standalone) -- PyInstaller's
# binary scanner is more commonly exercised against a standard python.org
# installer's layout. Fix: bundle them explicitly rather than relying on
# auto-detection, glob-matched (not a hardcoded exact filename) so a future
# OpenSSL version bump (e.g. libssl-3 -> libssl-4) doesn't silently break
# this again.
ssl_dll_dir = os.path.join(sys.base_prefix, "DLLs")
ssl_binaries = [
    (path, ".")
    for pattern in ("libssl-*.dll", "libcrypto-*.dll")
    for path in glob.glob(os.path.join(ssl_dll_dir, pattern))
]

# Vendored 3Dmol.js build (2026-07-13, IMPROVEMENTS.md item #7): replaces the
# CDN <script src="https://3Dmol.org/..."> tag every 3D view used to embed,
# which intermittently failed to load inside the embedded QWebEngineView.
# gui_main._vendor_asset_path() looks for this under sys._MEIPASS/vendor/ in
# a frozen build, matching the "vendor" destination folder given here.
vendor_datas = [
    (path, "vendor")
    for path in glob.glob(os.path.join(repo_root, "python", "vendor", "*"))
]

a = Analysis(
    ["python/gui_main.py"],
    pathex=[repo_root, os.path.join(repo_root, "python")],
    binaries=extension_binaries + ssl_binaries,
    datas=vendor_datas,
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
