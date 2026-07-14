# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for a portable, single-file ALMA build.
#
# Build (from repo root, after `python setup.py build_ext --inplace` has
# produced protein_physics*/protein_analysis* for the interpreter running
# PyInstaller):
#
#   pyinstaller alma.spec --noconfirm
#
# Output: dist/ALMA(.exe) — copy it anywhere and run it; no installer, no
# Python required on the target machine. A `data/` folder is created next
# to the executable on first run to cache downloaded PDB structures.
#
# Cross-platform (2026-07-13): originally Windows-only. macOS support added
# to the release workflow the same day -- this spec is shared by both, so
# the extension-file glob and the SSL-DLL bundling below are now
# platform-conditional rather than hardcoded to Windows' .pyd/DLL layout.
# No Linux release job exists yet, but nothing here is Windows/macOS-specific
# beyond what's explicitly guarded, so it likely mostly works there too --
# untested, not claimed.

import glob
import os
import sys

repo_root = os.path.abspath(SPECPATH)
is_win = sys.platform == "win32"

# Bundle whichever compiled physics extensions match the build interpreter.
# The CUDA variant is Windows/NVIDIA-only and optional -- only picked up if
# present (e.g. built locally with the CUDA toolkit installed); CI's Windows
# build produces it, CI's macOS build never will (setup.py's find_cuda()
# already no-ops there). protein_analysis (2026-07-13, IMPROVEMENTS.md item
# #7 performance follow-up) was missing from this list until a merge-
# readiness audit found it: gui_main.py imports it for accelerated
# ensemble-metrics/knot-classification code and falls back to pure Python
# silently if absent, so the gap never crashed -- it would have just
# silently shipped every portable build without the 14x-1628x speedup this
# extension exists for. Matches both suffixes (.pyd on Windows, .so on
# macOS/Linux) rather than assuming Windows -- glob on a suffix that
# doesn't apply on this platform just returns empty, so this is a safe
# unconditional check, not something needing an is_win guard.
#
# Patterns use a literal "." right after the module name (matching Python's
# EXT_SUFFIX convention, e.g. ".cp314-win_amd64.pyd" / ".cpython-312-darwin.so")
# rather than a bare "protein_physics*" -- found and fixed a real duplicate-
# bundling bug here (2026-07-13, macOS-support pass): a bare "*" after
# "protein_physics" also matches "protein_physics_cuda...", so the CUDA
# extension was being globbed twice (once under its own pattern, once again
# under the plain CPU pattern's wildcard) every prior build. Harmless in
# practice (PyInstaller just bundled the same file redundantly), but not
# correct, and worth fixing while touching this section anyway.
extension_binaries = [
    (path, ".")
    for stem in ("protein_physics.", "protein_physics_cuda.", "protein_analysis.")
    for suffix in ("pyd", "so")
    for path in glob.glob(os.path.join(repo_root, stem + "*." + suffix))
]

# OpenSSL DLLs (2026-07-09, Windows-only): PyInstaller's automatic dependency
# walker finds _ssl.pyd (it shows up in the build's own xref graph) but does
# NOT detect or bundle libssl-3-x64.dll / libcrypto-3-x64.dll, the two
# OpenSSL DLLs _ssl.pyd actually links against -- confirmed missing from
# both the xref graph and warn-alma.txt after a real build. Without them,
# ssl.py's `import _ssl` fails at runtime inside the frozen exe with "Can't
# connect to HTTPS URL because the SSL module is not available" -- breaking
# every RCSB/AlphaFold/SWISS-MODEL fetch (requests/urllib3 both need ssl for
# HTTPS), even though the exact same code works from the unfrozen venv. A
# known category of gap with python-build-standalone-style DLL layouts
# (this project's .venv is built via `uv venv`, which uses
# python-build-standalone) -- PyInstaller's binary scanner is more commonly
# exercised against a standard python.org installer's layout. Fix: bundle
# them explicitly rather than relying on auto-detection, glob-matched (not
# a hardcoded exact filename) so a future OpenSSL version bump (e.g.
# libssl-3 -> libssl-4) doesn't silently break this again. Explicitly
# Windows-only: macOS Python builds link against the system/Homebrew
# libssl differently, and PyInstaller's macOS dependency walker has not
# shown this specific gap (untested at scale, but sys.base_prefix/DLLs is a
# Windows-only directory convention regardless, so this must be guarded).
ssl_binaries = []
if is_win:
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

# macOS .app bundle (2026-07-14): EXE() alone produces a bare Mach-O
# executable, not a real .app -- fine on Windows, but on macOS
# QWebEngineView's separate QtWebEngineProcess helper locates its sibling
# Frameworks/Resources via rpaths that assume the standard
# Contents/MacOS + Contents/Frameworks bundle layout. Without that layout
# the helper process fails to launch silently (the failure is inside the
# Chromium subprocess, invisible to gui_main.py's own exception handling),
# leaving every QWebEngineView-based 3D view (all 3Dmol.js panels,
# including the landscape ensemble overlay) blank while the rest of the
# Qt-widget UI keeps working normally -- matches the reported symptom
# exactly. BUNDLE() wraps EXE() into a proper ALMA.app; only meaningful
# on macOS, so guarded here rather than applied unconditionally.
if sys.platform == "darwin":
    app = BUNDLE(
        exe,
        name="ALMA.app",
        bundle_identifier="dev.noira913.alma",
        info_plist={"NSHighResolutionCapable": "True"},
    )
