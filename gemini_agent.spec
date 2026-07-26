# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build spec for Gemini Agent CLI.

Usage:
    pip install pyinstaller
    pyinstaller gemini_agent.spec

Produces a standalone executable (dist/gemini_agent or dist/gemini_agent.exe
on Windows) that bundles Python + all dependencies — no separate Python
install needed to run it.

Notes on why this spec exists instead of a bare `pyinstaller cli.py`:
  - google-genai has internal dynamic imports PyInstaller's static analyzer
    can't always see on its own, so we explicitly collect it as hidden
    imports + data below.
  - config.py already handles the frozen-executable persistent-data-path
    issue (see config._get_persistent_base_dir) — no extra spec-level
    workaround is needed for that part.
"""
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

# google-genai (and its google.auth/google.api_core dependencies) use enough
# dynamic import/resolution internally that a plain PyInstaller run can miss
# submodules — collect_all pulls in everything (submodules, data files,
# binaries) for a package rather than relying on static analysis alone.
for pkg in ("google.genai", "google.auth", "google.api_core"):
    try:
        pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hiddenimports
    except Exception:
        pass  # if a package isn't installed (e.g. optional), skip it gracefully

# yaml is an optional dependency (only used by convert_file_format's
# json<->yaml conversion) — include it if present so that feature works in
# the built executable too, without hard-failing the build if it's missing.
try:
    collect_all("yaml")
    hiddenimports.append("yaml")
except Exception:
    pass

a = Analysis(
    ['cli.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='gemini_agent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # this is a CLI app — always show a console window on Windows
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
