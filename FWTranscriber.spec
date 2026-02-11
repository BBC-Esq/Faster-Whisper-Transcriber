# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for FWTranscriber
# This configuration creates a single .exe file (onefile mode)
# External folders cuda_lib/, models/ and config.yaml must be placed next to the .exe

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('qt.conf', '.'),          # Copy qt.conf for DPI awareness configuration
        # NOTE: cuda_lib/ and models/ are NOT included in datas - they must be external
    ],
    hiddenimports=[
        'ctranslate2',
        'faster_whisper',
        'PySide6',
        'nltk',
        'psutil',
        'pynput',
        'sounddevice',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,      # Include all binaries in the single exe
    a.zipfiles,      # Include zipfiles in the single exe
    a.datas,         # Include data files in the single exe
    [],
    exclude_binaries=False,  # ONEFILE mode: pack everything into one exe
    name='FWTranscriber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
