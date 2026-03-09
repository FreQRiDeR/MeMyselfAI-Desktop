# -*- mode: python ; coding: utf-8 -*-
# MeMyselfAI.spec - PyInstaller configuration

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# Collect all backend and ui modules
backend_files = [
    ('backend/llama_wrapper.py', 'backend'),
    ('backend/config.py', 'backend'),
    ('backend/model_manager.py', 'backend'),
    ('backend/unified_backend.py', 'backend'),
    ('backend/chat_history.py', 'backend'),
    ('backend/system_prompts.py', 'backend'),
]

ui_files = [
    ('ui/main_window.py', 'ui'),
    ('ui/settings_dialog.py', 'ui'),
    ('ui/model_manager_dialog.py', 'ui'),
    ('ui/ollama_manager_dialog.py', 'ui'),
    ('ui/system_prompts_dialog.py', 'ui'),
]

# Include logo if it exists
data_files = []
if Path('MeMyselfAi.png').exists():
    data_files.append(('MeMyselfAi.png', '.'))

# Include icon - comprehensive search with debug output
import os
print("\n" + "=" * 60)
print("ICON DETECTION")
print("=" * 60)
print(f"Current working directory: {Path.cwd()}")
print(f"Spec file directory: {Path(__file__).parent if '__file__' in dir() else 'N/A'}")
print(f"OS getcwd: {os.getcwd()}")
print()

icon_path = None
possible_icon_paths = [
    'MeMyselfAi.icns',
    './MeMyselfAi.icns',
    str(Path.cwd() / 'MeMyselfAi.icns'),
]

# Add spec file directory if available
if '__file__' in dir():
    spec_dir = Path(__file__).parent
    possible_icon_paths.append(str(spec_dir / 'MeMyselfAi.icns'))

print("Searching for icon in:")
for i, icon_candidate in enumerate(possible_icon_paths, 1):
    icon_path_obj = Path(icon_candidate)
    exists = icon_path_obj.exists()
    print(f"  {i}. {'✅' if exists else '❌'} {icon_candidate}")
    if exists:
        print(f"      → Absolute path: {icon_path_obj.absolute()}")
        print(f"      → Size: {icon_path_obj.stat().st_size} bytes")
    
    if exists and not icon_path:
        icon_path = str(icon_path_obj.absolute())

print()
if icon_path:
    print(f"✅ USING ICON: {icon_path}")
else:
    print("⚠️  WARNING: No icon found!")
    print()
    print("Icon files in current directory:")
    for icns in Path.cwd().glob('**/*.icns'):
        print(f"  - {icns.relative_to(Path.cwd())}")
    if not list(Path.cwd().glob('**/*.icns')):
        print("  (none)")
        
print("=" * 60 + "\n")

# Bundle binaries - paths are relative to the project root
binaries = []

llama_binary_path = 'backend/bin/llama-server'
if Path(llama_binary_path).exists():
    # Destination 'backend/bin' matches _find_bundled_ollama()'s search path
    binaries.append((llama_binary_path, 'backend/bin'))
    print(f"✅ Found llama-server at: {llama_binary_path}")
else:
    print(f"⚠️  WARNING: llama-server not found at: {llama_binary_path}")

ollama_binary_path = 'backend/bin/ollama'
if Path(ollama_binary_path).exists():
    binaries.append((ollama_binary_path, 'backend/bin'))
    print(f"✅ Found ollama at: {ollama_binary_path}")
else:
    print(f"⚠️  WARNING: ollama not found at: {ollama_binary_path}")

# Collect networking deps used by requests, including optional charset/chardet backends.
extra_datas = []
extra_binaries = []
extra_hiddenimports = []
for pkg_name in (
    'requests',
    'urllib3',
    'idna',
    'certifi',
    'charset_normalizer',
    'chardet',
):
    try:
        d, b, h = collect_all(pkg_name)
        extra_datas.extend(d)
        extra_binaries.extend(b)
        extra_hiddenimports.extend(h)
    except Exception as exc:
        print(f"⚠️  WARNING: collect_all({pkg_name}) failed: {exc}")

try:
    extra_hiddenimports.extend(collect_submodules('charset_normalizer'))
except Exception as exc:
    print(f"⚠️  WARNING: collect_submodules(charset_normalizer) failed: {exc}")


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries + extra_binaries,
    datas=backend_files + ui_files + data_files + extra_datas,
    hiddenimports=list(dict.fromkeys([
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.sip',
        'chardet',
        'charset_normalizer',
        'charset_normalizer.api',
        'charset_normalizer.md',
        'charset_normalizer.cd',
        'requests',
    ] + extra_hiddenimports)),
    hookspath=[],
    hooksconfig={
        'PyQt6': {
            'plugins': ['platforms', 'styles'],
        },
    },
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
    [],
    exclude_binaries=True,
    name='MeMyselfAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MeMyselfAI',
)

app = BUNDLE(
    coll,
    name='MeMyselfAI.app',
    icon=icon_path,  # Use the detected icon path
    bundle_identifier='com.memyselfai.app',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': 'True',
        'CFBundleName': 'MeMyselfAI',
        'CFBundleDisplayName': 'MeMyselfAI',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
    },
)
