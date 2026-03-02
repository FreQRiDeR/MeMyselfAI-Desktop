# -*- mode: python ; coding: utf-8 -*-
# MeMyselfAi_Linux.spec - PyInstaller configuration for Linux (Ubuntu)

from pathlib import Path

block_cipher = None

# Collect all backend and UI modules.
backend_files = [
    ("backend/llama_wrapper.py", "backend"),
    ("backend/config.py", "backend"),
    ("backend/model_manager.py", "backend"),
    ("backend/unified_backend.py", "backend"),
    ("backend/chat_history.py", "backend"),
    ("backend/system_prompts.py", "backend"),
]

ui_files = [
    ("ui/main_window.py", "ui"),
    ("ui/settings_dialog.py", "ui"),
    ("ui/model_manager_dialog.py", "ui"),
    ("ui/ollama_manager_dialog.py", "ui"),
    ("ui/system_prompts_dialog.py", "ui"),
]

data_files = []

# Include app logo so runtime UI loads correctly and packaging can reuse it.
linux_icon_path = None
for icon_candidate in ("MeMyselfAi.png", "./MeMyselfAi.png"):
    icon_obj = Path(icon_candidate)
    if icon_obj.exists():
        linux_icon_path = str(icon_obj.resolve())
        data_files.append((str(icon_obj), "."))
        break

# Bundle binaries - Linux executable names as committed in this repo.
binaries = []

llama_binary_path = Path("backend/bin/llama-server")
if llama_binary_path.exists():
    # Place where LlamaWrapper currently searches in frozen mode.
    binaries.append((str(llama_binary_path), "llama"))
    print(f"✅ Found llama-server at: {llama_binary_path}")
else:
    print(f"⚠️  WARNING: llama-server not found at: {llama_binary_path}")

ollama_binary_path = Path("backend/bin/ollama")
if ollama_binary_path.exists():
    # Place where UnifiedBackend._find_bundled_ollama() currently searches.
    binaries.append((str(ollama_binary_path), "backend/bin"))
    print(f"✅ Found ollama at: {ollama_binary_path}")
else:
    print(f"⚠️  WARNING: ollama not found at: {ollama_binary_path}")


a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=backend_files + ui_files + data_files,
    hiddenimports=[
        "PyQt6",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "PyQt6.sip",
    ],
    hookspath=[],
    hooksconfig={
        "PyQt6": {
            "plugins": ["platforms", "styles"],
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
    name="MeMyselfAI",
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
    icon=linux_icon_path,  # Ignored on Linux; kept for cross-platform compatibility.
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="MeMyselfAI",
)
