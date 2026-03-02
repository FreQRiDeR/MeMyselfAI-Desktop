#!/usr/bin/env bash
# Build Ubuntu/Debian package for MeMyselfAI.

set -euo pipefail

APP_NAME="MeMyselfAI"
PKG_NAME="memyselfai"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPEC_FILE="$ROOT_DIR/MeMyselfAi_Linux.spec"
ICON_SRC="$ROOT_DIR/MeMyselfAi.png"
DESKTOP_SRC="$ROOT_DIR/packaging/linux/memyselfai.desktop"

if [[ ! -f "$SPEC_FILE" ]]; then
  echo "Missing spec file: $SPEC_FILE"
  exit 1
fi

if [[ ! -f "$ICON_SRC" ]]; then
  echo "Missing icon file: $ICON_SRC"
  exit 1
fi

if [[ ! -f "$DESKTOP_SRC" ]]; then
  echo "Missing desktop file: $DESKTOP_SRC"
  exit 1
fi

if ! command -v pyinstaller >/dev/null 2>&1; then
  echo "pyinstaller is required but not found in PATH."
  exit 1
fi

if ! command -v dpkg-deb >/dev/null 2>&1; then
  echo "dpkg-deb is required but not found in PATH."
  echo "Run this on Ubuntu/Debian (or in a Debian-based container/VM)."
  exit 1
fi

DEFAULT_VERSION="$(awk -F'"' '/^version = / { print $2; exit }' "$ROOT_DIR/pyproject.toml" || true)"
VERSION="${1:-${DEFAULT_VERSION:-0.1.0}}"
ARCH="${2:-$(dpkg --print-architecture 2>/dev/null || echo amd64)}"

DIST_DIR="$ROOT_DIR/dist"
BUILD_ROOT="$ROOT_DIR/build/deb"
STAGE_DIR="$BUILD_ROOT/${PKG_NAME}_${VERSION}_${ARCH}"
APP_INSTALL_DIR="$STAGE_DIR/opt/${PKG_NAME}"
DEBIAN_DIR="$STAGE_DIR/DEBIAN"
OUTPUT_DEB="$DIST_DIR/${PKG_NAME}_${VERSION}_${ARCH}.deb"

echo "Building ${PKG_NAME} ${VERSION} (${ARCH})"
echo "Root: $ROOT_DIR"

echo "Cleaning previous outputs..."
rm -rf "$STAGE_DIR" "$DIST_DIR/$APP_NAME" "$OUTPUT_DEB"
mkdir -p "$DIST_DIR" "$BUILD_ROOT"

echo "Running PyInstaller..."
pyinstaller "$SPEC_FILE" --clean --noconfirm --distpath "$DIST_DIR" --workpath "$ROOT_DIR/build/pyinstaller"

if [[ ! -d "$DIST_DIR/$APP_NAME" ]]; then
  echo "PyInstaller output missing: $DIST_DIR/$APP_NAME"
  exit 1
fi

echo "Staging package filesystem..."
mkdir -p "$APP_INSTALL_DIR" "$DEBIAN_DIR" "$STAGE_DIR/usr/bin"
cp -a "$DIST_DIR/$APP_NAME/." "$APP_INSTALL_DIR/"
chmod 0755 "$APP_INSTALL_DIR/$APP_NAME" || true

cat >"$STAGE_DIR/usr/bin/$PKG_NAME" <<EOF
#!/usr/bin/env bash
exec /opt/${PKG_NAME}/${APP_NAME} "\$@"
EOF
chmod 0755 "$STAGE_DIR/usr/bin/$PKG_NAME"

install -Dm644 "$DESKTOP_SRC" "$STAGE_DIR/usr/share/applications/${PKG_NAME}.desktop"
install -Dm644 "$ICON_SRC" "$STAGE_DIR/usr/share/pixmaps/${PKG_NAME}.png"

# Install into common hicolor sizes so launchers can resolve the icon quickly.
for size in 512x512 256x256 128x128 64x64 48x48 32x32; do
  install -Dm644 "$ICON_SRC" "$STAGE_DIR/usr/share/icons/hicolor/${size}/apps/${PKG_NAME}.png"
done

cat >"$DEBIAN_DIR/control" <<EOF
Package: ${PKG_NAME}
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Maintainer: MeMyselfAI Team <noreply@memyselfai.local>
Depends: libglib2.0-0, libx11-6, libx11-xcb1, libxcb1, libxkbcommon-x11-0, libxcb-cursor0, libfontconfig1, libdbus-1-3, libgl1
Description: MeMyselfAI desktop app
 Local AI chat desktop application built with PyQt6 and bundled runtime.
EOF

cat >"$DEBIAN_DIR/postinst" <<'EOF'
#!/bin/sh
set -e

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database -q /usr/share/applications || true
fi

if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -q /usr/share/icons/hicolor || true
fi

exit 0
EOF
chmod 0755 "$DEBIAN_DIR/postinst"

cat >"$DEBIAN_DIR/postrm" <<'EOF'
#!/bin/sh
set -e

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database -q /usr/share/applications || true
fi

if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -q /usr/share/icons/hicolor || true
fi

exit 0
EOF
chmod 0755 "$DEBIAN_DIR/postrm"

echo "Building .deb..."
dpkg-deb --build --root-owner-group "$STAGE_DIR" "$OUTPUT_DEB"

echo ""
echo "Done:"
echo "  $OUTPUT_DEB"
echo ""
echo "Install with:"
echo "  sudo apt install ./$OUTPUT_DEB"
