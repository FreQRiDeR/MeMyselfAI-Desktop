#!/bin/bash
# build_app.sh - Build MeMyselfAI.app with PyInstaller

set -e

echo "🔨 Building MeMyselfAI.app with PyInstaller"
echo "============================================"
echo ""

USE_UV=0
if command -v uv &> /dev/null; then
    USE_UV=1
fi

if [ "$USE_UV" -eq 1 ]; then
    echo "📦 Syncing dependencies with uv..."
    uv sync
else
    # Fallback to pip environment if uv is unavailable.
    echo "📦 Installing dependencies with pip..."
    pip3 install -r requirements.txt
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build dist

# Build the app
echo "🔨 Building application..."
if [ "$USE_UV" -eq 1 ]; then
    uv run pyinstaller MeMyselfAI.spec --clean --noconfirm
else
    pyinstaller MeMyselfAI.spec --clean --noconfirm
fi

# Check if build was successful
if [ -d "dist/MeMyselfAI.app" ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    
    # Verify bundled binaries
    if [ -f "dist/MeMyselfAI.app/Contents/Frameworks/backend/bin/llama-server" ]; then
        echo "✅ llama-server is bundled"
    else
        echo "⚠️  WARNING: llama-server NOT found in bundle!"
    fi

    if [ -f "dist/MeMyselfAI.app/Contents/Frameworks/backend/bin/ollama" ]; then
        echo "✅ ollama is bundled"
    else
        echo "⚠️  WARNING: ollama NOT found in bundle!"
    fi
    
    echo ""
    echo "📦 Your app is ready at: dist/MeMyselfAI.app"
    echo ""
    echo "To test it:"
    echo "  open dist/MeMyselfAI.app"
    echo ""
    echo "To move it to Applications:"
    echo "  cp -r dist/MeMyselfAI.app /Applications/"
    echo ""
else
    echo ""
    echo "❌ Build failed!"
    echo "Check the output above for errors."
    exit 1
fi
