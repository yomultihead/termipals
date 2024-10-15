#!/bin/bash

# Exit on error
set -e

echo "🚀 Installing Termipals..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip in the virtual environment
echo "🔄 Upgrading pip..."
python3 -m pip install --upgrade pip

# Install PyTorch first for macOS
echo "📥 Installing PyTorch..."
pip3 install torch

# Install package in editable mode
echo "📥 Installing other dependencies..."
python3 -m pip install -e .

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p llm/models
mkdir -p termipals/assets/animals

echo "✨ Installation complete!"
echo
echo "To start using Termipals:"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo
echo "2. Try generating your first animal:"
echo "   termipals --create elephant"
echo
echo "3. List available animals:"
echo "   termipals -l"
echo
echo "Enjoy your terminal companions! 🐮"