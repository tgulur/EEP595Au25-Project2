#!/bin/bash
# setup_project.sh - Complete project setup script

set -e

echo "=============================================="
echo "CAN IDS Project Setup"
echo "=============================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "✓ Python packages installed"

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p data/raw/canmap_voltage
mkdir -p data/raw/road_can_ids
mkdir -p data/processed
mkdir -p models
mkdir -p results
mkdir -p logs
echo "✓ Directories created"

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x scripts/setup_vcan.sh
chmod +x scripts/generate_can_traffic.py
echo "✓ Scripts are executable"

# Check for CAN utils
echo ""
echo "Checking for CAN utilities..."
if command -v candump &> /dev/null; then
    echo "✓ CAN utilities found"
else
    echo "⚠ CAN utilities not found"
    echo "To install: sudo apt-get install can-utils"
fi

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Download datasets and place in data/raw/"
echo "3. Setup virtual CAN (requires sudo): sudo bash scripts/setup_vcan.sh"
echo "4. Run experiments: python main_experiment.py"
echo ""
echo "For interactive exploration, use:"
echo "  jupyter notebook notebooks/demo_notebook.ipynb"
echo ""
echo "=============================================="
