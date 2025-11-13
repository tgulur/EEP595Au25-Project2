#!/bin/bash
# setup_vcan.sh - Setup virtual CAN bus for testing and simulation

set -e

echo "==============================================="
echo "Setting up Virtual CAN (vcan) Interface"
echo "==============================================="

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root or with sudo"
    exit 1
fi

# Load CAN kernel modules
echo "Loading CAN kernel modules..."
modprobe can
modprobe can_raw
modprobe vcan

# Create virtual CAN interface
VCAN_INTERFACE=${1:-vcan0}
echo "Creating virtual CAN interface: $VCAN_INTERFACE"

# Remove interface if it already exists
ip link delete $VCAN_INTERFACE 2>/dev/null || true

# Create new virtual CAN interface
ip link add dev $VCAN_INTERFACE type vcan
ip link set up $VCAN_INTERFACE

# Verify interface is up
if ip link show $VCAN_INTERFACE &> /dev/null; then
    echo "✓ Virtual CAN interface $VCAN_INTERFACE is UP"
    ip -details link show $VCAN_INTERFACE
else
    echo "✗ Failed to create virtual CAN interface"
    exit 1
fi

echo ""
echo "==============================================="
echo "Setup Complete!"
echo "==============================================="
echo "You can now use $VCAN_INTERFACE for CAN bus simulation"
echo ""
echo "Useful commands:"
echo "  - View CAN traffic: candump $VCAN_INTERFACE"
echo "  - Send CAN message: cansend $VCAN_INTERFACE 123#DEADBEEF"
echo "  - Generate traffic: python scripts/generate_can_traffic.py"
echo ""
echo "To remove the interface: sudo ip link delete $VCAN_INTERFACE"
echo "==============================================="
