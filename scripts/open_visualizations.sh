#!/bin/bash
#
# Open All Visualizations - Quick viewer for all experiment results
#

LATEST_DIR=$(ls -dt results/202* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "❌ No results found!"
    exit 1
fi

echo "================================================================================"
echo "                    OPENING ALL VISUALIZATIONS"
echo "================================================================================"
echo ""
echo "Results from: $LATEST_DIR"
echo ""

# Check if running in a graphical environment
if [ -z "$DISPLAY" ]; then
    echo "⚠️  No graphical display detected."
    echo "Listing visualization files instead:"
    echo ""
    ls -lh "$LATEST_DIR"/*.png | awk '{printf "  📊 %-40s %8s\n", $9, $5}' | sed 's/.*\///'
    echo ""
    echo "To view these files, copy them to your local machine or use a graphical session."
    exit 0
fi

# Try to find an image viewer
VIEWER=""
for cmd in eog feh display xdg-open; do
    if command -v $cmd &> /dev/null; then
        VIEWER=$cmd
        break
    fi
done

if [ -z "$VIEWER" ]; then
    echo "⚠️  No image viewer found."
    echo "Install one of: eog, feh, imagemagick (display), or xdg-utils"
    echo ""
    echo "Available visualizations:"
    ls -1 "$LATEST_DIR"/*.png
    exit 1
fi

echo "Using viewer: $VIEWER"
echo ""

# Open visualizations by category
echo "📊 Opening Confusion Matrices..."
for f in "$LATEST_DIR"/*confusion_matrix.png; do
    [ -f "$f" ] && $VIEWER "$f" &
done
sleep 1

echo "📈 Opening ROC Curves..."
for f in "$LATEST_DIR"/*roc_curve.png; do
    [ -f "$f" ] && $VIEWER "$f" &
done
sleep 1

echo "🕐 Opening Attack Timelines..."
for f in "$LATEST_DIR"/*attack_timeline.png; do
    [ -f "$f" ] && $VIEWER "$f" &
done
sleep 1

echo "🔥 Opening Comparative Visualizations..."
for f in "$LATEST_DIR"/detection_heatmap.png "$LATEST_DIR"/model_comparison.png "$LATEST_DIR"/comprehensive_comparison.png; do
    [ -f "$f" ] && $VIEWER "$f" &
done

echo ""
echo "✅ All visualizations opened!"
echo ""
echo "================================================================================"
