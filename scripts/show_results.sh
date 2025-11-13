#!/bin/bash

# Find the most recent results directory
RESULTS_DIR=$(ls -td results/*/ 2>/dev/null | head -1)

if [ -z "$RESULTS_DIR" ]; then
    echo "No results directory found!"
    exit 1
fi

echo "========================================="
echo "  EXPERIMENT RESULTS"
echo "========================================="
echo "Results directory: $RESULTS_DIR"
echo ""

# Count generated files
PNG_COUNT=$(ls -1 "$RESULTS_DIR"*.png 2>/dev/null | wc -l)
TXT_COUNT=$(ls -1 "$RESULTS_DIR"*.txt 2>/dev/null | wc -l)
H5_COUNT=$(ls -1 "$RESULTS_DIR"*.h5 2>/dev/null | wc -l)

echo "Generated files:"
echo "  • $PNG_COUNT visualization images (PNG)"
echo "  • $TXT_COUNT text reports (TXT)"
echo "  • $H5_COUNT trained models (H5)"
echo ""

echo "========================================="
echo "  VISUALIZATIONS GENERATED"
echo "========================================="
echo ""

# List all PNG files
if [ $PNG_COUNT -gt 0 ]; then
    echo "Per-Model Visualizations:"
    ls -1h "$RESULTS_DIR"*confusion_matrix.png 2>/dev/null | sed 's|.*/||' | nl
    echo ""
    ls -1h "$RESULTS_DIR"*roc_curve.png 2>/dev/null | sed 's|.*/||' | nl  
    echo ""
    ls -1h "$RESULTS_DIR"*attack_timeline.png 2>/dev/null | sed 's|.*/||' | nl
    echo ""
    
    echo "Comparison Visualizations:"
    ls -1h "$RESULTS_DIR"model_comparison.png 2>/dev/null | sed 's|.*/||' | nl
    ls -1h "$RESULTS_DIR"detection_heatmap.png 2>/dev/null | sed 's|.*/||' | nl
    ls -1h "$RESULTS_DIR"comprehensive_comparison.png 2>/dev/null | sed 's|.*/||' | nl
fi

echo ""
echo "========================================="
echo "  TEXT REPORTS"
echo "========================================="
ls -1h "$RESULTS_DIR"*.txt 2>/dev/null | sed 's|.*/||' | nl

echo ""
echo "========================================="
echo "  MODEL FILES"
echo "========================================="
ls -1h "$RESULTS_DIR"*.h5 2>/dev/null | sed 's|.*/||' | nl

echo ""
echo "========================================="
echo "  FILE SIZES"
echo "========================================="
du -sh "$RESULTS_DIR"*.png 2>/dev/null | sort -h
echo ""
du -sh "$RESULTS_DIR"*.h5 2>/dev/null | sort -h

echo ""
echo "========================================="
echo "To view the visualizations, open:"
echo "$RESULTS_DIR"
echo "========================================="
