#!/bin/bash
#
# Results Viewer - Display experiment results and visualizations
#

# Find the latest results directory
LATEST_DIR=$(ls -dt results/202* 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "No results found!"
    exit 1
fi

echo "================================================================================"
echo "                    EXPERIMENT RESULTS SUMMARY"
echo "================================================================================"
echo ""
echo "Results Directory: $LATEST_DIR"
echo "Generated: $(stat -c '%y' "$LATEST_DIR" | cut -d'.' -f1)"
echo ""

# Count files
PNG_COUNT=$(find "$LATEST_DIR" -name "*.png" | wc -l)
TXT_COUNT=$(find "$LATEST_DIR" -name "*.txt" | wc -l)
H5_COUNT=$(find "$LATEST_DIR" -name "*.h5" | wc -l)

echo "Files Generated:"
echo "  • $PNG_COUNT Visualizations"
echo "  • $TXT_COUNT Evaluation Reports"
echo "  • $H5_COUNT Trained Models"
echo ""

echo "================================================================================"
echo "                        MODEL PERFORMANCE SUMMARY"
echo "================================================================================"
echo ""

# Extract key metrics from reports
for model in voltage cnn lstm fusion; do
    REPORT="$LATEST_DIR/${model}_report.txt"
    if [ -f "$REPORT" ]; then
        MODEL_NAME=$(grep "Evaluation Report:" "$REPORT" | cut -d':' -f2 | xargs)
        ACCURACY=$(grep "Accuracy:" "$REPORT" | head -1 | awk '{print $2}')
        PRECISION=$(grep "Precision:" "$REPORT" | head -1 | awk '{print $2}')
        RECALL=$(grep "Recall:" "$REPORT" | head -1 | awk '{print $2}')
        F1=$(grep "F1-Score:" "$REPORT" | head -1 | awk '{print $2}')
        ROC_AUC=$(grep "ROC AUC:" "$REPORT" | awk '{print $2}')
        
        printf "%-25s | Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f\n" \
               "$MODEL_NAME" "$ACCURACY" "$PRECISION" "$RECALL" "$F1" "$ROC_AUC"
    fi
done

echo ""
echo "================================================================================"
echo "                           GENERATED VISUALIZATIONS"
echo "================================================================================"
echo ""

# List all visualizations
echo "📊 Confusion Matrices:"
ls -1 "$LATEST_DIR"/*confusion_matrix.png | sed 's/.*\//  • /'

echo ""
echo "📈 ROC Curves:"
ls -1 "$LATEST_DIR"/*roc_curve.png | sed 's/.*\//  • /'

echo ""
echo "🕐 Attack Detection Timelines:"
ls -1 "$LATEST_DIR"/*attack_timeline.png | sed 's/.*\//  • /'

echo ""
echo "🔥 Additional Visualizations:"
ls -1 "$LATEST_DIR"/detection_heatmap.png "$LATEST_DIR"/model_comparison.png "$LATEST_DIR"/comprehensive_comparison.png 2>/dev/null | sed 's/.*\//  • /'

echo ""
echo "================================================================================"
echo "                              TRAINED MODELS"
echo "================================================================================"
echo ""

ls -lh "$LATEST_DIR"/*.h5 2>/dev/null | awk '{printf "  • %-25s %8s\n", $9, $5}' | sed 's/.*\///'

echo ""
echo "================================================================================"
echo ""
echo "To view visualizations, open the PNG files in: $LATEST_DIR/"
echo "To load trained models, use the .h5 files in: $LATEST_DIR/"
echo ""
echo "================================================================================"
