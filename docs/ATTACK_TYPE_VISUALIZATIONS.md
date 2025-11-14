# Attack Type Performance Visualizations

## Overview

The system now generates comprehensive visualizations showing how each model performs across different attack types. This helps you understand:
- Which attack types are detected best by each model
- Which models are strongest for specific attack types
- Trade-offs between detection rate (TPR) and false alarms (FPR)

## Visualization Types

### 1. Individual Model Performance (`attack_type_performance.png`)

**Location**: Each model's folder (e.g., `cnn/`, `lstm/`, `fusion/`)

**Content**: 2x2 grid showing:
- **Top Left**: TPR (True Positive Rate) by attack type - How well the model detects each attack
- **Top Right**: FPR (False Positive Rate) by attack type - False alarm rate for each attack type
- **Bottom Left**: Accuracy and F1-Score comparison by attack type
- **Bottom Right**: Test set distribution (sample counts) by attack type

**Use Case**: Quick overview of a single model's capabilities across attack types

### 2. Cross-Model Comparison (`attack_type_comparison.png`)

**Location**: `comparison/` folder

**Content**: 3x3 grid with 9 comprehensive visualizations:

1. **TPR by Attack Type** (Bar Chart) - Compare detection rates across models
2. **FPR by Attack Type** (Bar Chart) - Compare false alarm rates across models
3. **Accuracy by Attack Type** (Bar Chart) - Compare accuracy across models
4. **TPR Heatmap** - Visual heatmap showing TPR for each model-attack combination
5. **FPR Heatmap** - Visual heatmap showing FPR for each model-attack combination
6. **Accuracy Heatmap** - Visual heatmap showing accuracy for each model-attack combination
7. **TPR vs FPR Scatter** - Trade-off analysis with attack type labels
8. **F1-Score by Attack Type** - Balanced performance metric comparison
9. **Sample Distribution** - Test set composition by attack type

**Use Case**: Comprehensive comparison of all models across all attack types

## Generated Files

After running an experiment, you'll find:

```
results/YYYYMMDD_HHMMSS/
├── cnn/
│   ├── attack_type_report.txt          # Text report
│   └── attack_type_performance.png     # Individual visualization ⭐
├── lstm/
│   ├── attack_type_report.txt
│   └── attack_type_performance.png     # Individual visualization ⭐
├── fusion/
│   ├── attack_type_report.txt
│   └── attack_type_performance.png     # Individual visualization ⭐
└── comparison/
    └── attack_type_comparison.png      # Cross-model comparison ⭐
```

## How to Read the Visualizations

### TPR (True Positive Rate)
- **Higher is better** (1.0 = perfect detection)
- Shows what percentage of attacks are detected
- Example: TPR of 0.95 means 95% of attacks detected

### FPR (False Positive Rate)
- **Lower is better** (0.0 = no false alarms)
- Shows what percentage of normal traffic is flagged as attack
- Example: FPR of 0.05 means 5% false alarm rate

### Accuracy
- Overall correctness for each attack type
- Combines both TPR and FPR considerations

### Heatmaps
- **Green** = Good performance (high TPR/Accuracy, low FPR)
- **Red** = Poor performance (low TPR/Accuracy, high FPR)
- **Yellow** = Moderate performance

## Example Insights

From the visualizations, you can quickly see:

1. **Which model is best for each attack type**
   - Look at TPR heatmap - darker green = better detection
   
2. **Which attack types are hardest to detect**
   - Look for lower TPR values across all models
   
3. **Which models have false alarm issues**
   - Look at FPR heatmap - darker red = more false alarms
   
4. **Overall best model**
   - Look at accuracy heatmap - consistently green = best overall

## Standalone Script

You can also generate visualizations from existing results:

```bash
# Generate from latest results
python scripts/generate_attack_type_visualizations.py

# Generate from specific results directory
python scripts/generate_attack_type_visualizations.py --results-dir results/20251113_142805
```

## Automatic Generation

These visualizations are **automatically generated** during each experiment run:
- Individual model visualizations are created after each model evaluation
- Cross-model comparison is created during the comparison phase

No manual steps required!

---

*Feature added: 2025-11-13*
*Available in: All experiment runs*

