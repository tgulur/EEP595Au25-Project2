"""
Generate comparison table from existing experiment results
"""
import logging
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def parse_report_file(filepath):
    """Parse a report file to extract metrics"""
    metrics = {}
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract metrics using regex
    metric_patterns = {
        'accuracy': r'Accuracy:\s+([\d.]+)',
        'precision': r'Precision:\s+([\d.]+)',
        'recall': r'Recall:\s+([\d.]+)',
        'f1_score': r'F1-Score:\s+([\d.]+)',
        'tpr': r'True Positive Rate \(TPR\):\s+([\d.]+)',
        'fpr': r'False Positive Rate \(FPR\):\s+([\d.]+)',
        'mean_latency_ms': r'Mean:\s+([\d.]+)\s+ms'
    }
    
    for key, pattern in metric_patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[key] = float(match.group(1))
        else:
            metrics[key] = 0.0
    
    return metrics

def generate_comparison_table(results_dir):
    """Generate comparison table from experiment results"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return
    
    # Load all model results
    all_models = []
    
    # Baseline models
    baseline_files = {
        'Timing-Based IDS': 'timing_report.txt',
        'Frequency-Based IDS': 'frequency_report.txt'
    }
    
    for model_name, filename in baseline_files.items():
        filepath = results_path / filename
        if filepath.exists():
            metrics = parse_report_file(filepath)
            all_models.append((model_name, metrics, 'Baseline'))
    
    # Voltage fingerprinting
    voltage_file = results_path / 'voltage_report.txt'
    if voltage_file.exists():
        metrics = parse_report_file(voltage_file)
        all_models.append(('Voltage Fingerprinting', metrics, 'Feature-Based'))
    
    # Deep learning models
    dl_files = {
        'CNN': 'cnn_report.txt',
        'LSTM': 'lstm_report.txt'
    }
    
    for model_name, filename in dl_files.items():
        filepath = results_path / filename
        if filepath.exists():
            metrics = parse_report_file(filepath)
            all_models.append((model_name, metrics, 'Deep Learning'))
    
    # Fusion
    fusion_file = results_path / 'fusion_report.txt'
    if fusion_file.exists():
        metrics = parse_report_file(fusion_file)
        all_models.append(('Fusion Layer', metrics, 'Fusion'))
    
    if not all_models:
        logger.error("No model results found")
        return
    
    # Create formatted table
    table_lines = []
    table_lines.append("")
    table_lines.append("="*120)
    table_lines.append("COMPREHENSIVE MODEL COMPARISON TABLE")
    table_lines.append("="*120)
    header = f"{'Model':<25} {'Category':<15} {'Accuracy':>10} {'TPR':>8} {'FPR':>8} {'Precision':>10} {'Recall':>8} {'F1-Score':>10} {'Latency (ms)':>12}"
    table_lines.append(header)
    table_lines.append("="*120)
    
    for model_name, metrics, category in all_models:
        latency = metrics.get('mean_latency_ms', 0.0)
        latency_str = f"{latency:.2f}" if latency > 0 else "N/A"
        
        line = (f"{model_name:<25} {category:<15} "
               f"{metrics.get('accuracy', 0.0):>10.4f} "
               f"{metrics.get('tpr', metrics.get('recall', 0.0)):>8.4f} "
               f"{metrics.get('fpr', 0.0):>8.4f} "
               f"{metrics.get('precision', 0.0):>10.4f} "
               f"{metrics.get('recall', 0.0):>8.4f} "
               f"{metrics.get('f1_score', 0.0):>10.4f} "
               f"{latency_str:>12}")
        table_lines.append(line)
    
    table_lines.append("="*120)
    table_lines.append("")
    table_lines.append("NOTES:")
    table_lines.append("  - TPR (True Positive Rate): Percentage of attacks correctly detected")
    table_lines.append("  - FPR (False Positive Rate): Percentage of normal traffic incorrectly flagged as attack")
    table_lines.append("  - Latency: Mean inference time per sample (N/A for non-DL models)")
    table_lines.append("")
    table_lines.append("KEY FINDINGS:")
    
    # Find best models
    best_accuracy = max(all_models, key=lambda x: x[1].get('accuracy', 0.0))
    best_tpr = max(all_models, key=lambda x: x[1].get('tpr', x[1].get('recall', 0.0)))
    lowest_fpr = min(all_models, key=lambda x: x[1].get('fpr', 1.0))
    
    table_lines.append(f"  - Best Accuracy: {best_accuracy[0]} ({best_accuracy[1].get('accuracy', 0.0):.4f})")
    table_lines.append(f"  - Best TPR (Attack Detection): {best_tpr[0]} ({best_tpr[1].get('tpr', best_tpr[1].get('recall', 0.0)):.4f})")
    table_lines.append(f"  - Lowest FPR (Fewest False Alarms): {lowest_fpr[0]} ({lowest_fpr[1].get('fpr', 0.0):.4f})")
    table_lines.append("")
    
    # Print to console
    for line in table_lines:
        logger.info(line)
    
    # Save to file
    comparison_file = results_path / "model_comparison_table.txt"
    with open(comparison_file, 'w') as f:
        f.write('\n'.join(table_lines))
    
    logger.info(f"Comparison table saved to: {comparison_file}")

if __name__ == "__main__":
    # Use the most recent results directory
    results_dir = "results/20251112_232232"
    generate_comparison_table(results_dir)
