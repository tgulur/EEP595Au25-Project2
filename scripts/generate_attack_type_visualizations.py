#!/usr/bin/env python3
"""
Generate attack type comparison visualizations from existing results

This script can be run standalone to generate visualizations from
existing attack_type_report.txt files.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from evaluation_metrics import IDSEvaluator


def load_attack_type_reports(results_dir: Path) -> dict:
    """Load attack type reports from a results directory"""
    attack_type_results = {}
    
    model_dirs = {
        'CNN': results_dir / 'cnn',
        'LSTM': results_dir / 'lstm',
        'Fusion': results_dir / 'fusion'
    }
    
    evaluator = IDSEvaluator(save_dir=str(results_dir / 'comparison'))
    
    for model_name, model_dir in model_dirs.items():
        report_file = model_dir / 'attack_type_report.txt'
        if report_file.exists():
            with open(report_file, 'r') as f:
                report_text = f.read()
            
            # Parse the report
            parsed_results = evaluator._parse_attack_type_report(report_text)
            attack_type_results[model_name] = parsed_results
            print(f"✓ Loaded {model_name} attack type report: {len(parsed_results)} attack types")
        else:
            print(f"⚠ {model_name} attack type report not found: {report_file}")
    
    return attack_type_results


def main():
    parser = argparse.ArgumentParser(
        description='Generate attack type comparison visualizations'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Path to results directory (default: latest)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: results_dir/comparison)'
    )
    
    args = parser.parse_args()
    
    # Find results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Find latest results directory
        results_base = Path("results")
        experiment_dirs = [d for d in results_base.iterdir() 
                          if d.is_dir() and d.name.startswith("202")]
        if not experiment_dirs:
            print("Error: No results directories found!")
            sys.exit(1)
        
        results_dir = sorted(experiment_dirs)[-1]
        print(f"Using latest results directory: {results_dir}")
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Load attack type reports
    print(f"\nLoading attack type reports from {results_dir}...")
    attack_type_results = load_attack_type_reports(results_dir)
    
    if not attack_type_results:
        print("Error: No attack type reports found!")
        sys.exit(1)
    
    # Generate visualization
    output_dir = args.output_dir if args.output_dir else results_dir / 'comparison'
    evaluator = IDSEvaluator(save_dir=str(output_dir))
    
    print(f"\nGenerating attack type comparison visualization...")
    evaluator.plot_attack_type_comparison(
        attack_type_results,
        save_name="attack_type_comparison.png"
    )
    
    print(f"\n✓ Visualization saved to {output_dir / 'attack_type_comparison.png'}")
    print("\nDone!")


if __name__ == "__main__":
    main()

