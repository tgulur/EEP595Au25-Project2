"""Evaluation helpers.

This module provides small wrapper functions to centralize common evaluation
workflows used by the experiment runner and trainers. The functions delegate
to `IDSEvaluator` methods (plotting and metric calculations) but keep the
experiment orchestration code concise and testable.
"""
from pathlib import Path
from typing import Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


def evaluate_model(evaluator, y_true: np.ndarray, predictions: np.ndarray, scores: np.ndarray,
                   model_name: str, save_prefix: str = None) -> Dict[str, Any]:
    """Calculate metrics, produce plots and a text report for a single model.

    Args:
        evaluator: IDSEvaluator instance
        y_true: ground-truth labels
        predictions: integer predictions
        scores: confidence or score values (float)
        model_name: friendly model name for titles and filenames
        save_prefix: optional directory prefix under results to save plots/reports

    Returns:
        metrics dict returned by `evaluator.calculate_metrics`
    """
    save_prefix = (save_prefix or model_name.lower()).rstrip('/')

    metrics = evaluator.calculate_metrics(y_true, predictions, scores)

    # Visualizations
    evaluator.plot_confusion_matrix(
        y_true, predictions,
        title=f"{model_name} - Confusion Matrix",
        save_name=f"{save_prefix}/{model_name.lower()}_confusion_matrix.png"
    )

    evaluator.plot_roc_curve(
        y_true, scores,
        title=f"{model_name} - ROC Curve",
        save_name=f"{save_prefix}/{model_name.lower()}_roc_curve.png"
    )

    # Report
    report = evaluator.generate_report(
        model_name,
        metrics,
        save_name=f"{save_prefix}/{model_name.lower()}_report.txt"
    )

    logger.info(report)
    return metrics


def compare_and_visualize(evaluator, results: Dict[str, Any], results_dir: str) -> None:
    """Generate cross-model comparison visualizations and a comparison table.

    This wraps the higher-level comparison orchestration previously in
    `main_experiment.py` so the runner can call it.
    """
    results_dir = Path(results_dir)
    logger.info("\n" + "=" * 60)
    logger.info("Generating Model Comparisons")
    logger.info("=" * 60)

    # Collect metric summaries
    comparison_results = {
        'Voltage': results['voltage']['metrics'],
        'CNN': results['deep_learning']['CNN']['metrics'],
        'LSTM': results['deep_learning']['LSTM']['metrics'],
        'Fusion': results['fusion']['metrics']
    }

    # Include baselines if present
    if 'baselines' in results:
        for baseline_name, baseline_data in results['baselines'].items():
            comparison_results[f'{baseline_name}-IDS'] = baseline_data['metrics']

    # Comparison plot
    evaluator.compare_models(
        comparison_results,
        metric_names=['accuracy', 'precision', 'recall', 'f1_score'],
        save_name="comparison/model_comparison.png"
    )

    # Attack detection timelines for each model
    logger.info("Generating attack detection timelines...")
    timeline_models = []
    if 'voltage' in results:
        timeline_models.append(('Voltage', results['voltage']['predictions'], results['voltage']['y_test']))
    if 'deep_learning' in results:
        for name in ['CNN', 'LSTM']:
            timeline_models.append((name, results['deep_learning'][name]['predictions'], results['deep_learning'][name]['y_test']))
    if 'fusion' in results:
        timeline_models.append(('Fusion', results['fusion']['predictions'], results['fusion']['y_test']))

    for model_name, preds, y_test in timeline_models:
        evaluator.plot_attack_detection_timeline(
            y_test,
            preds,
            model_name=model_name,
            save_name=f"comparison/{model_name.lower()}_attack_timeline.png"
        )

    # Try to align predictions across models using any provided `test_indices`.
    aligned = _align_predictions(results)
    if aligned is not None and aligned.get('common_indices') is not None and len(aligned['common_indices']) > 0:
        # Use aligned arrays for heatmap and comprehensive dashboard
        ref_model = aligned.get('reference_model') or 'Voltage'
        y_test_ref = aligned['aligned'][ref_model]['y_test']

        # Build predictions dict for heatmap
        predictions_dict = {}
        for mname, mdata in aligned['aligned'].items():
            predictions_dict[mname] = mdata.get('predictions')

        try:
            evaluator.plot_detection_heatmap(
                y_test_ref,
                predictions_dict,
                save_name="comparison/detection_heatmap.png"
            )
        except Exception:
            logger.exception("Failed to plot detection heatmap from aligned predictions; falling back to per-model heatmap")

        # Comprehensive comparison
        comprehensive_results = {}
        for mname, mdata in aligned['aligned'].items():
            # Attempt to reuse metrics from the original results dict where available
            metrics = results.get(mname.lower(), {}).get('metrics') if isinstance(results.get(mname.lower()), dict) else None
            if metrics is None and 'deep_learning' in results and results['deep_learning'].get(mname):
                metrics = results['deep_learning'][mname].get('metrics')
            comprehensive_results[mname] = {
                'metrics': metrics or {},
                'predictions': mdata.get('predictions')
            }
        try:
            evaluator.plot_comprehensive_comparison(
                comprehensive_results,
                y_test_ref,
                save_name="comparison/comprehensive_comparison.png"
            )
        except Exception:
            logger.exception("Failed to plot comprehensive comparison from aligned predictions; skipping")
    else:
        # No alignment possible: preserve previous behavior (try heatmap with voltage ref)
        if 'voltage' in results:
            y_test_ref = results['voltage']['y_test']
            predictions_dict = {
                'Voltage': results['voltage']['predictions'],
                'CNN': results['deep_learning']['CNN']['predictions'],
                'LSTM': results['deep_learning']['LSTM']['predictions'],
            }

            try:
                evaluator.plot_detection_heatmap(
                    y_test_ref,
                    predictions_dict,
                    save_name="comparison/detection_heatmap.png"
                )
            except Exception:
                logger.exception("Failed to plot detection heatmap; skipping")

        # Fall back to length-check + skip behavior for comprehensive plot
        try:
            if 'voltage' in results:
                y_test_ref = results['voltage']['y_test']
                ref_len = len(y_test_ref)
                mismatch = False
                for mname, mdata in (('Voltage', results['voltage']),
                                     ('CNN', results['deep_learning']['CNN']),
                                     ('LSTM', results['deep_learning']['LSTM'])):
                    preds = mdata.get('predictions')
                    if preds is None:
                        continue
                    if len(preds) != ref_len:
                        mismatch = True
                        logger.warning(
                            "Skipping comprehensive comparison: length mismatch for %s (ref=%d, %s=%d)",
                            mname, ref_len, mname, len(preds)
                        )
                        break

                if not mismatch:
                    evaluator.plot_comprehensive_comparison(
                        {
                            'Voltage': {'metrics': results['voltage']['metrics'], 'predictions': results['voltage']['predictions']},
                            'CNN': {'metrics': results['deep_learning']['CNN']['metrics'], 'predictions': results['deep_learning']['CNN']['predictions']},
                            'LSTM': {'metrics': results['deep_learning']['LSTM']['metrics'], 'predictions': results['deep_learning']['LSTM']['predictions']}
                        },
                        y_test_ref,
                        save_name="comparison/comprehensive_comparison.png"
                    )
                else:
                    logger.warning("Comprehensive comparison plot skipped due to mismatched prediction lengths")
        except Exception:
            logger.exception("Failed while preparing comprehensive comparison plot; skipping")

    # Generate table
    try:
        _generate_comparison_table(results, results_dir)
    except Exception:
        logger.exception("Failed to generate comparison table")

    logger.info("All comparison visualizations complete")


def _align_predictions(results: Dict[str, Any]):
    """Attempt to align per-model predictions using provided `test_indices`.

    Returns None if alignment is not possible. Otherwise returns a dict with:
      - common_indices: np.ndarray of indices present in all models that provided indices
      - aligned: dict mapping model name -> {'y_test', 'predictions', 'scores'} aligned to common_indices
      - reference_model: model name used as reference (preference: Voltage)
    """
    # Gather models and their index arrays if present
    models = {}

    # Voltage
    if 'voltage' in results:
        v = results['voltage']
        models['Voltage'] = {
            'indices': np.array(v.get('test_indices')) if v.get('test_indices') is not None else None,
            'y_test': np.array(v.get('y_test')) if v.get('y_test') is not None else None,
            'predictions': np.array(v.get('predictions')) if v.get('predictions') is not None else None,
            'scores': np.array(v.get('scores')) if v.get('scores') is not None else None,
            'timestamps': np.array(v.get('test_timestamps')) if v.get('test_timestamps') is not None else None,
        }

    # Deep learning models
    if 'deep_learning' in results:
        for name in ['CNN', 'LSTM']:
            entry = results['deep_learning'].get(name, {})
            models[name] = {
                'indices': np.array(entry.get('test_indices')) if entry.get('test_indices') is not None else None,
                'y_test': np.array(entry.get('y_test')) if entry.get('y_test') is not None else None,
                'predictions': np.array(entry.get('predictions')) if entry.get('predictions') is not None else None,
                'scores': np.array(entry.get('scores')) if entry.get('scores') is not None else None,
                'timestamps': np.array(entry.get('test_timestamps')) if entry.get('test_timestamps') is not None else None,
            }

    # Fusion
    if 'fusion' in results:
        f = results['fusion']
        models['Fusion'] = {
            'indices': np.array(f.get('test_indices')) if f.get('test_indices') is not None else None,
            'y_test': np.array(f.get('y_test')) if f.get('y_test') is not None else None,
            'predictions': np.array(f.get('predictions')) if f.get('predictions') is not None else None,
            'scores': np.array(f.get('scores')) if f.get('scores') is not None else None,
            'timestamps': np.array(f.get('test_timestamps')) if f.get('test_timestamps') is not None else None,
        }

    # Collect only models that provided numeric indices
    models_with_indices = {k: v for k, v in models.items() if v.get('indices') is not None}
    if len(models_with_indices) < 2:
        # Not enough models with indices to align
        # Try timestamp-based alignment if indices insufficient
        models_with_timestamps = {k: v for k, v in models.items() if v.get('timestamps') is not None}
        if len(models_with_timestamps) < 2:
            return None
        # else fall through and compute based on timestamps below

    # Compute intersection of indices across models
    common = None
    for k, v in models_with_indices.items():
        idx = v['indices'].astype(int)
        if common is None:
            common = np.unique(idx)
        else:
            common = np.intersect1d(common, np.unique(idx))

    # If no numeric-index intersection, attempt timestamp-based alignment
    if common is None or len(common) == 0:
        # Look for timestamp arrays
        models_with_timestamps = {k: v for k, v in models.items() if v.get('timestamps') is not None}
        if len(models_with_timestamps) < 2:
            return None

        # Round timestamps to microsecond-level to avoid floating point mismatches
        ts_common = None
        for k, v in models_with_timestamps.items():
            ts = np.round(np.array(v['timestamps']).astype(float), 6)
            if ts_common is None:
                ts_common = np.unique(ts)
            else:
                ts_common = np.intersect1d(ts_common, np.unique(ts))

        if ts_common is None or len(ts_common) == 0:
            return None

        # Build aligned arrays keyed by timestamps
        aligned = {}
        for mname, mdata in models.items():
            ts = mdata.get('timestamps')
            if ts is None:
                continue
            ts_arr = np.round(np.array(ts).astype(float), 6)
            # Map each common timestamp to the first matching position in this model
            pos_map = {float(t): i for i, t in enumerate(ts_arr)}
            positions = [pos_map.get(float(t)) for t in ts_common if float(t) in pos_map]
            # Slice arrays safely
            y_test_aligned = None
            preds_aligned = None
            scores_aligned = None
            if mdata.get('y_test') is not None and len(mdata['y_test']) > 0:
                try:
                    y_test_aligned = np.array(mdata['y_test'])[positions]
                except Exception:
                    y_test_aligned = None
            if mdata.get('predictions') is not None and len(mdata['predictions']) > 0:
                try:
                    preds_aligned = np.array(mdata['predictions'])[positions]
                except Exception:
                    preds_aligned = None
            if mdata.get('scores') is not None and len(mdata['scores']) > 0:
                try:
                    scores_aligned = np.array(mdata['scores'])[positions]
                except Exception:
                    scores_aligned = None

            aligned[mname] = {
                'y_test': y_test_aligned,
                'predictions': preds_aligned,
                'scores': scores_aligned
            }

        return {
            'common_indices': np.array(ts_common),
            'aligned': aligned,
            'reference_model': 'Voltage' if 'Voltage' in aligned else next(iter(aligned.keys()))
        }

    # Build aligned arrays per model for indices in `common` (ordered by common)
    aligned = {}
    for mname, mdata in models.items():
        idx = mdata.get('indices')
        if idx is None:
            # Model had no indices — skip alignment for this model
            continue

        # Create mapping from index -> position
        idx = idx.astype(int)
        pos_map = {int(v): i for i, v in enumerate(idx)}
        positions = [pos_map[int(i)] for i in common if int(i) in pos_map]
        if len(positions) != len(common):
            # Some indices in common aren't present in this model (shouldn't happen
            # for models_with_indices), but guard defensively
            logger.debug("Model %s missing some common indices; will use available subset", mname)

        # Slice arrays safely
        y_test_aligned = None
        preds_aligned = None
        scores_aligned = None
        if mdata.get('y_test') is not None and len(mdata['y_test']) > 0:
            try:
                y_test_aligned = np.array(mdata['y_test'])[positions]
            except Exception:
                y_test_aligned = None
        if mdata.get('predictions') is not None and len(mdata['predictions']) > 0:
            try:
                preds_aligned = np.array(mdata['predictions'])[positions]
            except Exception:
                preds_aligned = None
        if mdata.get('scores') is not None and len(mdata['scores']) > 0:
            try:
                scores_aligned = np.array(mdata['scores'])[positions]
            except Exception:
                scores_aligned = None

        aligned[mname] = {
            'y_test': y_test_aligned,
            'predictions': preds_aligned,
            'scores': scores_aligned
        }

    return {
        'common_indices': np.array(common),
        'aligned': aligned,
        'reference_model': 'Voltage' if 'Voltage' in aligned else next(iter(aligned.keys()))
    }


def _generate_comparison_table(results: Dict[str, Any], results_dir: Path) -> None:
    """Create a plain-text comparison table saved under results_dir/comparison."""
    out_dir = results_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_models = []
    if 'baselines' in results:
        for name, data in results['baselines'].items():
            all_models.append((f"{name}-Based IDS", data['metrics'], 'Baseline'))

    # Voltage
    all_models.append(("Voltage Fingerprinting", results['voltage']['metrics'], 'Physical Layer'))

    # Deep learning models
    for name in ['CNN', 'LSTM']:
        all_models.append((name, results['deep_learning'][name]['metrics'], 'Deep Learning'))

    # Fusion
    all_models.append(("Fusion Layer", results['fusion']['metrics'], 'Multi-Modal'))

    table_lines = []
    table_lines.append("")
    table_lines.append("=" * 120)
    header = f"{ 'Model':<25} {'Category':<15} {'Accuracy':>10} {'TPR':>8} {'FPR':>8} {'Precision':>10} {'Recall':>8} {'F1-Score':>10} {'Latency (ms)':>12}"
    table_lines.append(header)
    table_lines.append("=" * 120)

    for model_name, metrics, category in all_models:
        latency = metrics.get('mean_latency_ms', 0.0)
        latency_str = f"{latency:.2f}" if latency > 0 else "N/A"
        tpr = metrics.get('tpr', metrics.get('true_positive_rate', metrics.get('recall', 0.0)))
        fpr = metrics.get('fpr', metrics.get('false_positive_rate', 0.0))

        line = (f"{model_name:<25} {category:<15} "
                f"{metrics.get('accuracy', 0.0):>10.4f} "
                f"{tpr:>8.4f} "
                f"{fpr:>8.4f} "
                f"{metrics.get('precision', 0.0):>10.4f} "
                f"{metrics.get('recall', 0.0):>8.4f} "
                f"{metrics.get('f1_score', 0.0):>10.4f} "
                f"{latency_str:>12}")
        table_lines.append(line)

    table_lines.append("=" * 120)
    table_lines.append("")

    comparison_file = out_dir / "model_comparison_table.txt"
    with open(comparison_file, 'w') as f:
        f.write('\n'.join(table_lines))

    logger.info(f"Comparison table saved to {comparison_file}")


def run_ablation_study(evaluator, results: Dict[str, Any], results_dir: str) -> None:
    """Run the ablation study and write reports/visualizations under results_dir/ablation."""
    results_dir = Path(results_dir)
    logger.info("\n" + "=" * 60)
    logger.info("Running Ablation Study")
    logger.info("=" * 60)

    try:
        ablation_dir = results_dir / "ablation"
        ablation_dir.mkdir(parents=True, exist_ok=True)

        v_metrics = results['voltage']['metrics']
        c_metrics = results['deep_learning']['CNN']['metrics']
        l_metrics = results['deep_learning']['LSTM']['metrics']
        f_metrics = results['fusion']['metrics']

        combinations = _simulate_fusion_combinations(v_metrics, c_metrics, l_metrics, f_metrics)
        _generate_ablation_report(combinations, ablation_dir)
        _generate_ablation_visualizations(combinations, ablation_dir)

        logger.info(f"✓ Ablation study complete: {ablation_dir}")
    except Exception:
        logger.exception("Ablation study failed")


def _simulate_fusion_combinations(v_metrics, c_metrics, l_metrics, f_metrics):
    combinations = {}
    combinations['Voltage Only'] = {
        'components': 'V',
        'accuracy': v_metrics.get('accuracy', 0.0),
        'tpr': v_metrics.get('true_positive_rate', 0.0),
        'fpr': v_metrics.get('false_positive_rate', 0.0),
        'description': 'Pure voltage fingerprinting'
    }
    combinations['CNN Only'] = {
        'components': 'C',
        'accuracy': c_metrics.get('accuracy', 0.0),
        'tpr': c_metrics.get('true_positive_rate', 0.0),
        'fpr': c_metrics.get('false_positive_rate', 0.0),
        'description': 'Pure CNN deep learning'
    }
    combinations['LSTM Only'] = {
        'components': 'L',
        'accuracy': l_metrics.get('accuracy', 0.0),
        'tpr': l_metrics.get('true_positive_rate', 0.0),
        'fpr': l_metrics.get('false_positive_rate', 0.0),
        'description': 'Pure LSTM deep learning'
    }

    v_acc, c_acc, l_acc = v_metrics.get('accuracy', 0), c_metrics.get('accuracy', 0), l_metrics.get('accuracy', 0)
    v_tpr, c_tpr, l_tpr = v_metrics.get('true_positive_rate', 0), c_metrics.get('true_positive_rate', 0), l_metrics.get('true_positive_rate', 0)
    v_fpr, c_fpr, l_fpr = v_metrics.get('false_positive_rate', 0), c_metrics.get('false_positive_rate', 0), l_metrics.get('false_positive_rate', 0)

    combinations['Voltage + CNN'] = {
        'components': 'V+C',
        'accuracy': v_acc * 0.3 + c_acc * 0.7,
        'tpr': v_tpr * 0.3 + c_tpr * 0.7,
        'fpr': v_fpr * 0.3 + c_fpr * 0.7,
        'description': 'Voltage + CNN fusion'
    }
    combinations['Voltage + LSTM'] = {
        'components': 'V+L',
        'accuracy': v_acc * 0.3 + l_acc * 0.7,
        'tpr': v_tpr * 0.3 + l_tpr * 0.7,
        'fpr': v_fpr * 0.3 + l_fpr * 0.7,
        'description': 'Voltage + LSTM fusion'
    }
    combinations['CNN + LSTM'] = {
        'components': 'C+L',
        'accuracy': c_acc * 0.5 + l_acc * 0.5,
        'tpr': c_tpr * 0.5 + l_tpr * 0.5,
        'fpr': c_fpr * 0.5 + l_fpr * 0.5,
        'description': 'Deep learning fusion (no voltage)'
    }

    combinations['Full Fusion'] = {
        'components': 'V+C+L',
        'accuracy': f_metrics.get('accuracy', 0.0),
        'tpr': f_metrics.get('true_positive_rate', 0.0),
        'fpr': f_metrics.get('false_positive_rate', 0.0),
        'description': 'All components combined'
    }

    return combinations


def _generate_ablation_report(combinations, ablation_dir: Path):
    output_file = ablation_dir / "ablation_results.txt"
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FUSION ABLATION STUDY RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Configuration':>15} {'Components':>10} {'Accuracy':>8} {'TPR':>7} {'FPR':>7} {'Description':>35}\n")
        for name, res in combinations.items():
            f.write(f"{name:>15} {res['components']:>10} {res['accuracy']:8.4f} {res['tpr']:7.4f} {res['fpr']:7.4f} {res['description']:>35}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        single_comps = {k: v for k, v in combinations.items() if '+' not in v['components']}
        best_single = max(single_comps.items(), key=lambda x: x[1]['accuracy'])
        f.write(f"Best Single Component: {best_single[0]}\n")
        f.write(f"  Accuracy: {best_single[1]['accuracy']:.4f}\n\n")

        pair_comps = {k: v for k, v in combinations.items() if v['components'].count('+') == 1}
        if pair_comps:
            best_pair = max(pair_comps.items(), key=lambda x: x[1]['accuracy'])
            f.write(f"Best Pairwise Combination: {best_pair[0]}\n")
            f.write(f"  Accuracy: {best_pair[1]['accuracy']:.4f}\n\n")

        full = combinations['Full Fusion']
        f.write(f"Full Fusion (All Components): {full['accuracy']:.4f}\n\n")

        if 'CNN + LSTM' in combinations:
            dl_only = combinations['CNN + LSTM']['accuracy']
            full_fusion = combinations['Full Fusion']['accuracy']
            improvement = full_fusion - dl_only
            f.write("\nVoltage Fingerprinting Contribution:\n")
            f.write(f"  DL-only (CNN+LSTM): {dl_only:.4f}\n")
            f.write(f"  Full Fusion (V+C+L): {full_fusion:.4f}\n")
            f.write(f"  Improvement: {improvement:+.4f}\n")
            if improvement > 0.001:
                f.write("  → Voltage adds value! ✓\n")
            else:
                f.write("  → DL models sufficient\n")

        f.write("\nNOTE: Pairwise combinations are simulated using weighted averaging.\n")

    logger.info(f"Ablation report saved to {output_file}")


def _generate_ablation_visualizations(combinations, ablation_dir: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    logger.info("Generating ablation visualizations...")
    configs = list(combinations.keys())
    accuracies = [combinations[c]['accuracy'] for c in configs]
    tprs = [combinations[c]['tpr'] for c in configs]
    fprs = [combinations[c]['fpr'] for c in configs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Fusion Ablation Study Results', fontsize=16, fontweight='bold')

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']

    ax1 = axes[0]
    bars1 = ax1.barh(configs, accuracies, color=colors[:len(configs)])
    ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy by Configuration', fontsize=14, fontweight='bold')
    ax1.set_xlim([0.6, 1.05])
    for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
        ax1.text(acc + 0.005, i, f'{acc:.3f}', va='center', fontsize=9)
    ax1.grid(axis='x', alpha=0.3)

    ax2 = axes[1]
    scatter = ax2.scatter(fprs, tprs, s=200, c=range(len(configs)), cmap='viridis', alpha=0.6, edgecolors='black', linewidth=2)
    for i, config in enumerate(configs):
        ax2.annotate(config, (fprs[i], tprs[i]), fontsize=8, xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
    ax2.set_title('TPR vs FPR Trade-off', fontsize=14, fontweight='bold')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax2.set_xlim([0, 0.4])
    ax2.set_ylim([0.7, 1.05])
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    component_names = ['Voltage', 'CNN', 'LSTM']
    components_matrix = []
    for config in configs:
        components_str = combinations[config]['components']
        row = [1 if 'V' in components_str else 0, 1 if 'C' in components_str else 0, 1 if 'L' in components_str else 0]
        components_matrix.append(row)

    accuracy_matrix = [[acc if val else 0 for val in row] for acc, row in zip(accuracies, components_matrix)]

    sns.heatmap(accuracy_matrix, annot=True, fmt='.3f', cmap='RdYlGn', xticklabels=component_names, yticklabels=configs, cbar_kws={'label': 'Accuracy'}, ax=ax3, vmin=0, vmax=1.0)
    ax3.set_title('Component Usage & Performance', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Components', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_file = ablation_dir / "ablation_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Ablation visualization saved to {output_file}")
    plt.close()
