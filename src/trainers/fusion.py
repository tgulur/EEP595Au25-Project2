"""Fusion trainer.

Provides `train_fusion_layer` which encapsulates the fusion training
and evaluation logic previously colocated in `main_experiment.py`.
"""
from typing import Dict, Any, Tuple
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


def train_fusion_layer(config: Dict[str, Any], results: Dict[str, Any], data: Dict[str, Any], evaluator, results_dir: str) -> Tuple[object, Dict[str, Any]]:
    """Train and evaluate the fusion layer.

    Args:
        config: raw configuration dict
        results: existing results dict (should contain 'voltage' and 'deep_learning')
        data: data dict with splits
        evaluator: IDSEvaluator instance
        results_dir: path to the experiment results directory

    Returns:
        (fusion_model, result_dict)
    """
    # Local imports to avoid import-time heavy dependencies
    from src.fusion_layer import FusionLayer

    results_dir = Path(results_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Training Fusion Layer")
    logger.info("=" * 60)

    # Get test data
    X_voltage_test, y_voltage_test = data['voltage']['test']
    X_can_test, y_can_test = data['can']['test']

    # Ensure same number of samples
    min_samples = min(len(X_voltage_test), len(X_can_test))

    # Get voltage scores
    voltage_scores = results['voltage']['scores'][:min_samples]
    # voltage_confidence: higher when score is nearer to 0 (we treat score as anomaly likelihood)
    voltage_confidences = 1.0 - voltage_scores

    # Get DL scores (use CNN by default)
    # Deep learning trainer stores per-model results under keys like 'CNN' and 'LSTM'
    dl_scores = results['deep_learning']['CNN']['scores'][:min_samples]
    dl_confidences = np.abs(dl_scores - 0.5) * 2  # convert score to 0..1 confidence

    y_test = y_can_test[:min_samples]

    # Split for training/testing fusion
    split_idx = int(0.7 * len(voltage_scores))

    fusion = FusionLayer(
        method=config['fusion']['method'],
        combiner_model=config['fusion']['combiner']['model']
    )

    fusion.train(
        voltage_scores[:split_idx],
        dl_scores[:split_idx],
        voltage_confidences[:split_idx],
        dl_confidences[:split_idx],
        y_test[:split_idx]
    )

    # Test fusion
    predictions, confidences = fusion.predict_batch(
        voltage_scores[split_idx:],
        dl_scores[split_idx:],
        voltage_confidences[split_idx:],
        dl_confidences[split_idx:]
    )

    # Calculate metrics
    metrics = evaluator.calculate_metrics(
        y_test[split_idx:],
        predictions,
        confidences
    )

    # Visualizations
    evaluator.plot_confusion_matrix(
        y_test[split_idx:], predictions,
        title="Fusion Layer - Confusion Matrix",
        save_name="fusion/fusion_confusion_matrix.png"
    )
    evaluator.plot_roc_curve(
        y_test[split_idx:], confidences,
        title="Fusion Layer - ROC Curve",
        save_name="fusion/fusion_roc_curve.png"
    )

    # Report
    report = evaluator.generate_report(
        "Fusion Layer",
        metrics,
        save_name="fusion/fusion_report.txt"
    )
    logger.info(report)

    result = {
        'metrics': metrics,
        'predictions': predictions,
        'scores': confidences,
        'model': fusion,
        'y_test': y_test[split_idx:],
        'test_indices': (np.array(data['can'].get('test_indices'))[:min_samples][split_idx:]
                         if data['can'].get('test_indices') is not None else None)
    }

    # Save fusion model artifact
    try:
        from src.artifacts import save_model

        save_model(fusion, results_dir, "fusion", subdir="fusion", filename="fusion_model")
    except Exception:
        logger.exception("Failed to save fusion model via artifacts.save_model; continuing")

    return fusion, result
