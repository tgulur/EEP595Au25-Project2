"""Voltage trainer module.

Contains a single function `train_voltage_model` that mirrors the previous
`CANIDSExperiment.train_voltage_model` behavior but is easier to unit-test
and reuse from other runners.
"""
from typing import Tuple, Dict, Any, List, cast
import logging
import numpy as np

logger = logging.getLogger(__name__)


def train_voltage_model(config: Dict[str, Any], data: Dict[str, Any], evaluator, results_dir: str) -> Tuple[object, Dict[str, Any]]:
    """Train voltage fingerprinter and evaluate.

    Args:
        config: Raw configuration dict (same as `self.config`)
        data: Data dict with 'voltage' splits and 'voltage_df'
        evaluator: IDSEvaluator instance for metrics/plots

    Returns:
        (fingerprinter, result_dict)
    """
    # Local imports to avoid circular import problems
    from src.voltage_fingerprinting import VoltageFingerprinter

    logger.info("\n" + "=" * 60)
    logger.info("Training Voltage Fingerprinting Model")
    logger.info("=" * 60)

    X_train, y_train = data['voltage']['train']
    X_test, y_test = data['voltage']['test']

    voltage_df = data['voltage_df']
    train_indices = data['voltage']['train_indices']
    test_indices = data['voltage']['test_indices']

    ecu_ids_train = cast(List[int], [int(x) for x in voltage_df.loc[train_indices, 'ecu_id'].tolist()])
    ecu_ids_test = cast(List[int], [int(x) for x in voltage_df.loc[test_indices, 'ecu_id'].tolist()])

    fingerprinter = VoltageFingerprinter(threshold=config['voltage']['anomaly_threshold'])
    fingerprinter.train(X_train, ecu_ids_train)

    predictions = []
    scores = []

    for i in range(len(X_test)):
        claimed_ecu = ecu_ids_test[i]
        is_anomaly, confidence = fingerprinter.detect_anomaly(X_test[i], claimed_ecu)
        predictions.append(int(is_anomaly))
        scores.append(1.0 - confidence)

    predictions = np.array(predictions)
    scores = np.array(scores)

    metrics = evaluator.calculate_metrics(y_test, predictions, scores)

    # Visualizations
    evaluator.plot_confusion_matrix(
        y_test, predictions,
        title="Voltage Fingerprinting - Confusion Matrix",
        save_name="voltage/voltage_confusion_matrix.png"
    )
    evaluator.plot_roc_curve(
        y_test, scores,
        title="Voltage Fingerprinting - ROC Curve",
        save_name="voltage/voltage_roc_curve.png"
    )

    report = evaluator.generate_report(
        "Voltage Fingerprinting",
        metrics,
        save_name="voltage/voltage_report.txt"
    )
    logger.info(report)

    result = {
        'metrics': metrics,
        'predictions': predictions,
        'scores': scores,
        'model': fingerprinter,
        'y_test': y_test,
        'test_indices': np.array(test_indices),
        'test_timestamps': np.array(voltage_df['timestamp'].values[test_indices]) if voltage_df is not None and len(test_indices) > 0 else None
    }
    # Attempt to save model artifact using artifacts helper
    try:
        from src.artifacts import save_model

        save_path = save_model(fingerprinter, results_dir, "voltage_fingerprinter", subdir="voltage", filename="voltage_model")
        logger.info(f"Saved voltage fingerprinter to {save_path}")
    except Exception:
        logger.exception("Failed to save voltage fingerprinter via artifacts.save_model; continuing")

    return fingerprinter, result
