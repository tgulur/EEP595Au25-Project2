"""Baseline trainers for timing and frequency based IDS.

This module encapsulates the baseline training logic that was previously in
`main_experiment.py`. It returns a dict with the same shape as the original
`self.results['baselines']` to keep compatibility.
"""
from typing import Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


def train_baseline_models(config: Dict[str, Any], data: Dict[str, Any], evaluator, results_dir: str) -> Dict[str, Any]:
    """Train baseline IDS models (timing-based, frequency-based).

    Args:
        config: configuration dict
        data: data dict (contains 'can' splits and possibly raw data)
        evaluator: IDSEvaluator instance

    Returns:
        baseline_results: dict mapping baseline names to result dicts
    """
    # Local imports to avoid top-level dependencies during import
    from src.baseline_models import TimingBasedIDS, FrequencyBasedIDS, RuleBasedIDS
    from src.dataset_loader import CANDatasetLoader
    from sklearn.model_selection import train_test_split

    logger.info("\n" + "=" * 60)
    logger.info("Training Baseline Models")
    logger.info("=" * 60)

    # For baselines, we prefer to operate on raw CAN dataframe to extract
    # timestamps and IDs as required by the baseline implementations.
    loader = CANDatasetLoader(config['data']['raw_path'])
    can_df = loader.load_road_dataset(config['data'].get('road_path'))

    # Split into train/test (matching existing behavior in main_experiment)
    train_df, test_df = train_test_split(
        can_df,
        test_size=(1 - config['data']['train_ratio']),
        random_state=config['data']['random_seed'],
        stratify=can_df['Label'] if 'Label' in can_df.columns else None,
    )

    baseline_results = {}

    # Timestamps, IDs and labels extraction helpers
    def _get_columns(df):
        timestamps = df['Timestamp'].values if 'Timestamp' in df.columns else np.arange(len(df))
        ids = df['ID'].values if 'ID' in df.columns else np.zeros(len(df))
        labels = df['Label'].values if 'Label' in df.columns else np.zeros(len(df))
        return timestamps, ids, labels

    train_timestamps, train_ids, train_labels = _get_columns(train_df)
    test_timestamps, test_ids, test_labels = _get_columns(test_df)

    # 1. Timing-Based IDS
    if config['baselines'].get('timing_based', {}).get('enabled', True):
        logger.info("\nTraining Timing-Based IDS...")

        timing_cfg = config['baselines'].get('timing_based', {})
        timing_ids = TimingBasedIDS(threshold=timing_cfg.get('threshold', 0.05))

        timing_ids.train(train_timestamps, train_ids, train_labels)

        predictions, scores = timing_ids.predict(test_timestamps, test_ids)

        metrics = evaluator.calculate_metrics(test_labels, predictions, scores)

        evaluator.plot_confusion_matrix(
            test_labels, predictions,
            title="Timing-Based IDS - Confusion Matrix",
            save_name="baselines/timing_confusion_matrix.png",
        )
        evaluator.plot_roc_curve(
            test_labels, scores,
            title="Timing-Based IDS - ROC Curve",
            save_name="baselines/timing_roc_curve.png",
        )

        report = evaluator.generate_report(
            "Timing-Based IDS", metrics, save_name="baselines/timing_report.txt"
        )
        logger.info(report)

        baseline_results['Timing'] = {
            'metrics': metrics,
            'predictions': predictions,
            'scores': scores,
            'model': timing_ids,
            'y_test': test_labels,
        }
        # Save timing IDS model
        try:
            from src.artifacts import save_model

            _ = save_model(timing_ids, results_dir, "timing_ids", subdir="baselines/timing", filename="timing_model")
        except Exception:
            logger.exception("Failed to save timing IDS via artifacts.save_model; continuing")

    # 2. Frequency-Based IDS
    if config['baselines'].get('frequency_based', {}).get('enabled', True):
        logger.info("\nTraining Frequency-Based IDS...")

        freq_cfg = config['baselines'].get('frequency_based', {})
        freq_ids = FrequencyBasedIDS(
            window_size=freq_cfg.get('window_size', 100),
            threshold=freq_cfg.get('threshold', 0.3),
        )

        freq_ids.train(train_timestamps, train_ids, train_labels)
        predictions, scores = freq_ids.predict(test_timestamps, test_ids)

        metrics = evaluator.calculate_metrics(test_labels, predictions, scores)

        evaluator.plot_confusion_matrix(
            test_labels, predictions,
            title="Frequency-Based IDS - Confusion Matrix",
            save_name="baselines/frequency_confusion_matrix.png",
        )
        evaluator.plot_roc_curve(
            test_labels, scores,
            title="Frequency-Based IDS - ROC Curve",
            save_name="baselines/frequency_roc_curve.png",
        )

        report = evaluator.generate_report(
            "Frequency-Based IDS", metrics, save_name="baselines/frequency_report.txt"
        )
        logger.info(report)

        baseline_results['Frequency'] = {
            'metrics': metrics,
            'predictions': predictions,
            'scores': scores,
            'model': freq_ids,
            'y_test': test_labels,
        }
        # Save frequency IDS model
        try:
            from src.artifacts import save_model

            _ = save_model(freq_ids, results_dir, "frequency_ids", subdir="baselines/frequency", filename="frequency_model")
        except Exception:
            logger.exception("Failed to save frequency IDS via artifacts.save_model; continuing")

    logger.info(f"\nBaseline models training complete. Trained {len(baseline_results)} models.")

    return baseline_results
