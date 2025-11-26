"""Deep learning trainers for CNN and LSTM models.

Provides a `train_deep_learning_models` function that mirrors the previous
`CANIDSExperiment.train_deep_learning_models` implementation but is isolated
for easier testing and reuse.
"""
from typing import Dict, Any, Union
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


def train_deep_learning_models(config: Dict[str, Any], data: Dict[str, Any], evaluator, results_dir: Union[str, Path]) -> Dict[str, Any]:
    """Train CNN and LSTM models and evaluate them.

    Args:
        config: Raw configuration dict
        data: Data dictionary with 'can' splits
        evaluator: IDSEvaluator instance
        results_dir: Path to results directory (string or Path)

    Returns:
        dl_results: dict with per-model metrics, predictions, scores, model, y_test
    """
    # Local imports
    from src.deep_learning_models import CNNModel, LSTMModel, HybridCNNLSTM

    results_dir = Path(results_dir)

    X_train, y_train = data['can']['train']
    X_val, y_val = data['can']['val']
    X_test, y_test = data['can']['test']

    input_shape = X_train.shape[1:]

    models_to_train = {
        'CNN': CNNModel(
            filters=config['deep_learning']['cnn']['filters'],
            kernel_size=config['deep_learning']['cnn']['kernel_size'],
            dropout=config['deep_learning']['cnn']['dropout']
        ),
        'LSTM': LSTMModel(
            hidden_units=config['deep_learning']['lstm']['hidden_units'],
            dropout=config['deep_learning']['lstm']['dropout'],
            bidirectional=config['deep_learning']['lstm']['bidirectional']
        )
    }

    dl_results = {}

    for model_name, model in models_to_train.items():
        logger.info(f"\nTraining {model_name}...")

        # Build and train
        model.build_model(input_shape=input_shape)
        model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=config['deep_learning']['epochs'],
            batch_size=config['deep_learning']['batch_size'],
            learning_rate=config['deep_learning']['learning_rate'],
            patience=config['deep_learning']['patience']
        )

        # Evaluate
        predictions, scores = model.predict(X_test)
        metrics = evaluator.calculate_metrics(y_test, predictions, scores)

        # Measure latency
        if config.get('evaluation', {}).get('measure_latency', False):
            latency_metrics = model.predict_with_latency(
                X_test[:1],
                n_iterations=config.get('evaluation', {}).get('latency_iterations', 100)
            )
            metrics.update(latency_metrics)

        # Visualizations
        evaluator.plot_confusion_matrix(
            y_test, predictions,
            title=f"{model_name} - Confusion Matrix",
            save_name=f"{model_name.lower()}/{model_name.lower()}_confusion_matrix.png"
        )
        evaluator.plot_roc_curve(
            y_test, scores,
            title=f"{model_name} - ROC Curve",
            save_name=f"{model_name.lower()}/{model_name.lower()}_roc_curve.png"
        )

        # Report
        report = evaluator.generate_report(
            model_name,
            metrics,
            save_name=f"{model_name.lower()}/{model_name.lower()}_report.txt"
        )
        logger.info(report)

        # Save model using artifacts helper for consistent naming
        try:
            from src.artifacts import save_model

            saved_path = save_model(model, str(results_dir), model_name, subdir=model_name.lower(), filename=f"{model_name.lower()}_model")
            logger.info(f"Saved {model_name} model to {saved_path}")
        except Exception:
            logger.exception("Failed to save model via artifacts.save_model; continuing")

        dl_results[model_name] = {
            'metrics': metrics,
            'predictions': predictions,
            'scores': scores,
            'model': model,
            'y_test': y_test,
            'test_indices': np.array(data['can'].get('test_indices')) if data['can'].get('test_indices') is not None else None,
            # Map sequence indices back to CAN dataframe timestamps (sequence i -> df timestamp at i)
            'test_timestamps': (np.array(data['can_df']['timestamp'])[np.array(data['can'].get('test_indices'))]
                                if data.get('can_df') is not None and data['can'].get('test_indices') is not None and len(data['can'].get('test_indices')) > 0 else None)
        }

    return dl_results
