"""
main_experiment.py - Main script for training and evaluating CAN IDS system
"""

import os
import sys
import yaml
import numpy as np
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Set CUDA library paths for TensorFlow GPU support
venv_path = Path(__file__).parent / ".venv"
nvidia_lib_paths = [
    venv_path / "lib64/python3.12/site-packages/nvidia/cublas/lib",
    venv_path / "lib64/python3.12/site-packages/nvidia/cudnn/lib",
    venv_path / "lib64/python3.12/site-packages/nvidia/cuda_runtime/lib",
    venv_path / "lib64/python3.12/site-packages/nvidia/cuda_cupti/lib",
    venv_path / "lib64/python3.12/site-packages/nvidia/cuda_nvrtc/lib",
]
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
nvidia_paths_str = ':'.join(str(p) for p in nvidia_lib_paths if p.exists())
if nvidia_paths_str:
    os.environ['LD_LIBRARY_PATH'] = f"{nvidia_paths_str}:{current_ld_path}" if current_ld_path else nvidia_paths_str

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from dataset_loader import CANDatasetLoader
from voltage_fingerprinting import VoltageFingerprinter
from deep_learning_models import CNNModel, LSTMModel, HybridCNNLSTM
from fusion_layer import FusionLayer, AdaptiveFusion
from baseline_models import TimingBasedIDS, FrequencyBasedIDS, RuleBasedIDS
from evaluation_metrics import IDSEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CANIDSExperiment:
    """Main experiment class for CAN IDS"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize experiment
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(self.config['output']['results_path']) / self.timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluator = IDSEvaluator(save_dir=str(self.results_dir))
        self.results = {}
        
        logger.info(f"Experiment initialized. Results will be saved to {self.results_dir}")
    
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    def load_and_prepare_data(self):
        """Load and prepare datasets"""
        logger.info("Loading datasets...")
        
        loader = CANDatasetLoader(self.config['data']['raw_path'])
        
        # Load voltage dataset
        voltage_df = loader.load_canmap_voltage_dataset(
            self.config['data'].get('canmap_path')
        )
        
        # Load CAN message dataset
        can_df = loader.load_road_dataset(
            self.config['data'].get('road_path')
        )
        
        # Preprocess voltage data
        logger.info("Preprocessing voltage data...")
        X_voltage, y_voltage = loader.preprocess_voltage_data(voltage_df)
        
        # Preprocess CAN message data
        logger.info("Preprocessing CAN message data...")
        sequence_length = self.config['deep_learning']['sequence_length']
        X_can, y_can = loader.preprocess_can_data(can_df, sequence_length=sequence_length)
        
        # Split data
        voltage_splits = loader.split_data(
            X_voltage, y_voltage,
            train_ratio=self.config['data']['train_ratio'],
            val_ratio=self.config['data']['val_ratio'],
            test_ratio=self.config['data']['test_ratio'],
            random_seed=self.config['data']['random_seed']
        )
        
        can_splits = loader.split_data(
            X_can, y_can,
            train_ratio=self.config['data']['train_ratio'],
            val_ratio=self.config['data']['val_ratio'],
            test_ratio=self.config['data']['test_ratio'],
            random_seed=self.config['data']['random_seed']
        )
        
        logger.info("Data preparation complete")
        
        return {
            'voltage': voltage_splits,
            'can': can_splits,
            'voltage_df': voltage_df,
            'can_df': can_df
        }
    
    def train_voltage_model(self, data: dict):
        """Train voltage fingerprinting model"""
        logger.info("\n" + "="*60)
        logger.info("Training Voltage Fingerprinting Model")
        logger.info("="*60)
        
        X_train, y_train = data['voltage']['train']
        X_test, y_test = data['voltage']['test']
        
        # Get indices to properly track ECU IDs from dataframe
        voltage_df = data['voltage_df']
        train_indices = data['voltage']['train_indices']
        test_indices = data['voltage']['test_indices']
        
        # Get ECU IDs using the correct indices
        ecu_ids_train = voltage_df['ecu_id'].values[train_indices]
        ecu_ids_test = voltage_df['ecu_id'].values[test_indices]
        
        # Train fingerprinter
        fingerprinter = VoltageFingerprinter(
            threshold=self.config['voltage']['anomaly_threshold']
        )
        fingerprinter.train(X_train, ecu_ids_train)
        
        # Evaluate
        predictions = []
        scores = []
        
        for i in range(len(X_test)):
            claimed_ecu = ecu_ids_test[i]
            is_anomaly, confidence = fingerprinter.detect_anomaly(X_test[i], claimed_ecu)
            predictions.append(int(is_anomaly))
            scores.append(1.0 - confidence)  # Higher score = more anomalous
        
        predictions = np.array(predictions)
        scores = np.array(scores)
        
        # Calculate metrics
        metrics = self.evaluator.calculate_metrics(y_test, predictions, scores)
        
        # Visualizations
        self.evaluator.plot_confusion_matrix(
            y_test, predictions,
            title="Voltage Fingerprinting - Confusion Matrix",
            save_name="voltage_confusion_matrix.png"
        )
        self.evaluator.plot_roc_curve(
            y_test, scores,
            title="Voltage Fingerprinting - ROC Curve",
            save_name="voltage_roc_curve.png"
        )
        
        # Report
        report = self.evaluator.generate_report(
            "Voltage Fingerprinting",
            metrics,
            save_name="voltage_report.txt"
        )
        logger.info(report)
        
        self.results['voltage'] = {
            'metrics': metrics,
            'predictions': predictions,
            'scores': scores,
            'model': fingerprinter,
            'y_test': y_test
        }
        
        return fingerprinter
    
    def train_baseline_models(self, data: dict):
        """Train baseline IDS models for comparison"""
        logger.info("\n" + "="*60)
        logger.info("Training Baseline Models")
        logger.info("="*60)
        
        # Extract CAN data (timestamps, IDs, labels)
        can_train_data = data['can']['train']
        can_test_data = data['can']['test']
        
        # For baselines, we need raw CAN data with timestamps and IDs
        # Use the CAN dataset loader to get this info
        loader = CANDatasetLoader(self.config['data']['raw_path'])
        can_df = loader.load_road_dataset(self.config['data'].get('road_path'))
        
        # Split into train/test (matching our existing split)
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            can_df, 
            test_size=(1 - self.config['data']['train_ratio']),
            random_state=self.config['data']['random_seed'],
            stratify=can_df['Label'] if 'Label' in can_df.columns else None
        )
        
        baseline_results = {}
        
        # 1. Timing-Based IDS
        if self.config['baselines'].get('timing_based', {}).get('enabled', True):
            logger.info("\nTraining Timing-Based IDS...")
            
            timing_ids = TimingBasedIDS(
                threshold=self.config['baselines']['timing_based'].get('threshold', 0.05)
            )
            
            # Extract timestamps, IDs, labels from training data
            train_timestamps = train_df['Timestamp'].values if 'Timestamp' in train_df.columns else np.arange(len(train_df))
            train_ids = train_df['ID'].values if 'ID' in train_df.columns else np.zeros(len(train_df))
            train_labels = train_df['Label'].values if 'Label' in train_df.columns else np.zeros(len(train_df))
            
            timing_ids.train(train_timestamps, train_ids, train_labels)
            
            # Predict on test data
            test_timestamps = test_df['Timestamp'].values if 'Timestamp' in test_df.columns else np.arange(len(test_df))
            test_ids = test_df['ID'].values if 'ID' in test_df.columns else np.zeros(len(test_df))
            test_labels = test_df['Label'].values if 'Label' in test_df.columns else np.zeros(len(test_df))
            
            predictions, scores = timing_ids.predict(test_timestamps, test_ids)
            
            # Evaluate
            metrics = self.evaluator.calculate_metrics(test_labels, predictions, scores)
            
            # Visualizations
            self.evaluator.plot_confusion_matrix(
                test_labels, predictions,
                title="Timing-Based IDS - Confusion Matrix",
                save_name="timing_confusion_matrix.png"
            )
            self.evaluator.plot_roc_curve(
                test_labels, scores,
                title="Timing-Based IDS - ROC Curve",
                save_name="timing_roc_curve.png"
            )
            
            # Report
            report = self.evaluator.generate_report(
                "Timing-Based IDS",
                metrics,
                save_name="timing_report.txt"
            )
            logger.info(report)
            
            baseline_results['Timing'] = {
                'metrics': metrics,
                'predictions': predictions,
                'scores': scores,
                'model': timing_ids,
                'y_test': test_labels
            }
        
        # 2. Frequency-Based IDS  
        if self.config['baselines'].get('frequency_based', {}).get('enabled', True):
            logger.info("\nTraining Frequency-Based IDS...")
            
            freq_ids = FrequencyBasedIDS(
                window_size=self.config['baselines']['frequency_based'].get('window_size', 100),
                threshold=self.config['baselines']['frequency_based'].get('threshold', 0.3)
            )
            
            freq_ids.train(train_timestamps, train_ids, train_labels)
            predictions, scores = freq_ids.predict(test_timestamps, test_ids)
            
            # Evaluate
            metrics = self.evaluator.calculate_metrics(test_labels, predictions, scores)
            
            # Visualizations
            self.evaluator.plot_confusion_matrix(
                test_labels, predictions,
                title="Frequency-Based IDS - Confusion Matrix",
                save_name="frequency_confusion_matrix.png"
            )
            self.evaluator.plot_roc_curve(
                test_labels, scores,
                title="Frequency-Based IDS - ROC Curve",
                save_name="frequency_roc_curve.png"
            )
            
            # Report
            report = self.evaluator.generate_report(
                "Frequency-Based IDS",
                metrics,
                save_name="frequency_report.txt"
            )
            logger.info(report)
            
            baseline_results['Frequency'] = {
                'metrics': metrics,
                'predictions': predictions,
                'scores': scores,
                'model': freq_ids,
                'y_test': test_labels
            }
        
        self.results['baselines'] = baseline_results
        logger.info(f"\nBaseline models training complete. Trained {len(baseline_results)} models.")
        
        return baseline_results
    
    def train_deep_learning_models(self, data: dict):
        """Train deep learning models"""
        logger.info("\n" + "="*60)
        logger.info("Training Deep Learning Models")
        logger.info("="*60)
        
        X_train, y_train = data['can']['train']
        X_val, y_val = data['can']['val']
        X_test, y_test = data['can']['test']
        
        input_shape = X_train.shape[1:]
        
        models_to_train = {
            'CNN': CNNModel(
                filters=self.config['deep_learning']['cnn']['filters'],
                kernel_size=self.config['deep_learning']['cnn']['kernel_size'],
                dropout=self.config['deep_learning']['cnn']['dropout']
            ),
            'LSTM': LSTMModel(
                hidden_units=self.config['deep_learning']['lstm']['hidden_units'],
                dropout=self.config['deep_learning']['lstm']['dropout'],
                bidirectional=self.config['deep_learning']['lstm']['bidirectional']
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
                epochs=self.config['deep_learning']['epochs'],
                batch_size=self.config['deep_learning']['batch_size'],
                learning_rate=self.config['deep_learning']['learning_rate'],
                patience=self.config['deep_learning']['patience']
            )
            
            # Evaluate
            predictions, scores = model.predict(X_test)
            metrics = self.evaluator.calculate_metrics(y_test, predictions, scores)
            
            # Measure latency
            if self.config['evaluation']['measure_latency']:
                latency_metrics = model.predict_with_latency(
                    X_test[:1],
                    n_iterations=self.config['evaluation']['latency_iterations']
                )
                metrics.update(latency_metrics)
            
            # Visualizations
            self.evaluator.plot_confusion_matrix(
                y_test, predictions,
                title=f"{model_name} - Confusion Matrix",
                save_name=f"{model_name.lower()}_confusion_matrix.png"
            )
            self.evaluator.plot_roc_curve(
                y_test, scores,
                title=f"{model_name} - ROC Curve",
                save_name=f"{model_name.lower()}_roc_curve.png"
            )
            
            # Report
            report = self.evaluator.generate_report(
                model_name,
                metrics,
                save_name=f"{model_name.lower()}_report.txt"
            )
            logger.info(report)
            
            # Save model
            model_path = self.results_dir / f"{model_name.lower()}_model.h5"
            model.save_model(str(model_path))
            
            dl_results[model_name] = {
                'metrics': metrics,
                'predictions': predictions,
                'scores': scores,
                'model': model,
                'y_test': y_test
            }
        
        self.results['deep_learning'] = dl_results
        
        return dl_results
    
    def train_fusion_layer(self, voltage_model, dl_models, data: dict):
        """Train fusion layer"""
        logger.info("\n" + "="*60)
        logger.info("Training Fusion Layer")
        logger.info("="*60)
        
        # Get test data
        X_voltage_test, y_voltage_test = data['voltage']['test']
        X_can_test, y_can_test = data['can']['test']
        
        # Ensure same number of samples
        min_samples = min(len(X_voltage_test), len(X_can_test))
        
        # Get voltage scores
        voltage_scores = self.results['voltage']['scores'][:min_samples]
        voltage_confidences = 1.0 - voltage_scores
        
        # Get DL scores (use CNN)
        dl_scores = self.results['deep_learning']['CNN']['scores'][:min_samples]
        dl_confidences = np.abs(dl_scores - 0.5) * 2  # Convert to confidence
        
        y_test = y_can_test[:min_samples]
        
        # Split for training/testing fusion
        split_idx = int(0.7 * len(voltage_scores))
        
        # Train fusion
        fusion = FusionLayer(
            method=self.config['fusion']['method'],
            combiner_model=self.config['fusion']['combiner']['model']
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
        metrics = self.evaluator.calculate_metrics(
            y_test[split_idx:],
            predictions,
            confidences
        )
        
        # Visualizations
        self.evaluator.plot_confusion_matrix(
            y_test[split_idx:], predictions,
            title="Fusion Layer - Confusion Matrix",
            save_name="fusion_confusion_matrix.png"
        )
        self.evaluator.plot_roc_curve(
            y_test[split_idx:], confidences,
            title="Fusion Layer - ROC Curve",
            save_name="fusion_roc_curve.png"
        )
        
        # Report
        report = self.evaluator.generate_report(
            "Fusion Layer",
            metrics,
            save_name="fusion_report.txt"
        )
        logger.info(report)
        
        self.results['fusion'] = {
            'metrics': metrics,
            'predictions': predictions,
            'scores': confidences,
            'model': fusion,
            'y_test': y_test[split_idx:]
        }
        
        return fusion
    
    def compare_all_models(self):
        """Generate comprehensive comparison visualizations"""
        logger.info("\n" + "="*60)
        logger.info("Generating Model Comparisons")
        logger.info("="*60)
        
        # Collect all results (including baselines)
        comparison_results = {
            'Voltage': self.results['voltage']['metrics'],
            'CNN': self.results['deep_learning']['CNN']['metrics'],
            'LSTM': self.results['deep_learning']['LSTM']['metrics'],
            'Fusion': self.results['fusion']['metrics']
        }
        
        # Add baseline models if they exist
        if 'baselines' in self.results:
            for baseline_name, baseline_data in self.results['baselines'].items():
                comparison_results[f'{baseline_name}-IDS'] = baseline_data['metrics']
        
        # Basic comparison plot (now includes baselines)
        self.evaluator.compare_models(
            comparison_results,
            metric_names=['accuracy', 'precision', 'recall', 'f1_score'],
            save_name="model_comparison.png"
        )
        
        # Create attack detection timeline for each model (use model-specific y_test)
        logger.info("Generating attack detection timelines...")
        
        for model_name in ['Voltage', 'CNN', 'LSTM', 'Fusion']:
            if model_name == 'Voltage':
                predictions = self.results['voltage']['predictions']
                y_test = self.results['voltage']['y_test']
            elif model_name == 'CNN':
                predictions = self.results['deep_learning']['CNN']['predictions']
                y_test = self.results['deep_learning']['CNN']['y_test']
            elif model_name == 'LSTM':
                predictions = self.results['deep_learning']['LSTM']['predictions']
                y_test = self.results['deep_learning']['LSTM']['y_test']
            else:  # Fusion
                predictions = self.results['fusion']['predictions']
                y_test = self.results['fusion']['y_test']
            
            self.evaluator.plot_attack_detection_timeline(
                y_test,
                predictions,
                model_name=model_name,
                save_name=f"{model_name.lower()}_attack_timeline.png"
            )
        
        # Create detection heatmap showing which models detected which attacks
        # Use voltage y_test as reference (all models tested on compatible data)
        logger.info("Generating detection heatmap...")
        y_test_ref = self.results['voltage']['y_test']
        predictions_dict = {
            'Voltage': self.results['voltage']['predictions'],
            'CNN': self.results['deep_learning']['CNN']['predictions'],
            'LSTM': self.results['deep_learning']['LSTM']['predictions'],
        }
        
        self.evaluator.plot_detection_heatmap(
            y_test_ref,
            predictions_dict,
            save_name="detection_heatmap.png"
        )
        
        # Create comprehensive comparison dashboard
        # Use voltage y_test as reference for comprehensive comparison
        logger.info("Generating comprehensive comparison dashboard...")
        y_test_ref = self.results['voltage']['y_test']
        comprehensive_results = {
            'Voltage': {
                'metrics': self.results['voltage']['metrics'],
                'predictions': self.results['voltage']['predictions']
            },
            'CNN': {
                'metrics': self.results['deep_learning']['CNN']['metrics'],
                'predictions': self.results['deep_learning']['CNN']['predictions']
            },
            'LSTM': {
                'metrics': self.results['deep_learning']['LSTM']['metrics'],
                'predictions': self.results['deep_learning']['LSTM']['predictions']
            }
        }
        
        self.evaluator.plot_comprehensive_comparison(
            comprehensive_results,
            y_test_ref,
            save_name="comprehensive_comparison.png"
        )
        
        logger.info("All comparison visualizations complete")
        
        # Generate comprehensive comparison table
        self.generate_comparison_table()
    
    def generate_comparison_table(self):
        """Generate a comprehensive comparison table of all models"""
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE MODEL COMPARISON TABLE")
        logger.info("="*60)
        
        # Collect all model results
        all_models = []
        
        # Baseline models
        if 'baselines' in self.results:
            for name, data in self.results['baselines'].items():
                all_models.append((f"{name}-Based IDS", data['metrics'], 'Baseline'))
        
        # Voltage fingerprinting
        all_models.append(("Voltage Fingerprinting", self.results['voltage']['metrics'], 'Physical Layer'))
        
        # Deep learning models
        for name in ['CNN', 'LSTM']:
            all_models.append((name, self.results['deep_learning'][name]['metrics'], 'Deep Learning'))
        
        # Fusion
        all_models.append(("Fusion Layer", self.results['fusion']['metrics'], 'Multi-Modal'))
        
        # Create formatted table
        table_lines = []
        table_lines.append("")
        table_lines.append("="*120)
        header = f"{'Model':<25} {'Category':<15} {'Accuracy':>10} {'TPR':>8} {'FPR':>8} {'Precision':>10} {'Recall':>8} {'F1-Score':>10} {'Latency (ms)':>12}"
        table_lines.append(header)
        table_lines.append("="*120)
        
        for model_name, metrics, category in all_models:
            latency = metrics.get('mean_latency_ms', 0.0)
            latency_str = f"{latency:.2f}" if latency > 0 else "N/A"
            
            # Handle different metric key formats (some models use 'tpr', others use 'true_positive_rate')
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
        
        table_lines.append("="*120)
        table_lines.append("")
        table_lines.append("NOTES:")
        table_lines.append("  - TPR (True Positive Rate): Percentage of attacks correctly detected")
        table_lines.append("  - FPR (False Positive Rate): Percentage of normal traffic incorrectly flagged as attack")
        table_lines.append("  - Latency: Mean inference time per sample (N/A for non-DL models)")
        table_lines.append("")
        table_lines.append("KEY FINDINGS:")
        
        # Find best models (with safe key access)
        def get_metric(model_tuple, key, fallback_keys=[]):
            metrics = model_tuple[1]
            value = metrics.get(key)
            if value is None:
                for fallback in fallback_keys:
                    value = metrics.get(fallback)
                    if value is not None:
                        break
            return value if value is not None else 0.0
        
        best_accuracy = max(all_models, key=lambda x: get_metric(x, 'accuracy'))
        best_tpr = max(all_models, key=lambda x: get_metric(x, 'tpr', ['true_positive_rate', 'recall']))
        lowest_fpr = min(all_models, key=lambda x: get_metric(x, 'fpr', ['false_positive_rate']))
        
        table_lines.append(f"  - Best Accuracy: {best_accuracy[0]} ({get_metric(best_accuracy, 'accuracy'):.4f})")
        table_lines.append(f"  - Best TPR (Attack Detection): {best_tpr[0]} ({get_metric(best_tpr, 'tpr', ['true_positive_rate', 'recall']):.4f})")
        table_lines.append(f"  - Lowest FPR (Fewest False Alarms): {lowest_fpr[0]} ({get_metric(lowest_fpr, 'fpr', ['false_positive_rate']):.4f})")
        table_lines.append("")
        
        # Print to console
        for line in table_lines:
            logger.info(line)
        
        # Save to file
        comparison_file = self.results_dir / "model_comparison_table.txt"
        with open(comparison_file, 'w') as f:
            f.write('\n'.join(table_lines))
        
        logger.info(f"Comparison table saved to {comparison_file}")
    
    def run_experiment(self):
        """Run full experiment"""
        logger.info("\n" + "="*60)
        logger.info("Starting CAN IDS Experiment")
        logger.info("="*60)
        
        try:
            # Load data
            data = self.load_and_prepare_data()
            
            # Train voltage model
            voltage_model = self.train_voltage_model(data)
            
            # Train baseline models
            baseline_models = self.train_baseline_models(data)
            
            # Train DL models
            dl_models = self.train_deep_learning_models(data)
            
            # Train fusion layer
            fusion_model = self.train_fusion_layer(voltage_model, dl_models, data)
            
            # Compare models (including baselines)
            self.compare_all_models()
            
            logger.info("\n" + "="*60)
            logger.info("Experiment Complete!")
            logger.info(f"Results saved to: {self.results_dir}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            raise


def main():
    parser = argparse.ArgumentParser(description='CAN IDS Experiment')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Run experiment
    experiment = CANIDSExperiment(config_path=args.config)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
