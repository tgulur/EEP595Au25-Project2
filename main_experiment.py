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
        
        # Create subdirectories for each model type
        self.model_dirs = {
            'voltage': self.results_dir / 'voltage',
            'cnn': self.results_dir / 'cnn',
            'lstm': self.results_dir / 'lstm',
            'fusion': self.results_dir / 'fusion',
            'comparison': self.results_dir / 'comparison'  # For comparison plots
        }
        for model_dir in self.model_dirs.values():
            model_dir.mkdir(parents=True, exist_ok=True)
        
        # Main evaluator for comparison plots (saved in comparison folder)
        self.evaluator = IDSEvaluator(save_dir=str(self.model_dirs['comparison']))
        self.results = {}
        
        logger.info(f"Experiment initialized. Results will be saved to {self.results_dir}")
        logger.info(f"Model-specific results will be organized in subdirectories")
    
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
            self.config['data'].get('canmap_path'),
            n_samples=self.config['data'].get('voltage_samples', 1000)
        )
        
        # Load CAN message dataset
        can_df = loader.load_road_dataset(
            self.config['data'].get('road_path'),
            n_samples=self.config['data'].get('can_samples', 10000)
        )
        
        # Preprocess voltage data
        logger.info("Preprocessing voltage data...")
        X_voltage, y_voltage = loader.preprocess_voltage_data(voltage_df)
        
        # Preprocess CAN message data
        logger.info("Preprocessing CAN message data...")
        sequence_length = self.config['deep_learning']['sequence_length']
        X_can, y_can, attack_types_can = loader.preprocess_can_data(can_df, sequence_length=sequence_length)
        
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
        
        # Split attack types using same indices
        train_indices = can_splits['train_indices']
        val_indices = can_splits['val_indices']
        test_indices = can_splits['test_indices']
        
        attack_types_splits = {
            'train': attack_types_can[train_indices],
            'val': attack_types_can[val_indices],
            'test': attack_types_can[test_indices]
        }
        
        logger.info("Data preparation complete")
        
        return {
            'voltage': voltage_splits,
            'can': can_splits,
            'attack_types': attack_types_splits,
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
        
        # Create model-specific evaluator
        voltage_evaluator = IDSEvaluator(save_dir=str(self.model_dirs['voltage']))
        
        # Calculate metrics
        metrics = voltage_evaluator.calculate_metrics(y_test, predictions, scores)
        
        # Visualizations
        voltage_evaluator.plot_confusion_matrix(
            y_test, predictions,
            title="Voltage Fingerprinting - Confusion Matrix",
            save_name="confusion_matrix.png"
        )
        voltage_evaluator.plot_roc_curve(
            y_test, scores,
            title="Voltage Fingerprinting - ROC Curve",
            save_name="roc_curve.png"
        )
        
        # Report
        report = voltage_evaluator.generate_report(
            "Voltage Fingerprinting",
            metrics,
            save_name="report.txt"
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
        
        # NOTE: Timing-Based and Frequency-Based IDS are disabled
        # They were not detecting attacks effectively (0% TPR)
        # Uncomment the code below if you want to re-enable them
        
        baseline_results = {}
        
        # # 1. Timing-Based IDS - DISABLED
        # if self.config['baselines'].get('timing_based', {}).get('enabled', True):
        #     logger.info("\nTraining Timing-Based IDS...")
        #     
        #     # Extract CAN data (timestamps, IDs, labels)
        #     can_train_data = data['can']['train']
        #     can_test_data = data['can']['test']
        #     
        #     # For baselines, we need raw CAN data with timestamps and IDs
        #     loader = CANDatasetLoader(self.config['data']['raw_path'])
        #     can_df = loader.load_road_dataset(self.config['data'].get('road_path'))
        #     
        #     # Split into train/test (matching our existing split)
        #     from sklearn.model_selection import train_test_split
        #     train_df, test_df = train_test_split(
        #         can_df, 
        #         test_size=(1 - self.config['data']['train_ratio']),
        #         random_state=self.config['data']['random_seed'],
        #         stratify=can_df['Label'] if 'Label' in can_df.columns else None
        #     )
        #     
        #     timing_ids = TimingBasedIDS(
        #         threshold=self.config['baselines']['timing_based'].get('threshold', 0.05)
        #     )
        #     
        #     # Extract timestamps, IDs, labels from training data
        #     train_timestamps = train_df['Timestamp'].values if 'Timestamp' in train_df.columns else np.arange(len(train_df))
        #     train_ids = train_df['ID'].values if 'ID' in train_df.columns else np.zeros(len(train_df))
        #     train_labels = train_df['Label'].values if 'Label' in train_df.columns else np.zeros(len(train_df))
        #     
        #     timing_ids.train(train_timestamps, train_ids, train_labels)
        #     
        #     # Predict on test data
        #     test_timestamps = test_df['Timestamp'].values if 'Timestamp' in test_df.columns else np.arange(len(test_df))
        #     test_ids = test_df['ID'].values if 'ID' in test_df.columns else np.zeros(len(test_df))
        #     test_labels = test_df['Label'].values if 'Label' in test_df.columns else np.zeros(len(test_df))
        #     
        #     predictions, scores = timing_ids.predict(test_timestamps, test_ids)
        #     
        #     # Evaluate
        #     metrics = self.evaluator.calculate_metrics(test_labels, predictions, scores)
        #     
        #     # Visualizations
        #     self.evaluator.plot_confusion_matrix(
        #         test_labels, predictions,
        #         title="Timing-Based IDS - Confusion Matrix",
        #         save_name="timing_confusion_matrix.png"
        #     )
        #     self.evaluator.plot_roc_curve(
        #         test_labels, scores,
        #         title="Timing-Based IDS - ROC Curve",
        #         save_name="timing_roc_curve.png"
        #     )
        #     
        #     # Report
        #     report = self.evaluator.generate_report(
        #         "Timing-Based IDS",
        #         metrics,
        #         save_name="timing_report.txt"
        #     )
        #     logger.info(report)
        #     
        #     baseline_results['Timing'] = {
        #         'metrics': metrics,
        #         'predictions': predictions,
        #         'scores': scores,
        #         'model': timing_ids,
        #         'y_test': test_labels
        #     }
        
        # # 2. Frequency-Based IDS - DISABLED
        # if self.config['baselines'].get('frequency_based', {}).get('enabled', True):
        #     logger.info("\nTraining Frequency-Based IDS...")
        #     
        #     freq_ids = FrequencyBasedIDS(
        #         window_size=self.config['baselines']['frequency_based'].get('window_size', 100),
        #         threshold=self.config['baselines']['frequency_based'].get('threshold', 0.3)
        #     )
        #     
        #     freq_ids.train(train_timestamps, train_ids, train_labels)
        #     predictions, scores = freq_ids.predict(test_timestamps, test_ids)
        #     
        #     # Evaluate
        #     metrics = self.evaluator.calculate_metrics(test_labels, predictions, scores)
        #     
        #     # Visualizations
        #     self.evaluator.plot_confusion_matrix(
        #         test_labels, predictions,
        #         title="Frequency-Based IDS - Confusion Matrix",
        #         save_name="frequency_confusion_matrix.png"
        #     )
        #     self.evaluator.plot_roc_curve(
        #         test_labels, scores,
        #         title="Frequency-Based IDS - ROC Curve",
        #         save_name="frequency_roc_curve.png"
        #     )
        #     
        #     # Report
        #     report = self.evaluator.generate_report(
        #         "Frequency-Based IDS",
        #         metrics,
        #         save_name="frequency_report.txt"
        #     )
        #     logger.info(report)
        #     
        #     baseline_results['Frequency'] = {
        #         'metrics': metrics,
        #         'predictions': predictions,
        #         'scores': scores,
        #         'model': freq_ids,
        #         'y_test': test_labels
        #     }
        
        self.results['baselines'] = baseline_results
        logger.info(f"\nBaseline models training complete. Trained {len(baseline_results)} models.")
        logger.info("Note: Timing-Based and Frequency-Based IDS are disabled (not detecting attacks effectively)")
        
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
            
            # Create model-specific evaluator
            model_dir = self.model_dirs[model_name.lower()]
            model_evaluator = IDSEvaluator(save_dir=str(model_dir))
            
            # Evaluate
            predictions, scores = model.predict(X_test)
            metrics = model_evaluator.calculate_metrics(y_test, predictions, scores)
            
            # Measure latency
            if self.config['evaluation']['measure_latency']:
                latency_metrics = model.predict_with_latency(
                    X_test[:1],
                    n_iterations=self.config['evaluation']['latency_iterations']
                )
                metrics.update(latency_metrics)
            
            # Visualizations
            model_evaluator.plot_confusion_matrix(
                y_test, predictions,
                title=f"{model_name} - Confusion Matrix",
                save_name="confusion_matrix.png"
            )
            model_evaluator.plot_roc_curve(
                y_test, scores,
                title=f"{model_name} - ROC Curve",
                save_name="roc_curve.png"
            )
            
            # Report
            report = model_evaluator.generate_report(
                model_name,
                metrics,
                save_name="report.txt"
            )
            logger.info(report)
            
            # Per-attack-type evaluation
            attack_types_test = data['attack_types']['test']
            attack_type_results = model_evaluator.evaluate_by_attack_type(
                y_test, predictions, attack_types_test, scores
            )
            attack_type_report = model_evaluator.generate_attack_type_report(
                attack_type_results,
                save_name="attack_type_report.txt"
            )
            logger.info("\n" + attack_type_report)
            
            # Generate per-attack-type visualization for this model
            model_evaluator.plot_model_attack_type_performance(
                attack_type_results,
                model_name=model_name,
                save_name="attack_type_performance.png"
            )
            
            # Save model
            model_path = model_dir / "model.h5"
            model.save_model(str(model_path))
            
            dl_results[model_name] = {
                'metrics': metrics,
                'predictions': predictions,
                'scores': scores,
                'model': model,
                'y_test': y_test,
                'attack_type_results': attack_type_results
            }
        
        self.results['deep_learning'] = dl_results
        
        return dl_results
    
    def train_fusion_layer(self, voltage_model, dl_models, data: dict):
        """Train fusion layer with improved test set size"""
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
        
        # Get DL scores from both CNN and LSTM, then combine
        cnn_scores = self.results['deep_learning']['CNN']['scores'][:min_samples]
        lstm_scores = self.results['deep_learning']['LSTM']['scores'][:min_samples]
        
        # Combine CNN and LSTM scores (weighted average: LSTM is better)
        dl_scores = 0.4 * cnn_scores + 0.6 * lstm_scores
        
        # Get confidences
        cnn_confidences = np.abs(cnn_scores - 0.5) * 2
        lstm_confidences = np.abs(lstm_scores - 0.5) * 2
        dl_confidences = 0.4 * cnn_confidences + 0.6 * lstm_confidences
        
        y_test = y_can_test[:min_samples]
        
        # Use larger test set: 60% train, 40% test (instead of 70/30)
        # This gives us more samples for evaluation
        split_idx = int(0.6 * len(voltage_scores))
        
        logger.info(f"Fusion training: {split_idx} samples for training, {len(voltage_scores) - split_idx} for testing")
        
        # Get attack types for fusion evaluation (need to align with min_samples and split_idx)
        attack_types_test = data['attack_types']['test']
        # Align attack types with the min_samples used
        attack_types_aligned = attack_types_test[:min_samples]
        
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
        
        # Create fusion-specific evaluator
        fusion_evaluator = IDSEvaluator(save_dir=str(self.model_dirs['fusion']))
        
        # Calculate metrics
        metrics = fusion_evaluator.calculate_metrics(
            y_test[split_idx:],
            predictions,
            confidences
        )
        
        # Visualizations
        fusion_evaluator.plot_confusion_matrix(
            y_test[split_idx:], predictions,
            title="Fusion Layer - Confusion Matrix",
            save_name="confusion_matrix.png"
        )
        fusion_evaluator.plot_roc_curve(
            y_test[split_idx:], confidences,
            title="Fusion Layer - ROC Curve",
            save_name="roc_curve.png"
        )
        
        # Report
        report = fusion_evaluator.generate_report(
            "Fusion Layer",
            metrics,
            save_name="report.txt"
        )
        logger.info(report)
        
        # Per-attack-type evaluation for fusion
        # Note: fusion uses subset of test data (after split_idx), so we need to slice attack_types accordingly
        if len(attack_types_aligned) > split_idx:
            attack_types_fusion = attack_types_aligned[split_idx:]
            # Ensure sizes match
            fusion_test_size = len(y_test[split_idx:])
            if len(attack_types_fusion) == fusion_test_size:
                attack_type_results = fusion_evaluator.evaluate_by_attack_type(
                    y_test[split_idx:], predictions, attack_types_fusion, confidences
                )
                attack_type_report = fusion_evaluator.generate_attack_type_report(
                    attack_type_results,
                    save_name="attack_type_report.txt"
                )
                logger.info("\n" + attack_type_report)
                
                # Generate per-attack-type visualization for fusion
                fusion_evaluator.plot_model_attack_type_performance(
                    attack_type_results,
                    model_name="Fusion",
                    save_name="attack_type_performance.png"
                )
            else:
                logger.warning(f"Size mismatch: attack_types={len(attack_types_fusion)}, y_test={fusion_test_size}. Skipping per-attack-type evaluation for fusion.")
                attack_type_results = {}
        else:
            logger.warning("Not enough test data for fusion attack type evaluation.")
            attack_type_results = {}
        
        self.results['fusion'] = {
            'metrics': metrics,
            'predictions': predictions,
            'scores': confidences,
            'model': fusion,
            'y_test': y_test[split_idx:],
            'attack_type_results': attack_type_results
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
        
        # Add baseline models if they exist (Timing/Frequency IDS are disabled)
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
                model_evaluator = IDSEvaluator(save_dir=str(self.model_dirs['voltage']))
            elif model_name == 'CNN':
                predictions = self.results['deep_learning']['CNN']['predictions']
                y_test = self.results['deep_learning']['CNN']['y_test']
                model_evaluator = IDSEvaluator(save_dir=str(self.model_dirs['cnn']))
            elif model_name == 'LSTM':
                predictions = self.results['deep_learning']['LSTM']['predictions']
                y_test = self.results['deep_learning']['LSTM']['y_test']
                model_evaluator = IDSEvaluator(save_dir=str(self.model_dirs['lstm']))
            else:  # Fusion
                predictions = self.results['fusion']['predictions']
                y_test = self.results['fusion']['y_test']
                model_evaluator = IDSEvaluator(save_dir=str(self.model_dirs['fusion']))
            
            model_evaluator.plot_attack_detection_timeline(
                y_test,
                predictions,
                model_name=model_name,
                save_name="attack_timeline.png"
            )
        
        # Create detection heatmap showing which models detected which attacks
        # Align array sizes: voltage and CAN data have different test set sizes
        logger.info("Generating detection heatmap...")
        y_test_voltage = self.results['voltage']['y_test']
        pred_voltage = self.results['voltage']['predictions']
        pred_cnn = self.results['deep_learning']['CNN']['predictions']
        pred_lstm = self.results['deep_learning']['LSTM']['predictions']
        
        # Calculate minimum samples (same logic as fusion layer)
        min_samples = min(len(y_test_voltage), len(pred_cnn), len(pred_lstm))
        
        if min_samples < len(y_test_voltage) or min_samples < len(pred_cnn) or min_samples < len(pred_lstm):
            logger.info(f"Aligning arrays to minimum size: {min_samples} samples "
                       f"(voltage: {len(y_test_voltage)}, CNN: {len(pred_cnn)}, LSTM: {len(pred_lstm)})")
        
        # Slice all arrays to min_samples
        y_test_ref = y_test_voltage[:min_samples]
        predictions_dict = {
            'Voltage': pred_voltage[:min_samples],
            'CNN': pred_cnn[:min_samples],
            'LSTM': pred_lstm[:min_samples],
        }
        
        # Save heatmap in comparison folder
        self.evaluator.plot_detection_heatmap(
            y_test_ref,
            predictions_dict,
            save_name="detection_heatmap.png"
        )
        
        # Create attack type comparison visualization
        logger.info("Generating attack type comparison visualization...")
        attack_type_results_dict = {}
        
        # Collect attack type results from all models
        if 'deep_learning' in self.results:
            if 'CNN' in self.results['deep_learning'] and 'attack_type_results' in self.results['deep_learning']['CNN']:
                attack_type_results_dict['CNN'] = self.results['deep_learning']['CNN']['attack_type_results']
            if 'LSTM' in self.results['deep_learning'] and 'attack_type_results' in self.results['deep_learning']['LSTM']:
                attack_type_results_dict['LSTM'] = self.results['deep_learning']['LSTM']['attack_type_results']
        
        if 'fusion' in self.results and 'attack_type_results' in self.results['fusion']:
            attack_type_results_dict['Fusion'] = self.results['fusion']['attack_type_results']
        
        # Generate visualization if we have results
        if attack_type_results_dict:
            self.evaluator.plot_attack_type_comparison(
                attack_type_results_dict,
                save_name="attack_type_comparison.png"
            )
            logger.info("✓ Attack type comparison visualization generated")
        else:
            logger.warning("No attack type results available for comparison visualization")
        
        # Create comprehensive comparison dashboard
        # Align array sizes: voltage and CAN data have different test set sizes
        logger.info("Generating comprehensive comparison dashboard...")
        y_test_voltage = self.results['voltage']['y_test']
        pred_voltage = self.results['voltage']['predictions']
        pred_cnn = self.results['deep_learning']['CNN']['predictions']
        pred_lstm = self.results['deep_learning']['LSTM']['predictions']
        
        # Calculate minimum samples (same logic as fusion layer)
        min_samples = min(len(y_test_voltage), len(pred_cnn), len(pred_lstm))
        
        if min_samples < len(y_test_voltage) or min_samples < len(pred_cnn) or min_samples < len(pred_lstm):
            logger.info(f"Aligning arrays to minimum size: {min_samples} samples "
                       f"(voltage: {len(y_test_voltage)}, CNN: {len(pred_cnn)}, LSTM: {len(pred_lstm)})")
        
        # Slice all arrays to min_samples
        y_test_ref = y_test_voltage[:min_samples]
        comprehensive_results = {
            'Voltage': {
                'metrics': self.results['voltage']['metrics'],
                'predictions': pred_voltage[:min_samples]
            },
            'CNN': {
                'metrics': self.results['deep_learning']['CNN']['metrics'],
                'predictions': pred_cnn[:min_samples]
            },
            'LSTM': {
                'metrics': self.results['deep_learning']['LSTM']['metrics'],
                'predictions': pred_lstm[:min_samples]
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
        
        # Baseline models (Timing/Frequency IDS are disabled - not detecting attacks effectively)
        # if 'baselines' in self.results:
        #     for name, data in self.results['baselines'].items():
        #         all_models.append((f"{name}-Based IDS", data['metrics'], 'Baseline'))
        
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
        
        # Save to file in comparison folder
        comparison_file = self.model_dirs['comparison'] / "model_comparison_table.txt"
        with open(comparison_file, 'w') as f:
            f.write('\n'.join(table_lines))
        
        logger.info(f"Comparison table saved to {comparison_file}")
    
    def run_ablation_study(self):
        """Run fusion ablation study automatically after experiment"""
        logger.info("\n" + "="*60)
        logger.info("Running Fusion Ablation Study")
        logger.info("="*60)
        
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create ablation results directory
            ablation_dir = self.results_dir / 'ablation'
            ablation_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract metrics from current results (no need to parse files)
            individual_results = {
                'voltage': self.results['voltage']['metrics'],
                'cnn': self.results['deep_learning']['CNN']['metrics'],
                'lstm': self.results['deep_learning']['LSTM']['metrics'],
                'fusion': self.results['fusion']['metrics']
            }
            
            # Get metrics
            v_acc = individual_results['voltage'].get('accuracy', 0.0)
            c_acc = individual_results['cnn'].get('accuracy', 0.0)
            l_acc = individual_results['lstm'].get('accuracy', 0.0)
            
            v_tpr = individual_results['voltage'].get('true_positive_rate', 
                    individual_results['voltage'].get('tpr', 0.0))
            c_tpr = individual_results['cnn'].get('true_positive_rate', 
                    individual_results['cnn'].get('tpr', 0.0))
            l_tpr = individual_results['lstm'].get('true_positive_rate', 
                    individual_results['lstm'].get('tpr', 0.0))
            
            v_fpr = individual_results['voltage'].get('false_positive_rate', 
                    individual_results['voltage'].get('fpr', 0.0))
            c_fpr = individual_results['cnn'].get('false_positive_rate', 
                    individual_results['cnn'].get('fpr', 0.0))
            l_fpr = individual_results['lstm'].get('false_positive_rate', 
                    individual_results['lstm'].get('fpr', 0.0))
            
            # Fusion metrics
            f_acc = individual_results['fusion'].get('accuracy', 0.0)
            f_tpr = individual_results['fusion'].get('true_positive_rate', 
                    individual_results['fusion'].get('tpr', 0.0))
            f_fpr = individual_results['fusion'].get('false_positive_rate', 
                    individual_results['fusion'].get('fpr', 0.0))
            
            combinations = {}
            
            # Individual models
            combinations['Voltage Only'] = {
                'components': ['V'],
                'accuracy': v_acc,
                'tpr': v_tpr,
                'fpr': v_fpr,
                'description': 'Pure voltage fingerprinting'
            }
            
            combinations['CNN Only'] = {
                'components': ['C'],
                'accuracy': c_acc,
                'tpr': c_tpr,
                'fpr': c_fpr,
                'description': 'Pure CNN deep learning'
            }
            
            combinations['LSTM Only'] = {
                'components': ['L'],
                'accuracy': l_acc,
                'tpr': l_tpr,
                'fpr': l_fpr,
                'description': 'Pure LSTM deep learning'
            }
            
            # Pairwise combinations (weighted average)
            combinations['Voltage + CNN'] = {
                'components': ['V', 'C'],
                'accuracy': (v_acc * 0.3 + c_acc * 0.7),
                'tpr': (v_tpr * 0.3 + c_tpr * 0.7),
                'fpr': (v_fpr * 0.3 + c_fpr * 0.7),
                'description': 'Voltage + CNN fusion'
            }
            
            combinations['Voltage + LSTM'] = {
                'components': ['V', 'L'],
                'accuracy': (v_acc * 0.3 + l_acc * 0.7),
                'tpr': (v_tpr * 0.3 + l_tpr * 0.7),
                'fpr': (v_fpr * 0.3 + l_fpr * 0.7),
                'description': 'Voltage + LSTM fusion'
            }
            
            combinations['CNN + LSTM'] = {
                'components': ['C', 'L'],
                'accuracy': (c_acc * 0.5 + l_acc * 0.5),
                'tpr': (c_tpr * 0.5 + l_tpr * 0.5),
                'fpr': (c_fpr * 0.5 + l_fpr * 0.5),
                'description': 'Deep learning fusion (no voltage)'
            }
            
            # Full fusion (from actual results)
            combinations['Full Fusion'] = {
                'components': ['V', 'C', 'L'],
                'accuracy': f_acc,
                'tpr': f_tpr,
                'fpr': f_fpr,
                'description': 'All components combined'
            }
            
            # Generate table
            data = []
            for name, res in combinations.items():
                data.append({
                    'Configuration': name,
                    'Components': '+'.join(res['components']),
                    'Accuracy': f"{res['accuracy']:.4f}",
                    'TPR': f"{res['tpr']:.4f}",
                    'FPR': f"{res['fpr']:.4f}",
                    'Description': res['description']
                })
            
            df = pd.DataFrame(data)
            
            # Save results
            output_file = ablation_dir / "ablation_results.txt"
            with open(output_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("FUSION ABLATION STUDY RESULTS\n")
                f.write("="*80 + "\n\n")
                f.write(df.to_string(index=False))
                f.write("\n\n")
                
                # Add analysis
                f.write("="*80 + "\n")
                f.write("ANALYSIS\n")
                f.write("="*80 + "\n\n")
                
                # Best single component
                single_comps = {k: v for k, v in combinations.items() if len(v['components']) == 1}
                best_single = max(single_comps.items(), key=lambda x: x[1]['accuracy'])
                f.write(f"Best Single Component: {best_single[0]}\n")
                f.write(f"  Accuracy: {best_single[1]['accuracy']:.4f}\n\n")
                
                # Best pairwise
                pair_comps = {k: v for k, v in combinations.items() if len(v['components']) == 2}
                if pair_comps:
                    best_pair = max(pair_comps.items(), key=lambda x: x[1]['accuracy'])
                    f.write(f"Best Pairwise Combination: {best_pair[0]}\n")
                    f.write(f"  Accuracy: {best_pair[1]['accuracy']:.4f}\n\n")
                
                # Full fusion
                full = combinations['Full Fusion']
                f.write(f"Full Fusion (All Components): {full['accuracy']:.4f}\n\n")
                
                # Value of voltage
                if 'CNN + LSTM' in combinations and 'Full Fusion' in combinations:
                    dl_only = combinations['CNN + LSTM']['accuracy']
                    full_fusion = combinations['Full Fusion']['accuracy']
                    voltage_contribution = full_fusion - dl_only
                    f.write(f"\nVoltage Fingerprinting Contribution:\n")
                    f.write(f"  DL-only (CNN+LSTM): {dl_only:.4f}\n")
                    f.write(f"  Full Fusion (V+C+L): {full_fusion:.4f}\n")
                    f.write(f"  Improvement: {voltage_contribution:+.4f}\n")
                    if voltage_contribution > 0.001:
                        f.write(f"  → Voltage adds value! ✓\n")
                    else:
                        f.write(f"  → DL models alone are sufficient (voltage adds minimal value)\n")
                
                f.write("\n")
                f.write("NOTE: Pairwise combinations are simulated using weighted averaging.\n")
                f.write("For exact results, the fusion layer would need to be retrained with each\n")
                f.write("specific component combination.\n")
            
            logger.info(f"\n✓ Ablation results saved to {output_file}")
            
            # Generate visualization
            configs = list(combinations.keys())
            accuracies = [combinations[c]['accuracy'] for c in configs]
            tprs = [combinations[c]['tpr'] for c in configs]
            fprs = [combinations[c]['fpr'] for c in configs]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle('Fusion Ablation Study Results', fontsize=16, fontweight='bold')
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
            
            # Plot 1: Accuracy comparison
            ax1 = axes[0]
            bars1 = ax1.barh(configs, accuracies, color=colors[:len(configs)])
            ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
            ax1.set_title('Accuracy by Configuration', fontsize=14, fontweight='bold')
            ax1.set_xlim([0.7, 1.0])
            for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
                ax1.text(acc + 0.005, i, f'{acc:.3f}', va='center', fontsize=9)
            ax1.grid(axis='x', alpha=0.3)
            
            # Plot 2: TPR vs FPR
            ax2 = axes[1]
            scatter = ax2.scatter(fprs, tprs, s=200, c=range(len(configs)), cmap='viridis', 
                                  alpha=0.6, edgecolors='black', linewidth=2)
            for i, config in enumerate(configs):
                ax2.annotate(config, (fprs[i], tprs[i]), fontsize=8, 
                             xytext=(5, 5), textcoords='offset points')
            ax2.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
            ax2.set_title('TPR vs FPR Trade-off', fontsize=14, fontweight='bold')
            ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
            ax2.set_xlim([0, max(0.3, max(fprs) * 1.2)])
            ax2.set_ylim([0.7, 1.05])
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            # Plot 3: Component contribution heatmap
            ax3 = axes[2]
            component_names = ['Voltage', 'CNN', 'LSTM']
            components_matrix = []
            for config in configs:
                row = [1 if 'V' in combinations[config]['components'] else 0,
                       1 if 'C' in combinations[config]['components'] else 0,
                       1 if 'L' in combinations[config]['components'] else 0]
                components_matrix.append(row)
            
            accuracy_matrix = [[acc if val else 0 for val in row] 
                             for acc, row in zip(accuracies, components_matrix)]
            
            sns.heatmap(accuracy_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                        xticklabels=component_names, yticklabels=configs,
                        cbar_kws={'label': 'Accuracy'}, ax=ax3, vmin=0, vmax=1.0)
            ax3.set_title('Component Usage & Performance', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Components', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            # Save
            output_file = ablation_dir / "ablation_comparison.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"✓ Ablation visualization saved to {output_file}")
            logger.info("\n" + "="*60)
            logger.info("Ablation Study Complete!")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Error in ablation study: {e}", exc_info=True)
            logger.warning("Continuing despite ablation study error...")
    
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
            
            # Run ablation study automatically
            self.run_ablation_study()
            
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
