"""
Evaluation metrics and visualization for IDS performance
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from typing import Dict, Tuple, Optional, List
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IDSEvaluator:
    """Handles all the evaluation metrics and plots"""
    
    def __init__(self, save_dir: str = "results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_score: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate all the important metrics"""
        metrics = {}
        
        # Standard classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Break down the confusion matrix - handle when only one class is predicted
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            # Only one class present in predictions
            tn, fp, fn, tp = 0, 0, 0, 0
            if len(np.unique(y_pred)) == 1:
                # All predicted as one class
                if y_pred[0] == 0:
                    tn = np.sum(y_true == 0)
                    fn = np.sum(y_true == 1)
                else:
                    tp = np.sum(y_true == 1)
                    fp = np.sum(y_true == 0)
        
        metrics['true_positive'] = int(tp)
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        
        # Calculate rates
        metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['true_negative_rate'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Alternative names for same metrics
        metrics['specificity'] = metrics['true_negative_rate']
        metrics['sensitivity'] = metrics['true_positive_rate']
        
        # Detection rate
        total_attacks = np.sum(y_true == 1)
        if total_attacks > 0:
            metrics['detection_rate'] = tp / total_attacks
        else:
            metrics['detection_rate'] = 0.0
        
        # ROC AUC if scores provided
        if y_score is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_score)
                metrics['average_precision'] = average_precision_score(y_true, y_score)
            except:
                metrics['roc_auc'] = 0.0
                metrics['average_precision'] = 0.0
        
        return metrics
    
    def calculate_latency_metrics(self, latency_measurements: List[float]) -> Dict[str, float]:
        """Calculate latency stats - important for real-time systems"""
        latencies = np.array(latency_measurements)
        
        return {
            'mean_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99))
        }
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             title: str = "Confusion Matrix",
                             save_name: Optional[str] = None):
        """Draw a nice confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray,
                      title: str = "ROC Curve",
                      save_name: Optional[str] = None):
        """Plot ROC curve to visualize TPR vs FPR tradeoff"""
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved to {save_path}")
        
        plt.close()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_score: np.ndarray,
                                   title: str = "Precision-Recall Curve",
                                   save_name: Optional[str] = None):
        """Plot precision-recall curve - good for imbalanced datasets"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        avg_precision = average_precision_score(y_true, y_score)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved to {save_path}")
        
        plt.close()
    
    def compare_models(self, results: Dict[str, Dict[str, float]],
                      metric_names: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
                      save_name: Optional[str] = None):
        """Make a comparison bar chart for different models"""
        models = list(results.keys())
        metrics_data = {metric: [] for metric in metric_names}
        
        for model in models:
            for metric in metric_names:
                metrics_data[metric].append(results[model].get(metric, 0))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(models))
        width = 0.8 / len(metric_names)
        
        for i, metric in enumerate(metric_names):
            offset = width * i - (width * len(metric_names)) / 2 + width / 2
            ax.bar(x + offset, metrics_data[metric], width, 
                  label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved to {save_path}")
        
        plt.close()
    
    def plot_latency_comparison(self, latency_results: Dict[str, Dict[str, float]],
                               save_name: Optional[str] = None):
        """
        Plot latency comparison across models
        
        Args:
            latency_results: Dictionary of model_name -> latency_metrics
            save_name: Filename to save plot
        """
        models = list(latency_results.keys())
        mean_latencies = [latency_results[m]['mean_latency_ms'] for m in models]
        std_latencies = [latency_results[m]['std_latency_ms'] for m in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(models))
        ax.bar(x, mean_latencies, yerr=std_latencies, capsize=5, 
              color='skyblue', edgecolor='navy')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Model Latency Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Latency comparison plot saved to {save_path}")
        
        plt.close()
    
    def plot_attack_detection_timeline(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      timestamps: Optional[np.ndarray] = None,
                                      model_name: Optional[str] = None,
                                      save_name: Optional[str] = None):
        """
        Plot attack detection timeline showing when attacks occur and when they're detected
        
        Args:
            y_true: True labels (0=normal, 1=attack)
            y_pred: Predicted labels
            timestamps: Optional timestamps for x-axis
            model_name: Name of the model being evaluated
            save_name: Filename to save plot
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
        
        if timestamps is None:
            timestamps = np.arange(len(y_true))
        
        # Plot 1: Ground truth (what actually happened)
        ax1.fill_between(timestamps, 0, y_true, step='mid', alpha=0.6, color='red', label='Actual Attacks')
        ax1.fill_between(timestamps, 0, 1-y_true, step='mid', alpha=0.6, color='green', label='Normal Traffic')
        ax1.set_ylabel('Actual State', fontweight='bold', fontsize=12)
        
        # Add main title and prominent model name
        if model_name:
            fig.suptitle(f'Attack Detection Timeline Analysis', fontsize=16, fontweight='bold', y=0.995)
            ax1.text(0.5, 1.15, f'MODEL: {model_name.upper()}', transform=ax1.transAxes,
                    fontsize=18, fontweight='bold', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='black', linewidth=2))
        else:
            fig.suptitle(f'Attack Detection Timeline Analysis', fontsize=16, fontweight='bold', y=0.995)
        ax1.text(0.01, 0.5, '← What Actually Happened', transform=ax1.transAxes, 
                fontsize=10, style='italic', verticalalignment='center')
        ax1.text(0.01, 0.5, '← What Actually Happened', transform=ax1.transAxes, 
                fontsize=10, style='italic', verticalalignment='center')
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['Normal', 'Attack'], fontsize=11)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Model predictions (what the model detected)
        ax2.fill_between(timestamps, 0, y_pred, step='mid', alpha=0.6, color='orange', label='Predicted Attacks')
        ax2.fill_between(timestamps, 0, 1-y_pred, step='mid', alpha=0.6, color='blue', label='Predicted Normal')
        ax2.set_ylabel('Predicted State', fontweight='bold', fontsize=12)
        ax2.text(0.01, 0.5, '← What Model Detected', transform=ax2.transAxes, 
                fontsize=10, style='italic', verticalalignment='center')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Normal', 'Attack'], fontsize=11)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Plot 3: Detection accuracy (did the model get it right?)
        correct = (y_true == y_pred).astype(int)
        colors = ['red' if c == 0 else 'green' for c in correct]
        ax3.scatter(timestamps, correct, c=colors, s=5, alpha=0.6)
        ax3.fill_between(timestamps, 0, correct, step='mid', alpha=0.3, color='green')
        ax3.set_ylabel('Detection', fontweight='bold', fontsize=12)
        ax3.text(0.01, 0.5, '← Model Accuracy', transform=ax3.transAxes, 
                fontsize=10, style='italic', verticalalignment='center')
        ax3.set_xlabel('Time/Sample Index', fontweight='bold', fontsize=12)
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['Wrong', 'Correct'], fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # Calculate and display metrics
        accuracy = np.mean(correct)
        attack_indices = np.where(y_true == 1)[0]
        detected_attacks = np.sum((y_true == 1) & (y_pred == 1))
        total_attacks = np.sum(y_true == 1)
        detection_rate = detected_attacks / total_attacks if total_attacks > 0 else 0
        
        # Calculate missed and false alarms
        missed_attacks = np.sum((y_true == 1) & (y_pred == 0))
        false_alarms = np.sum((y_true == 0) & (y_pred == 1))
        
        textstr = (f'Overall Accuracy: {accuracy:.2%}\n'
                  f'Attacks Detected: {detected_attacks}/{total_attacks} ({detection_rate:.2%})\n'
                  f'Missed Attacks: {missed_attacks}\n'
                  f'False Alarms: {false_alarms}')
        ax3.text(0.02, 0.95, textstr, transform=ax3.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attack detection timeline saved to {save_path}")
        
        plt.close()
    
    def plot_detection_heatmap(self, y_true: np.ndarray, predictions_dict: Dict[str, np.ndarray],
                              save_name: Optional[str] = None):
        """
        Plot heatmap showing which models detected which attacks
        
        Args:
            y_true: True labels
            predictions_dict: Dictionary of model_name -> predictions
            save_name: Filename to save plot
        """
        # Validate array sizes match
        y_true_len = len(y_true)
        for model_name, y_pred in predictions_dict.items():
            if len(y_pred) != y_true_len:
                raise ValueError(
                    f"Array size mismatch in plot_detection_heatmap: "
                    f"y_true has {y_true_len} samples, but {model_name} predictions have {len(y_pred)} samples. "
                    f"Please align arrays to the same size before calling this function."
                )
        
        # Find attack indices
        attack_indices = np.where(y_true == 1)[0]
        
        if len(attack_indices) == 0:
            logger.warning("No attacks in dataset for heatmap visualization")
            return
        
        # Create matrix: rows=attacks, columns=models
        models = list(predictions_dict.keys())
        detection_matrix = np.zeros((len(attack_indices), len(models)))
        
        for j, model_name in enumerate(models):
            y_pred = predictions_dict[model_name]
            for i, idx in enumerate(attack_indices):
                detection_matrix[i, j] = y_pred[idx]
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(max(10, len(models)*2), max(8, len(attack_indices)//5)))
        
        im = ax.imshow(detection_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(models)))
        ax.set_yticks(np.arange(min(len(attack_indices), 50)))  # Limit to 50 attacks shown
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_yticklabels([f'Attack {idx}' for idx in attack_indices[:50]])
        
        ax.set_xlabel('Model', fontweight='bold', fontsize=12)
        ax.set_ylabel('Attack Samples', fontweight='bold', fontsize=12)
        ax.set_title('Attack Detection Heatmap - Which Models Detected Each Attack\n(Green=Detected, Red=Missed)', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Detected (1) / Missed (0)', rotation=270, labelpad=20)
        
        # Add detection statistics (recall rate on attacks only)
        detection_rates = detection_matrix.mean(axis=0)
        for j, (model, rate) in enumerate(zip(models, detection_rates)):
            ax.text(j, len(attack_indices) + 2, f'{rate:.1%}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Add note about metric
        ax.text(0.5, -0.08, 'Note: Percentages show Attack Detection Rate (Recall) - portion of actual attacks detected',
               ha='center', va='top', transform=ax.transAxes, fontsize=9, style='italic')
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Detection heatmap saved to {save_path}")
        
        plt.close()
    
    def plot_comprehensive_comparison(self, results_dict: Dict[str, Dict],
                                     y_true: np.ndarray,
                                     save_name: Optional[str] = None):
        """
        Create comprehensive multi-panel comparison of all models
        
        Args:
            results_dict: Dictionary with model results (metrics and predictions)
            y_true: True labels
            save_name: Filename to save plot
        """
        # Validate array sizes match
        y_true_len = len(y_true)
        for model_name, model_data in results_dict.items():
            if 'predictions' in model_data:
                y_pred = model_data['predictions']
                if len(y_pred) != y_true_len:
                    raise ValueError(
                        f"Array size mismatch in plot_comprehensive_comparison: "
                        f"y_true has {y_true_len} samples, but {model_name} predictions have {len(y_pred)} samples. "
                        f"Please align arrays to the same size before calling this function."
                    )
        
        models = list(results_dict.keys())
        n_models = len(models)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Accuracy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        accuracies = [results_dict[m]['metrics']['accuracy'] for m in models]
        colors_acc = ['green' if a > 0.9 else 'orange' if a > 0.7 else 'red' for a in accuracies]
        bars1 = ax1.bar(range(n_models), accuracies, color=colors_acc, alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(n_models))
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Model Accuracy Comparison', fontweight='bold')
        ax1.set_ylim(0, 1.0)
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='90% threshold')
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2%}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Precision vs Recall
        ax2 = fig.add_subplot(gs[0, 1])
        precisions = [results_dict[m]['metrics']['precision'] for m in models]
        recalls = [results_dict[m]['metrics']['recall'] for m in models]
        scatter = ax2.scatter(recalls, precisions, s=200, c=range(n_models), 
                            cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
        for i, model in enumerate(models):
            ax2.annotate(model, (recalls[i], precisions[i]), fontsize=9, ha='center')
        ax2.set_xlabel('Recall (Attack Detection Rate)', fontweight='bold')
        ax2.set_ylabel('Precision', fontweight='bold')
        ax2.set_title('Precision vs Recall Trade-off', fontweight='bold')
        ax2.set_xlim(0, 1.05)
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.axline((0, 0), (1, 1), color='gray', linestyle='--', alpha=0.5)
        
        # Plot 3: F1 Scores
        ax3 = fig.add_subplot(gs[0, 2])
        f1_scores = [results_dict[m]['metrics']['f1_score'] for m in models]
        bars3 = ax3.bar(range(n_models), f1_scores, color='skyblue', alpha=0.7, edgecolor='navy')
        ax3.set_xticks(range(n_models))
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.set_ylabel('F1-Score', fontweight='bold')
        ax3.set_title('F1-Score Comparison', fontweight='bold')
        ax3.set_ylim(0, 1.0)
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, f1 in zip(bars3, f1_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: True Positive Rate (Attack Detection)
        ax4 = fig.add_subplot(gs[1, 0])
        tprs = [results_dict[m]['metrics'].get('true_positive_rate', 0) for m in models]
        colors_tpr = ['darkgreen' if t > 0.9 else 'orange' if t > 0.7 else 'red' for t in tprs]
        bars4 = ax4.bar(range(n_models), tprs, color=colors_tpr, alpha=0.7, edgecolor='black')
        ax4.set_xticks(range(n_models))
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.set_ylabel('True Positive Rate', fontweight='bold')
        ax4.set_title('Attack Detection Rate (Sensitivity)', fontweight='bold')
        ax4.set_ylim(0, 1.0)
        ax4.grid(axis='y', alpha=0.3)
        
        for bar, tpr in zip(bars4, tprs):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{tpr:.2%}', ha='center', va='bottom', fontsize=9)
        
        # Plot 5: False Positive Rate
        ax5 = fig.add_subplot(gs[1, 1])
        fprs = [results_dict[m]['metrics'].get('false_positive_rate', 0) for m in models]
        colors_fpr = ['green' if f < 0.1 else 'orange' if f < 0.3 else 'red' for f in fprs]
        bars5 = ax5.bar(range(n_models), fprs, color=colors_fpr, alpha=0.7, edgecolor='black')
        ax5.set_xticks(range(n_models))
        ax5.set_xticklabels(models, rotation=45, ha='right')
        ax5.set_ylabel('False Positive Rate', fontweight='bold')
        ax5.set_title('False Alarm Rate', fontweight='bold')
        ax5.set_ylim(0, max(fprs) * 1.2 if fprs else 0.5)
        ax5.grid(axis='y', alpha=0.3)
        
        for bar, fpr in zip(bars5, fprs):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{fpr:.2%}', ha='center', va='bottom', fontsize=9)
        
        # Plot 6: Confusion Matrix Summary
        ax6 = fig.add_subplot(gs[1, 2])
        conf_data = []
        for model in models:
            metrics = results_dict[model]['metrics']
            tp = metrics.get('true_positive', 0)
            tn = metrics.get('true_negative', 0)
            fp = metrics.get('false_positive', 0)
            fn = metrics.get('false_negative', 0)
            total = tp + tn + fp + fn
            conf_data.append([tp/total if total > 0 else 0, 
                            fn/total if total > 0 else 0,
                            fp/total if total > 0 else 0,
                            tn/total if total > 0 else 0])
        
        conf_data = np.array(conf_data)
        im6 = ax6.imshow(conf_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax6.set_xticks(range(4))
        ax6.set_xticklabels(['TP', 'FN', 'FP', 'TN'], fontweight='bold')
        ax6.set_yticks(range(n_models))
        ax6.set_yticklabels(models)
        ax6.set_title('Normalized Confusion Matrix', fontweight='bold')
        plt.colorbar(im6, ax=ax6, label='Proportion')
        
        # Plot 7: Detection latency (if available)
        ax7 = fig.add_subplot(gs[2, :])
        has_latency = any('mean_latency_ms' in results_dict[m]['metrics'] for m in models)
        
        if has_latency:
            latencies = [results_dict[m]['metrics'].get('mean_latency_ms', 0) for m in models]
            bars7 = ax7.bar(range(n_models), latencies, color='coral', alpha=0.7, edgecolor='darkred')
            ax7.set_xticks(range(n_models))
            ax7.set_xticklabels(models, rotation=45, ha='right')
            ax7.set_ylabel('Latency (ms)', fontweight='bold')
            ax7.set_title('Average Detection Latency', fontweight='bold')
            ax7.grid(axis='y', alpha=0.3)
            
            for bar, lat in zip(bars7, latencies):
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height,
                        f'{lat:.1f}ms', ha='center', va='bottom', fontsize=9)
        else:
            # Show detection counts instead
            attack_counts = [np.sum(y_true == 1)] * n_models
            detected_counts = []
            for model in models:
                if 'predictions' in results_dict[model]:
                    y_pred = results_dict[model]['predictions']
                    detected = np.sum((y_true == 1) & (y_pred == 1))
                    detected_counts.append(detected)
                else:
                    detected_counts.append(0)
            
            x = np.arange(n_models)
            width = 0.35
            bars7a = ax7.bar(x - width/2, attack_counts, width, label='Total Attacks', 
                            color='red', alpha=0.6)
            bars7b = ax7.bar(x + width/2, detected_counts, width, label='Detected', 
                            color='green', alpha=0.6)
            ax7.set_xticks(x)
            ax7.set_xticklabels(models, rotation=45, ha='right')
            ax7.set_ylabel('Count', fontweight='bold')
            ax7.set_title('Attack Detection Counts', fontweight='bold')
            ax7.legend()
            ax7.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Comprehensive Model Performance Comparison', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comprehensive comparison saved to {save_path}")
        
        plt.close()
    
    def generate_report(self, model_name: str, metrics: Dict[str, float],
                       save_name: Optional[str] = None) -> str:
        """
        Generate text report of evaluation results
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
            save_name: Filename to save report
            
        Returns:
            Report string
        """
        report = f"""
{'='*60}
Evaluation Report: {model_name}
{'='*60}

Classification Metrics:
  Accuracy:      {metrics.get('accuracy', 0):.4f}
  Precision:     {metrics.get('precision', 0):.4f}
  Recall:        {metrics.get('recall', 0):.4f}
  F1-Score:      {metrics.get('f1_score', 0):.4f}

Detection Rates:
  True Positive Rate (TPR):   {metrics.get('true_positive_rate', 0):.4f}
  False Positive Rate (FPR):  {metrics.get('false_positive_rate', 0):.4f}
  True Negative Rate (TNR):   {metrics.get('true_negative_rate', 0):.4f}
  False Negative Rate (FNR):  {metrics.get('false_negative_rate', 0):.4f}

Confusion Matrix:
  True Positives:  {metrics.get('true_positive', 0)}
  True Negatives:  {metrics.get('true_negative', 0)}
  False Positives: {metrics.get('false_positive', 0)}
  False Negatives: {metrics.get('false_negative', 0)}

Additional Metrics:
  Specificity:       {metrics.get('specificity', 0):.4f}
  Sensitivity:       {metrics.get('sensitivity', 0):.4f}
  Detection Rate:    {metrics.get('detection_rate', 0):.4f}
"""
        
        if 'roc_auc' in metrics:
            report += f"  ROC AUC:           {metrics['roc_auc']:.4f}\n"
        
        if 'average_precision' in metrics:
            report += f"  Average Precision: {metrics['average_precision']:.4f}\n"
        
        if 'mean_latency_ms' in metrics:
            report += f"\nLatency Metrics:\n"
            report += f"  Mean:    {metrics['mean_latency_ms']:.3f} ms\n"
            report += f"  Median:  {metrics.get('median_latency_ms', 0):.3f} ms\n"
            report += f"  Std Dev: {metrics.get('std_latency_ms', 0):.3f} ms\n"
            report += f"  P95:     {metrics.get('p95_latency_ms', 0):.3f} ms\n"
            report += f"  P99:     {metrics.get('p99_latency_ms', 0):.3f} ms\n"
        
        report += f"\n{'='*60}\n"
        
        if save_name:
            save_path = self.save_dir / save_name
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {save_path}")
        
        return report
    
    def evaluate_by_attack_type(self, y_true: np.ndarray, y_pred: np.ndarray,
                                attack_types: np.ndarray,
                                y_score: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance for each attack type separately.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            attack_types: Attack type for each sample ('normal', 'spoofing', 'dos', etc.)
            y_score: Optional confidence scores
            
        Returns:
            Dictionary mapping attack type to metrics
        """
        results = {}
        
        # Get unique attack types
        unique_types = np.unique(attack_types)
        
        for attack_type in unique_types:
            # Create mask for this attack type
            mask = attack_types == attack_type
            
            if np.sum(mask) == 0:
                continue
            
            # Get subset of data for this attack type
            y_true_subset = y_true[mask]
            y_pred_subset = y_pred[mask]
            y_score_subset = y_score[mask] if y_score is not None else None
            
            # Calculate metrics for this attack type
            metrics = self.calculate_metrics(y_true_subset, y_pred_subset, y_score_subset)
            
            # Add sample count
            metrics['sample_count'] = int(np.sum(mask))
            metrics['attack_count'] = int(np.sum(y_true_subset == 1))
            metrics['normal_count'] = int(np.sum(y_true_subset == 0))
            
            results[attack_type] = metrics
        
        return results
    
    def generate_attack_type_report(self, attack_type_results: Dict[str, Dict[str, float]],
                                   save_name: Optional[str] = None) -> str:
        """
        Generate a report showing performance for each attack type.
        
        Args:
            attack_type_results: Results from evaluate_by_attack_type
            save_name: Optional filename to save report
            
        Returns:
            Report string
        """
        report = f"""
{'='*80}
PER-ATTACK-TYPE EVALUATION REPORT
{'='*80}

"""
        
        # Create table
        report += f"{'Attack Type':<15} {'Samples':<10} {'Attacks':<10} {'Accuracy':<12} {'TPR':<10} {'FPR':<10} {'Precision':<12} {'Recall':<10} {'F1-Score':<10}\n"
        report += "-" * 80 + "\n"
        
        for attack_type, metrics in sorted(attack_type_results.items()):
            samples = metrics.get('sample_count', 0)
            attacks = metrics.get('attack_count', 0)
            accuracy = metrics.get('accuracy', 0)
            tpr = metrics.get('true_positive_rate', metrics.get('tpr', 0))
            fpr = metrics.get('false_positive_rate', metrics.get('fpr', 0))
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1 = metrics.get('f1_score', 0)
            
            report += f"{attack_type:<15} {samples:<10} {attacks:<10} {accuracy:<12.4f} {tpr:<10.4f} {fpr:<10.4f} {precision:<12.4f} {recall:<10.4f} {f1:<10.4f}\n"
        
        report += "\n" + "="*80 + "\n"
        report += "\nDetailed Metrics by Attack Type:\n"
        report += "="*80 + "\n\n"
        
        for attack_type, metrics in sorted(attack_type_results.items()):
            report += f"\n{attack_type.upper()}:\n"
            report += f"  Samples:        {metrics.get('sample_count', 0)}\n"
            report += f"  Attacks:        {metrics.get('attack_count', 0)}\n"
            report += f"  Normal:         {metrics.get('normal_count', 0)}\n"
            report += f"  Accuracy:       {metrics.get('accuracy', 0):.4f}\n"
            report += f"  Precision:      {metrics.get('precision', 0):.4f}\n"
            report += f"  Recall (TPR):   {metrics.get('recall', metrics.get('true_positive_rate', 0)):.4f}\n"
            report += f"  F1-Score:      {metrics.get('f1_score', 0):.4f}\n"
            report += f"  TPR:            {metrics.get('true_positive_rate', metrics.get('tpr', 0)):.4f}\n"
            report += f"  FPR:            {metrics.get('false_positive_rate', metrics.get('fpr', 0)):.4f}\n"
            
            if 'roc_auc' in metrics:
                report += f"  ROC AUC:        {metrics['roc_auc']:.4f}\n"
            
            report += f"  TP:             {metrics.get('true_positive', 0)}\n"
            report += f"  TN:             {metrics.get('true_negative', 0)}\n"
            report += f"  FP:             {metrics.get('false_positive', 0)}\n"
            report += f"  FN:             {metrics.get('false_negative', 0)}\n"
        
        report += "\n" + "="*80 + "\n"
        
        if save_name:
            save_path = self.save_dir / save_name
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Attack type report saved to {save_path}")
        
        return report
    
    def plot_attack_type_comparison(self, attack_type_results_dict: Dict[str, Dict[str, Dict[str, float]]],
                                   save_name: Optional[str] = None):
        """
        Create comprehensive visualizations comparing models across attack types.
        
        Args:
            attack_type_results_dict: Dictionary mapping model names to their attack_type_results
                                     e.g., {'CNN': {...}, 'LSTM': {...}, 'Fusion': {...}}
            save_name: Optional filename to save visualization
        """
        import re
        
        # Parse attack type results from text reports if needed
        # If results are already dictionaries, use them directly
        models_data = {}
        
        for model_name, results in attack_type_results_dict.items():
            if isinstance(results, str):
                # Parse from text file
                results = self._parse_attack_type_report(results)
            models_data[model_name] = results
        
        # Get all unique attack types across all models
        all_attack_types = set()
        for model_results in models_data.values():
            all_attack_types.update(model_results.keys())
        all_attack_types = sorted([at for at in all_attack_types if at.lower() != 'normal'])
        
        # Create comprehensive figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Color scheme
        model_colors = {
            'CNN': '#FF6B6B',
            'LSTM': '#4ECDC4',
            'Fusion': '#45B7D1',
            'Voltage': '#FFA07A'
        }
        
        # Plot 1: TPR by Attack Type (Bar Chart)
        ax1 = fig.add_subplot(gs[0, 0])
        x = np.arange(len(all_attack_types))
        width = 0.25
        offset = 0
        
        for model_name in sorted(models_data.keys()):
            tprs = []
            for attack_type in all_attack_types:
                if attack_type in models_data[model_name]:
                    tpr = models_data[model_name][attack_type].get('true_positive_rate', 
                                                                    models_data[model_name][attack_type].get('tpr', 0))
                else:
                    tpr = 0
                tprs.append(tpr)
            
            color = model_colors.get(model_name, '#95A5A6')
            ax1.bar(x + offset * width, tprs, width, label=model_name, color=color, alpha=0.8)
            offset += 1
        
        ax1.set_xlabel('Attack Type', fontsize=11, fontweight='bold')
        ax1.set_ylabel('True Positive Rate (TPR)', fontsize=11, fontweight='bold')
        ax1.set_title('Attack Detection Rate (TPR) by Attack Type', fontsize=12, fontweight='bold')
        ax1.set_xticks(x + width * (len(models_data) - 1) / 2)
        ax1.set_xticklabels(all_attack_types, rotation=45, ha='right')
        ax1.set_ylim([0, 1.1])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: FPR by Attack Type (Bar Chart)
        ax2 = fig.add_subplot(gs[0, 1])
        offset = 0
        
        for model_name in sorted(models_data.keys()):
            fprs = []
            for attack_type in all_attack_types:
                if attack_type in models_data[model_name]:
                    fpr = models_data[model_name][attack_type].get('false_positive_rate',
                                                                    models_data[model_name][attack_type].get('fpr', 0))
                else:
                    fpr = 0
                fprs.append(fpr)
            
            color = model_colors.get(model_name, '#95A5A6')
            ax2.bar(x + offset * width, fprs, width, label=model_name, color=color, alpha=0.8)
            offset += 1
        
        ax2.set_xlabel('Attack Type', fontsize=11, fontweight='bold')
        ax2.set_ylabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
        ax2.set_title('False Positive Rate (FPR) by Attack Type', fontsize=12, fontweight='bold')
        ax2.set_xticks(x + width * (len(models_data) - 1) / 2)
        ax2.set_xticklabels(all_attack_types, rotation=45, ha='right')
        ax2.set_ylim([0, max(0.1, max([max([models_data[m][at].get('false_positive_rate', 
                                                                      models_data[m][at].get('fpr', 0)) 
                                             for at in all_attack_types if at in models_data[m]]) 
                                        for m in models_data.keys()]) * 1.2)])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Accuracy by Attack Type
        ax3 = fig.add_subplot(gs[0, 2])
        offset = 0
        
        for model_name in sorted(models_data.keys()):
            accuracies = []
            for attack_type in all_attack_types:
                if attack_type in models_data[model_name]:
                    acc = models_data[model_name][attack_type].get('accuracy', 0)
                else:
                    acc = 0
                accuracies.append(acc)
            
            color = model_colors.get(model_name, '#95A5A6')
            ax3.bar(x + offset * width, accuracies, width, label=model_name, color=color, alpha=0.8)
            offset += 1
        
        ax3.set_xlabel('Attack Type', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax3.set_title('Accuracy by Attack Type', fontsize=12, fontweight='bold')
        ax3.set_xticks(x + width * (len(models_data) - 1) / 2)
        ax3.set_xticklabels(all_attack_types, rotation=45, ha='right')
        ax3.set_ylim([0, 1.1])
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Heatmap - TPR across models and attack types
        ax4 = fig.add_subplot(gs[1, 0])
        heatmap_data_tpr = []
        model_names_sorted = sorted(models_data.keys())
        
        for model_name in model_names_sorted:
            row = []
            for attack_type in all_attack_types:
                if attack_type in models_data[model_name]:
                    tpr = models_data[model_name][attack_type].get('true_positive_rate',
                                                                   models_data[model_name][attack_type].get('tpr', 0))
                else:
                    tpr = 0
                row.append(tpr)
            heatmap_data_tpr.append(row)
        
        sns.heatmap(heatmap_data_tpr, annot=True, fmt='.3f', cmap='RdYlGn',
                    xticklabels=all_attack_types, yticklabels=model_names_sorted,
                    cbar_kws={'label': 'TPR'}, ax=ax4, vmin=0, vmax=1.0)
        ax4.set_title('True Positive Rate Heatmap', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Attack Type', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Model', fontsize=11, fontweight='bold')
        
        # Plot 5: Heatmap - FPR across models and attack types
        ax5 = fig.add_subplot(gs[1, 1])
        heatmap_data_fpr = []
        
        for model_name in model_names_sorted:
            row = []
            for attack_type in all_attack_types:
                if attack_type in models_data[model_name]:
                    fpr = models_data[model_name][attack_type].get('false_positive_rate',
                                                                   models_data[model_name][attack_type].get('fpr', 0))
                else:
                    fpr = 0
                row.append(fpr)
            heatmap_data_fpr.append(row)
        
        max_fpr = max([max(row) for row in heatmap_data_fpr]) if heatmap_data_fpr else 0.1
        sns.heatmap(heatmap_data_fpr, annot=True, fmt='.4f', cmap='YlOrRd',
                    xticklabels=all_attack_types, yticklabels=model_names_sorted,
                    cbar_kws={'label': 'FPR'}, ax=ax5, vmin=0, vmax=max(0.1, max_fpr))
        ax5.set_title('False Positive Rate Heatmap', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Attack Type', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Model', fontsize=11, fontweight='bold')
        
        # Plot 6: Heatmap - Accuracy across models and attack types
        ax6 = fig.add_subplot(gs[1, 2])
        heatmap_data_acc = []
        
        for model_name in model_names_sorted:
            row = []
            for attack_type in all_attack_types:
                if attack_type in models_data[model_name]:
                    acc = models_data[model_name][attack_type].get('accuracy', 0)
                else:
                    acc = 0
                row.append(acc)
            heatmap_data_acc.append(row)
        
        sns.heatmap(heatmap_data_acc, annot=True, fmt='.3f', cmap='RdYlGn',
                    xticklabels=all_attack_types, yticklabels=model_names_sorted,
                    cbar_kws={'label': 'Accuracy'}, ax=ax6, vmin=0, vmax=1.0)
        ax6.set_title('Accuracy Heatmap', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Attack Type', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Model', fontsize=11, fontweight='bold')
        
        # Plot 7: TPR vs FPR Scatter (one point per model-attack combination)
        ax7 = fig.add_subplot(gs[2, 0])
        for model_name in model_names_sorted:
            tprs = []
            fprs = []
            labels = []
            for attack_type in all_attack_types:
                if attack_type in models_data[model_name]:
                    tpr = models_data[model_name][attack_type].get('true_positive_rate',
                                                                    models_data[model_name][attack_type].get('tpr', 0))
                    fpr = models_data[model_name][attack_type].get('false_positive_rate',
                                                                    models_data[model_name][attack_type].get('fpr', 0))
                    tprs.append(tpr)
                    fprs.append(fpr)
                    labels.append(attack_type)
            
            color = model_colors.get(model_name, '#95A5A6')
            scatter = ax7.scatter(fprs, tprs, s=150, alpha=0.6, color=color, 
                                 label=model_name, edgecolors='black', linewidth=1.5)
            # Add labels
            for i, label in enumerate(labels):
                ax7.annotate(label, (fprs[i], tprs[i]), fontsize=8, 
                           xytext=(5, 5), textcoords='offset points')
        
        ax7.set_xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
        ax7.set_ylabel('True Positive Rate (TPR)', fontsize=11, fontweight='bold')
        ax7.set_title('TPR vs FPR Trade-off by Attack Type', fontsize=12, fontweight='bold')
        ax7.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        ax7.set_xlim([-0.01, max(0.1, max([max([models_data[m][at].get('false_positive_rate',
                                                                         models_data[m][at].get('fpr', 0)) 
                                                 for at in all_attack_types if at in models_data[m]]) 
                                            for m in models_data.keys()]) * 1.2)])
        ax7.set_ylim([0.9, 1.05])
        ax7.legend()
        ax7.grid(alpha=0.3)
        
        # Plot 8: F1-Score by Attack Type
        ax8 = fig.add_subplot(gs[2, 1])
        offset = 0
        
        for model_name in sorted(models_data.keys()):
            f1_scores = []
            for attack_type in all_attack_types:
                if attack_type in models_data[model_name]:
                    f1 = models_data[model_name][attack_type].get('f1_score', 0)
                else:
                    f1 = 0
                f1_scores.append(f1)
            
            color = model_colors.get(model_name, '#95A5A6')
            ax8.bar(x + offset * width, f1_scores, width, label=model_name, color=color, alpha=0.8)
            offset += 1
        
        ax8.set_xlabel('Attack Type', fontsize=11, fontweight='bold')
        ax8.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
        ax8.set_title('F1-Score by Attack Type', fontsize=12, fontweight='bold')
        ax8.set_xticks(x + width * (len(models_data) - 1) / 2)
        ax8.set_xticklabels(all_attack_types, rotation=45, ha='right')
        ax8.set_ylim([0, 1.1])
        ax8.legend()
        ax8.grid(axis='y', alpha=0.3)
        
        # Plot 9: Sample Counts by Attack Type
        ax9 = fig.add_subplot(gs[2, 2])
        # Get sample counts from first model (they should be similar)
        first_model = model_names_sorted[0]
        sample_counts = []
        for attack_type in all_attack_types:
            if attack_type in models_data[first_model]:
                count = models_data[first_model][attack_type].get('sample_count', 0)
            else:
                count = 0
            sample_counts.append(count)
        
        bars = ax9.bar(all_attack_types, sample_counts, color='#95A5A6', alpha=0.7)
        ax9.set_xlabel('Attack Type', fontsize=11, fontweight='bold')
        ax9.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
        ax9.set_title('Test Set Distribution by Attack Type', fontsize=12, fontweight='bold')
        ax9.tick_params(axis='x', rotation=45)
        ax9.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        fig.suptitle('Model Performance Comparison by Attack Type', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        try:
            plt.tight_layout()
        except:
            # Some subplots may not be compatible with tight_layout
            pass
        
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attack type comparison visualization saved to {save_path}")
        
        plt.close()
    
    def _parse_attack_type_report(self, report_text: str) -> Dict[str, Dict[str, float]]:
        """
        Parse attack type report text into structured dictionary.
        
        Args:
            report_text: Text content of attack type report
            
        Returns:
            Dictionary mapping attack type to metrics
        """
        results = {}
        lines = report_text.split('\n')
        
        # Find the table section
        in_table = False
        for line in lines:
            line = line.strip()
            if 'Attack Type' in line and 'Samples' in line:
                in_table = True
                continue
            
            if in_table and line.startswith('-'):
                continue
            
            if in_table and line and not line.startswith('='):
                # Parse table row
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        # Attack type might be multiple words (e.g., "Voltage + CNN")
                        # Find where numbers start
                        attack_type_parts = []
                        num_start_idx = 0
                        for i, part in enumerate(parts):
                            try:
                                int(part)  # Try to convert to int
                                num_start_idx = i
                                break
                            except ValueError:
                                attack_type_parts.append(part)
                        
                        attack_type = ' '.join(attack_type_parts).lower()
                        if not attack_type:
                            continue
                        
                        # Parse numbers starting from num_start_idx
                        samples = int(parts[num_start_idx])
                        attacks = int(parts[num_start_idx + 1])
                        accuracy = float(parts[num_start_idx + 2])
                        tpr = float(parts[num_start_idx + 3])
                        fpr = float(parts[num_start_idx + 4])
                        precision = float(parts[num_start_idx + 5])
                        recall = float(parts[num_start_idx + 6])
                        f1_score = float(parts[num_start_idx + 7]) if len(parts) > num_start_idx + 7 else 0.0
                        
                        results[attack_type] = {
                            'sample_count': samples,
                            'attack_count': attacks,
                            'accuracy': accuracy,
                            'true_positive_rate': tpr,
                            'tpr': tpr,
                            'false_positive_rate': fpr,
                            'fpr': fpr,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1_score
                        }
                    except (ValueError, IndexError):
                        continue
            
            if in_table and line.startswith('='):
                break
        
        return results
    
    def plot_model_attack_type_performance(self, attack_type_results: Dict[str, Dict[str, float]],
                                          model_name: str,
                                          save_name: Optional[str] = None):
        """
        Create visualization showing a single model's performance across attack types.
        
        Args:
            attack_type_results: Results from evaluate_by_attack_type
            model_name: Name of the model
            save_name: Optional filename to save visualization
        """
        # Filter out 'normal' attack type for visualization
        attack_types = sorted([at for at in attack_type_results.keys() if at.lower() != 'normal'])
        
        if not attack_types:
            logger.warning("No attack types found for visualization")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{model_name} - Performance by Attack Type', 
                    fontsize=16, fontweight='bold')
        
        # Extract data
        tprs = [attack_type_results[at].get('true_positive_rate', 
                                            attack_type_results[at].get('tpr', 0)) 
                for at in attack_types]
        fprs = [attack_type_results[at].get('false_positive_rate',
                                             attack_type_results[at].get('fpr', 0)) 
                for at in attack_types]
        accuracies = [attack_type_results[at].get('accuracy', 0) for at in attack_types]
        f1_scores = [attack_type_results[at].get('f1_score', 0) for at in attack_types]
        sample_counts = [attack_type_results[at].get('sample_count', 0) for at in attack_types]
        
        # Plot 1: TPR by Attack Type
        ax1 = axes[0, 0]
        bars1 = ax1.bar(attack_types, tprs, color='#4ECDC4', alpha=0.8, edgecolor='black')
        ax1.set_ylabel('True Positive Rate (TPR)', fontsize=11, fontweight='bold')
        ax1.set_title('Attack Detection Rate', fontsize=12, fontweight='bold')
        ax1.set_ylim([0, 1.1])
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        for bar, tpr in zip(bars1, tprs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{tpr:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: FPR by Attack Type
        ax2 = axes[0, 1]
        bars2 = ax2.bar(attack_types, fprs, color='#FF6B6B', alpha=0.8, edgecolor='black')
        ax2.set_ylabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
        ax2.set_title('False Alarm Rate', fontsize=12, fontweight='bold')
        max_fpr = max(fprs) if fprs else 0.1
        ax2.set_ylim([0, max(0.1, max_fpr * 1.2)])
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        for bar, fpr in zip(bars2, fprs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                   f'{fpr:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: Accuracy and F1-Score
        ax3 = axes[1, 0]
        x = np.arange(len(attack_types))
        width = 0.35
        bars3a = ax3.bar(x - width/2, accuracies, width, label='Accuracy', 
                        color='#45B7D1', alpha=0.8, edgecolor='black')
        bars3b = ax3.bar(x + width/2, f1_scores, width, label='F1-Score', 
                        color='#FFA07A', alpha=0.8, edgecolor='black')
        ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax3.set_title('Accuracy and F1-Score', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(attack_types, rotation=45, ha='right')
        ax3.set_ylim([0, 1.1])
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Sample Distribution
        ax4 = axes[1, 1]
        bars4 = ax4.bar(attack_types, sample_counts, color='#95A5A6', alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
        ax4.set_title('Test Set Distribution', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', alpha=0.3)
        for bar, count in zip(bars4, sample_counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model attack type performance visualization saved to {save_path}")
        
        plt.close()


def main():
    """Test evaluation metrics"""
    logger.info("Testing evaluation metrics...")
    
    # Generate sample predictions
    np.random.seed(42)
    n_samples = 1000
    
    y_true = np.random.randint(0, 2, n_samples)
    y_score = np.random.rand(n_samples)
    y_pred = (y_score > 0.5).astype(int)
    
    # Create evaluator
    evaluator = IDSEvaluator(save_dir="results/test")
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred, y_score)
    logger.info("\nMetrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")
    
    # Generate visualizations
    evaluator.plot_confusion_matrix(y_true, y_pred, save_name="test_confusion_matrix.png")
    evaluator.plot_roc_curve(y_true, y_score, save_name="test_roc_curve.png")
    evaluator.plot_precision_recall_curve(y_true, y_score, save_name="test_pr_curve.png")
    
    # Generate report
    report = evaluator.generate_report("Test Model", metrics, save_name="test_report.txt")
    print(report)
    
    logger.info("\nEvaluation metrics testing complete!")


if __name__ == "__main__":
    main()
