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
            save_path.parent.mkdir(parents=True, exist_ok=True)
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
            save_path.parent.mkdir(parents=True, exist_ok=True)
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
            save_path.parent.mkdir(parents=True, exist_ok=True)
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
            save_path.parent.mkdir(parents=True, exist_ok=True)
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
            save_path.parent.mkdir(parents=True, exist_ok=True)
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
            save_path.parent.mkdir(parents=True, exist_ok=True)
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
            save_path.parent.mkdir(parents=True, exist_ok=True)
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
            save_path.parent.mkdir(parents=True, exist_ok=True)
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
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {save_path}")
        
        return report


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
