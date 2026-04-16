"""
Detailed metrics computation for model evaluation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_curve, auc, precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os


class MetricsEvaluator:
    """Comprehensive metrics evaluation for spam classification models."""
    
    def __init__(self, visualization_dir='visualizations'):
        """
        Initialize the metrics evaluator.
        
        Args:
            visualization_dir: Directory to save metric plots
        """
        self.visualization_dir = visualization_dir
        os.makedirs(visualization_dir, exist_ok=True)
    
    def compute_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Compute comprehensive metrics for a model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Add AUC metrics if probabilities are provided
        if y_pred_proba is not None:
            try:
                metrics['auc'] = auc(
                    *roc_curve(y_true, y_pred_proba)[:2]
                )
                metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
            except:
                metrics['auc'] = 0
                metrics['average_precision'] = 0
        
        return metrics
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name, filename=None):
        """
        Plot ROC curve for a model.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            filename: Custom filename (optional)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        
        plt.plot(fpr, tpr, color='#FDB01C', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='#13545A', lw=1, linestyle='--', alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{model_name} - ROC Curve', fontsize=14, fontweight='bold', color='#13545A')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if filename is None:
            filename = f'{model_name.lower()}_roc_curve.png'
        
        plot_path = os.path.join(self.visualization_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curve saved to {plot_path}")
        
        return roc_auc
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, model_name, filename=None):
        """
        Plot Precision-Recall curve for a model.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            filename: Custom filename (optional)
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        
        plt.plot(recall, precision, color='#71E8F0', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'{model_name} - Precision-Recall Curve', fontsize=14, fontweight='bold', color='#13545A')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if filename is None:
            filename = f'{model_name.lower()}_pr_curve.png'
        
        plot_path = os.path.join(self.visualization_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"PR curve saved to {plot_path}")
        
        return avg_precision
    
    def plot_classification_report(self, y_true, y_pred, model_name, filename=None):
        """
        Plot classification report heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            filename: Custom filename (optional)
        """
        # Get classification report as dictionary
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Extract metrics for each class
        classes = ['ham', 'spam']
        metrics = ['precision', 'recall', 'f1-score']
        
        # Create data matrix
        data = []
        for cls in classes:
            row = []
            for metric in metrics:
                row.append(report[cls][metric])
            data.append(row)
        
        data = np.array(data)
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(data, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=metrics, yticklabels=classes,
                   cbar_kws={'label': 'Score'})
        
        plt.title(f'{model_name} - Classification Report', fontsize=14, fontweight='bold', color='#13545A')
        
        if filename is None:
            filename = f'{model_name.lower()}_classification_report.png'
        
        plot_path = os.path.join(self.visualization_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Classification report saved to {plot_path}")
    
    def generate_all_metrics(self, results, y_true):
        """
        Generate all metrics and plots for all models.
        
        Args:
            results: Dictionary containing model results
            y_true: True labels for test set
        """
        print("\n" + "="*60)
        print("GENERATING DETAILED METRICS")
        print("="*60)
        
        all_metrics = {}
        
        for model_name, result in results.items():
            print(f"\nEvaluating {model_name}...")
            
            # Compute metrics
            metrics = self.compute_metrics(
                y_true, 
                result['predictions'], 
                result['probabilities']
            )
            all_metrics[model_name] = metrics
            
            # Print metrics
            print(f"Metrics for {model_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
            
            # Generate plots
            if result['probabilities'] is not None:
                self.plot_roc_curve(y_true, result['probabilities'], model_name)
                self.plot_precision_recall_curve(y_true, result['probabilities'], model_name)
            
            self.plot_classification_report(y_true, result['predictions'], model_name)
        
        return all_metrics
    
    def compare_models(self, all_metrics):
        """
        Create a comparison table of all models.
        
        Args:
            all_metrics: Dictionary of metrics for all models
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        # Create comparison table
        print("\nModel Comparison Table:")
        print("-" * 80)
        print(f"{'Model':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'AUC':<10}")
        print("-" * 80)
        
        for model_name, metrics in all_metrics.items():
            print(f"{model_name:<10} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} {metrics.get('auc', 'N/A'):<10}")
        
        print("-" * 80)
        
        # Find best model
        best_model = max(all_metrics.keys(), key=lambda k: all_metrics[k]['f1_score'])
        print(f"\nBest Model (by F1 Score): {best_model}")
        print(f"F1 Score: {all_metrics[best_model]['f1_score']:.4f}")


def main():
    """Example usage of metrics evaluator."""
    # Sample data for demonstration
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10)
    
    # Sample predictions for different models
    results = {
        'RNN': {
            'predictions': np.array([0, 1, 0, 1, 1, 1, 0, 0, 0, 1] * 10),
            'probabilities': np.random.random(100)
        },
        'LSTM': {
            'predictions': np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 1] * 10),
            'probabilities': np.random.random(100)
        },
        'GRU': {
            'predictions': np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0] * 10),
            'probabilities': np.random.random(100)
        }
    }
    
    # Initialize evaluator
    evaluator = MetricsEvaluator()
    
    # Generate all metrics
    all_metrics = evaluator.generate_all_metrics(results, y_true)
    
    # Compare models
    evaluator.compare_models(all_metrics)
    
    print("\nMetrics evaluation completed!")


if __name__ == "__main__":
    main()