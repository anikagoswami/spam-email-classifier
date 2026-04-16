"""
Model comparison visualization and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


class ModelComparator:
    """Creates visual comparisons between different models."""
    
    def __init__(self, visualization_dir='visualizations'):
        """
        Initialize the model comparator.
        
        Args:
            visualization_dir: Directory to save comparison plots
        """
        self.visualization_dir = visualization_dir
        os.makedirs(visualization_dir, exist_ok=True)
    
    def plot_performance_comparison(self, results, model_names=None):
        """
        Create a bar chart comparing model performance metrics.
        
        Args:
            results: Dictionary containing model results
            model_names: List of model names to include (optional)
        """
        if model_names is None:
            model_names = list(results.keys())
        
        # Extract metrics for each model
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        colors = ['#FDB01C', '#71E8F0', '#990000', '#13545A']
        
        for i, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
            values = []
            for model_name in model_names:
                if model_name in results:
                    values.append(results[model_name][metric])
            
            # Create bar chart
            bars = axes[i].bar(model_names, values, color=colors[:len(model_names)])
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                           f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            axes[i].set_title(f'{metric_label} Comparison', fontsize=14, fontweight='bold', color='#13545A')
            axes[i].set_ylabel('Score', fontsize=12)
            axes[i].set_ylim(0, 1.1)  # Set y-axis limit to accommodate labels
            axes[i].grid(True, axis='y', alpha=0.3)
            
            # Color the bars by model
            for j, bar in enumerate(bars):
                bar.set_color(colors[j % len(colors)])
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', color='#13545A')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.visualization_dir, 'model_performance_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance comparison saved to {plot_path}")
    
    def plot_combined_performance_radar(self, results, model_names=None):
        """
        Create a radar chart comparing all metrics for each model.
        
        Args:
            results: Dictionary containing model results
            model_names: List of model names to include (optional)
        """
        if model_names is None:
            model_names = list(results.keys())
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        # Number of metrics
        n_metrics = len(metrics)
        
        # Calculate angle for each metric
        angles = [n / float(n_metrics) * 2 * np.pi for n in range(n_metrics)]
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Draw one axe per variable and add labels
        plt.xticks(angles[:-1], metric_labels, color='#13545A', size=12)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="#13545A", size=10)
        plt.ylim(0, 1)
        
        colors = ['#FDB01C', '#71E8F0', '#990000']
        
        # Plot data for each model
        for i, model_name in enumerate(model_names):
            if model_name in results:
                values = [results[model_name][metric] for metric in metrics]
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=colors[i % len(colors)])
                ax.fill(angles, values, colors[i % len(colors)], alpha=0.1)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('Model Performance Radar Chart', size=16, color='#13545A', y=1.1)
        
        # Save plot
        plot_path = os.path.join(self.visualization_dir, 'model_radar_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Radar comparison saved to {plot_path}")
    
    def plot_training_history_comparison(self, histories):
        """
        Create a comparison plot of training histories for all models.
        
        Args:
            histories: Dictionary of training histories for each model
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = ['#FDB01C', '#71E8F0', '#990000']
        
        # Plot accuracy comparison
        for i, (model_name, history) in enumerate(histories.items()):
            axes[0].plot(history.history['accuracy'], label=f'{model_name} Train', 
                        color=colors[i % len(colors)], linestyle='-')
            axes[0].plot(history.history['val_accuracy'], label=f'{model_name} Val', 
                        color=colors[i % len(colors)], linestyle='--')
        
        axes[0].set_title('Training & Validation Accuracy Comparison', fontsize=14, fontweight='bold', color='#13545A')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss comparison
        for i, (model_name, history) in enumerate(histories.items()):
            axes[1].plot(history.history['loss'], label=f'{model_name} Train', 
                        color=colors[i % len(colors)], linestyle='-')
            axes[1].plot(history.history['val_loss'], label=f'{model_name} Val', 
                        color=colors[i % len(colors)], linestyle='--')
        
        axes[1].set_title('Training & Validation Loss Comparison', fontsize=14, fontweight='bold', color='#13545A')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Training History Comparison', fontsize=16, fontweight='bold', color='#13545A')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.visualization_dir, 'training_history_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history comparison saved to {plot_path}")
    
    def create_comprehensive_comparison(self, results, histories=None):
        """
        Create all comparison visualizations.
        
        Args:
            results: Dictionary containing model results
            histories: Dictionary of training histories (optional)
        """
        print("\n" + "="*60)
        print("CREATING COMPREHENSIVE MODEL COMPARISON")
        print("="*60)
        
        # Performance bar chart comparison
        self.plot_performance_comparison(results)
        
        # Radar chart comparison
        self.plot_combined_performance_radar(results)
        
        # Training history comparison (if histories provided)
        if histories:
            self.plot_training_history_comparison(histories)
    
    def generate_comparison_report(self, results, output_file='model_comparison_report.txt'):
        """
        Generate a text report comparing all models.
        
        Args:
            results: Dictionary containing model results
            output_file: Path to save the report
        """
        report_path = os.path.join(self.visualization_dir, output_file)
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("SPAM EMAIL CLASSIFICATION - MODEL COMPARISON REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Individual model results
            f.write("INDIVIDUAL MODEL PERFORMANCE\n")
            f.write("-"*40 + "\n\n")
            
            for model_name, metrics in results.items():
                f.write(f"{model_name} Results:\n")
                f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {metrics['recall']:.4f}\n")
                f.write(f"  F1 Score:  {metrics['f1_score']:.4f}\n\n")
            
            # Best model
            best_model = max(results.keys(), key=lambda k: results[k]['f1_score'])
            f.write(f"\nBEST MODEL (by F1 Score): {best_model}\n")
            f.write(f"F1 Score: {results[best_model]['f1_score']:.4f}\n\n")
            
            # Detailed analysis
            f.write("\nDETAILED ANALYSIS\n")
            f.write("-"*40 + "\n\n")
            
            # Accuracy ranking
            accuracy_ranking = sorted(results.keys(), key=lambda k: results[k]['accuracy'], reverse=True)
            f.write("Accuracy Ranking:\n")
            for i, model in enumerate(accuracy_ranking, 1):
                f.write(f"  {i}. {model}: {results[model]['accuracy']:.4f}\n")
            
            # Precision ranking
            precision_ranking = sorted(results.keys(), key=lambda k: results[k]['precision'], reverse=True)
            f.write("\nPrecision Ranking:\n")
            for i, model in enumerate(precision_ranking, 1):
                f.write(f"  {i}. {model}: {results[model]['precision']:.4f}\n")
            
            # Recall ranking
            recall_ranking = sorted(results.keys(), key=lambda k: results[k]['recall'], reverse=True)
            f.write("\nRecall Ranking:\n")
            for i, model in enumerate(recall_ranking, 1):
                f.write(f"  {i}. {model}: {results[model]['recall']:.4f}\n")
            
            # F1 ranking
            f1_ranking = sorted(results.keys(), key=lambda k: results[k]['f1_score'], reverse=True)
            f.write("\nF1 Score Ranking:\n")
            for i, model in enumerate(f1_ranking, 1):
                f.write(f"  {i}. {model}: {results[model]['f1_score']:.4f}\n")
            
            # Recommendations
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-"*40 + "\n\n")
            
            if best_model == 'LSTM':
                f.write("The LSTM model performed best, likely due to its ability to capture\n")
                f.write("long-term dependencies in email text sequences.\n")
            elif best_model == 'GRU':
                f.write("The GRU model performed best, offering a good balance between\n")
                f.write("performance and computational efficiency.\n")
            elif best_model == 'RNN':
                f.write("The Simple RNN model performed best, which is unusual but may indicate\n")
                f.write("that the dataset has simpler patterns that don't require complex architectures.\n")
            
            f.write("\nFor production deployment, consider using the best performing model\n")
            f.write("while also factoring in inference speed and model size.\n")
        
        print(f"Comparison report saved to {report_path}")


def main():
    """Example usage of model comparator."""
    # Sample results for demonstration
    results = {
        'RNN': {
            'accuracy': 0.85,
            'precision': 0.83,
            'recall': 0.80,
            'f1_score': 0.815,
            'predictions': np.random.randint(0, 2, 100),
            'probabilities': np.random.random(100)
        },
        'LSTM': {
            'accuracy': 0.92,
            'precision': 0.91,
            'recall': 0.89,
            'f1_score': 0.90,
            'predictions': np.random.randint(0, 2, 100),
            'probabilities': np.random.random(100)
        },
        'GRU': {
            'accuracy': 0.90,
            'precision': 0.88,
            'recall': 0.87,
            'f1_score': 0.875,
            'predictions': np.random.randint(0, 2, 100),
            'probabilities': np.random.random(100)
        }
    }
    
    # Initialize comparator
    comparator = ModelComparator()
    
    # Create comprehensive comparison
    comparator.create_comprehensive_comparison(results)
    
    # Generate comparison report
    comparator.generate_comparison_report(results)
    
    print("\nModel comparison completed!")


if __name__ == "__main__":
    main()