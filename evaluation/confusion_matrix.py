"""
Confusion matrix generation and visualization for model evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


class ConfusionMatrixGenerator:
    """Generates and visualizes confusion matrices for model evaluation."""
    
    def __init__(self, visualization_dir='visualizations'):
        """
        Initialize the confusion matrix generator.
        
        Args:
            visualization_dir: Directory to save confusion matrix plots
        """
        self.visualization_dir = visualization_dir
        os.makedirs(visualization_dir, exist_ok=True)
    
    def create_confusion_matrix(self, y_true, y_pred, model_name):
        """
        Create confusion matrix for a model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            Confusion matrix as numpy array
        """
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\n{model_name} Confusion Matrix:")
        print(f"[[TN  FP]")
        print(f" [FN  TP]]")
        print(f"{cm}")
        
        return cm
    
    def plot_confusion_matrix(self, cm, model_name, filename=None):
        """
        Plot and save confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix
            model_name: Name of the model
            filename: Custom filename (optional)
        """
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Ham (0)', 'Spam (1)'],
            yticklabels=['Ham (0)', 'Spam (1)'],
            cbar_kws={'label': 'Count'}
        )
        
        plt.title(f'{model_name} - Confusion Matrix', 
                 fontsize=16, fontweight='bold', color='#13545A')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add detailed labels
        tn, fp, fn, tp = cm.ravel()
        plt.figtext(0.02, 0.02, 
                   f'TN (True Negative): {tn}\nFP (False Positive): {fp}\nFN (False Negative): {fn}\nTP (True Positive): {tp}',
                   fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        # Save plot
        if filename is None:
            filename = f'{model_name.lower()}_confusion_matrix.png'
        
        plot_path = os.path.join(self.visualization_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {plot_path}")
    
    def plot_normalized_confusion_matrix(self, cm, model_name, filename=None):
        """
        Plot and save normalized confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix
            model_name: Name of the model
            filename: Custom filename (optional)
        """
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        
        # Create heatmap with normalized values
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=['Ham (0)', 'Spam (1)'],
            yticklabels=['Ham (0)', 'Spam (1)'],
            cbar_kws={'label': 'Normalized Count'}
        )
        
        plt.title(f'{model_name} - Normalized Confusion Matrix', 
                 fontsize=16, fontweight='bold', color='#13545A')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        if filename is None:
            filename = f'{model_name.lower()}_normalized_confusion_matrix.png'
        
        plot_path = os.path.join(self.visualization_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Normalized confusion matrix saved to {plot_path}")
    
    def generate_all_confusion_matrices(self, results, y_true):
        """
        Generate confusion matrices for all models.
        
        Args:
            results: Dictionary containing model results
            y_true: True labels for test set
        """
        print("\n" + "="*60)
        print("GENERATING CONFUSION MATRICES")
        print("="*60)
        
        for model_name, result in results.items():
            print(f"\nGenerating confusion matrix for {model_name}...")
            
            # Create confusion matrix
            cm = self.create_confusion_matrix(y_true, result['predictions'], model_name)
            
            # Plot regular confusion matrix
            self.plot_confusion_matrix(cm, model_name)
            
            # Plot normalized confusion matrix
            self.plot_normalized_confusion_matrix(cm, model_name)
    
    def create_comparison_confusion_matrix(self, results, y_true, model_names=None):
        """
        Create a comparison plot showing confusion matrices for multiple models.
        
        Args:
            results: Dictionary containing model results
            y_true: True labels for test set
            model_names: List of model names to include (optional)
        """
        if model_names is None:
            model_names = list(results.keys())
        
        n_models = len(model_names)
        
        # Create subplot grid
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        # Handle single model case
        if n_models == 1:
            axes = [axes]
        
        for i, model_name in enumerate(model_names):
            if model_name in results:
                cm = confusion_matrix(y_true, results[model_name]['predictions'])
                
                # Create heatmap
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'],
                    ax=axes[i],
                    cbar=i==n_models-1  # Only show colorbar for last plot
                )
                
                axes[i].set_title(f'{model_name}\nConfusion Matrix', 
                                fontsize=14, fontweight='bold', color='#13545A')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        plt.suptitle('Model Comparison - Confusion Matrices', 
                    fontsize=16, fontweight='bold', color='#13545A')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.visualization_dir, 'comparison_confusion_matrices.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison confusion matrix saved to {plot_path}")


def main():
    """Example usage of confusion matrix generator."""
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
    
    # Initialize generator
    cm_generator = ConfusionMatrixGenerator()
    
    # Generate all confusion matrices
    cm_generator.generate_all_confusion_matrices(results, y_true)
    
    # Create comparison plot
    cm_generator.create_comparison_confusion_matrix(results, y_true)
    
    print("\nConfusion matrix generation completed!")


if __name__ == "__main__":
    main()
