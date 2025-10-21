import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_training_curves(training_history, save_path='plots'):
    """
    Plot comprehensive training curves from training history.
    
    Args:
        training_history (dict): Dictionary containing training metrics
        save_path (str): Directory to save plots
    """
    # Create plots directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    epochs = training_history['epochs']
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Loss curves
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, training_history['train_losses'], label='Training Loss', linewidth=2)
    plt.plot(epochs, training_history['val_losses'], label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Accuracy curves
    plt.subplot(2, 2, 2)
    plt.plot(epochs, training_history['train_accuracies'], label='Training Accuracy', linewidth=2)
    plt.plot(epochs, training_history['val_accuracies'], label='Validation Accuracy', linewidth=2)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Precision curves (Class 1)
    plt.subplot(2, 2, 3)
    plt.plot(epochs, training_history['train_precisions'], label='Training Precision (Class 1)', linewidth=2)
    plt.plot(epochs, training_history['val_precisions'], label='Validation Precision (Class 1)', linewidth=2)
    plt.title('Precision for Class 1 (Positive)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Recall curves (Class 1)
    plt.subplot(2, 2, 4)
    plt.plot(epochs, training_history['train_recalls'], label='Training Recall (Class 1)', linewidth=2)
    plt.plot(epochs, training_history['val_recalls'], label='Validation Recall (Class 1)', linewidth=2)
    plt.title('Recall for Class 1 (Positive)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Precision vs Recall scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(training_history['val_recalls'], training_history['val_precisions'], 
                c=epochs, cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    plt.colorbar(label='Epoch')
    plt.xlabel('Validation Recall (Class 1)', fontsize=12)
    plt.ylabel('Validation Precision (Class 1)', fontsize=12)
    plt.title('Precision vs Recall Trade-off (Validation)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add epoch numbers as annotations
    for i, epoch in enumerate(epochs):
        if i % 2 == 0:  # Show every other epoch to avoid clutter
            plt.annotate(f'E{epoch}', (training_history['val_recalls'][i], 
                                     training_history['val_precisions'][i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.savefig(f'{save_path}/precision_vs_recall.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Confusion matrices over time
    if 'confusion_matrices' in training_history and training_history['confusion_matrices']:
        plot_confusion_matrices(training_history['confusion_matrices'], epochs, save_path)
    
    print(f"Plots saved to '{save_path}' directory")

def plot_confusion_matrices(confusion_matrices, epochs, save_path='plots'):
    """
    Plot confusion matrices for different epochs.
    
    Args:
        confusion_matrices (list): List of confusion matrices
        epochs (list): List of epoch numbers
        save_path (str): Directory to save plots
    """
    # Select key epochs to plot (beginning, middle, end)
    n_matrices = len(confusion_matrices)
    if n_matrices == 0:
        return
    
    # Select epochs to plot
    if n_matrices <= 3:
        selected_indices = list(range(n_matrices))
    else:
        selected_indices = [0, n_matrices//2, n_matrices-1]
    
    fig, axes = plt.subplots(1, len(selected_indices), figsize=(5*len(selected_indices), 4))
    if len(selected_indices) == 1:
        axes = [axes]
    
    for i, idx in enumerate(selected_indices):
        cm = confusion_matrices[idx]
        epoch = epochs[idx]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Class 0', 'Class 1'],
                   yticklabels=['Class 0', 'Class 1'],
                   ax=axes[i])
        axes[i].set_title(f'Confusion Matrix - Epoch {epoch}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_learning_rate_schedule(optimizer_history, save_path='plots'):
    """
    Plot learning rate schedule over training.
    
    Args:
        optimizer_history (list): List of learning rates
        save_path (str): Directory to save plots
    """
    plt.figure(figsize=(10, 6))
    plt.plot(optimizer_history, linewidth=2)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}/learning_rate_schedule.png', dpi=300, bbox_inches='tight')
    plt.show()

def load_and_plot_training_history(file_path='training_history.pt', save_path='plots'):
    """
    Load training history from file and create plots.
    
    Args:
        file_path (str): Path to training history file
        save_path (str): Directory to save plots
    """
    try:
        training_history = torch.load(file_path, map_location='cpu', weights_only=False)['training_history']
        print(f"Loaded training history from {file_path}")
        print(f"Training completed for {len(training_history['epochs'])} epochs")
        
        # Print final metrics
        final_epoch = training_history['epochs'][-1]
        print(f"\nFinal Metrics (Epoch {final_epoch}):")
        print(f"  Training Loss: {training_history['train_losses'][-1]:.4f}")
        print(f"  Validation Loss: {training_history['val_losses'][-1]:.4f}")
        print(f"  Training Accuracy: {training_history['train_accuracies'][-1]:.4f}")
        print(f"  Validation Accuracy: {training_history['val_accuracies'][-1]:.4f}")
        print(f"  Training Precision (Class 1): {training_history['train_precisions'][-1]:.4f}")
        print(f"  Validation Precision (Class 1): {training_history['val_precisions'][-1]:.4f}")
        print(f"  Training Recall (Class 1): {training_history['train_recalls'][-1]:.4f}")
        print(f"  Validation Recall (Class 1): {training_history['val_recalls'][-1]:.4f}")
        
        # Create plots
        plot_training_curves(training_history, save_path)
        
    except FileNotFoundError:
        print(f"Training history file '{file_path}' not found.")
    except Exception as e:
        print(f"Error loading training history: {e}")

if __name__ == "__main__":
    # Example usage
    load_and_plot_training_history()
