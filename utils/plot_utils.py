import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve, roc_auc_score
import os
from collections import defaultdict
from tqdm import tqdm

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
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.plot(epochs, training_history['train_losses'], label='Training Loss', linewidth=2)
    plt.plot(epochs, training_history['val_losses'], label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Accuracy curves
    plt.subplot(2, 3, 2)
    plt.plot(epochs, training_history['train_accuracies'], label='Training Accuracy', linewidth=2)
    plt.plot(epochs, training_history['val_accuracies'], label='Validation Accuracy', linewidth=2)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Precision curves (Class 1)
    plt.subplot(2, 3, 3)
    plt.plot(epochs, training_history['train_precisions'], label='Training Precision (Class 1)', linewidth=2)
    plt.plot(epochs, training_history['val_precisions'], label='Validation Precision (Class 1)', linewidth=2)
    plt.title('Precision for Class 1 (Positive)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Recall curves (Class 1)
    plt.subplot(2, 3, 4)
    plt.plot(epochs, training_history['train_recalls'], label='Training Recall (Class 1)', linewidth=2)
    plt.plot(epochs, training_history['val_recalls'], label='Validation Recall (Class 1)', linewidth=2)
    plt.title('Recall for Class 1 (Positive)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Cohen's Kappa curves
    if 'train_kappas' in training_history and 'val_kappas' in training_history:
        plt.subplot(2, 3, 5)
        plt.plot(epochs, training_history['train_kappas'], label='Training Kappa', linewidth=2)
        plt.plot(epochs, training_history['val_kappas'], label='Validation Kappa', linewidth=2)
        plt.title("Cohen's Kappa", fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel("Cohen's Kappa")
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

def evaluate_model_per_region(model, val_dataset, device, batch_size=16):
    """
    Evaluate model on validation set and collect predictions grouped by region.
    
    Args:
        model: PyTorch model
        val_dataset: Validation dataset (Subset of MuraDataset)
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        
    Returns:
        dict: Dictionary mapping region names to dicts with 'cm' (confusion matrix) and 'kappa' (Cohen's kappa)
    """
    from torch.utils.data import DataLoader
    
    model.eval()
    region_predictions = defaultdict(lambda: {'y_true': [], 'y_pred': [], 'y_proba': []})
    
    # Create a dataloader
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Get indices for validation set (if it's a Subset)
    if hasattr(val_dataset, 'indices') and hasattr(val_dataset, 'dataset'):
        # It's a Subset from StratifiedShuffleSplit
        indices = list(val_dataset.indices)
        dataset = val_dataset.dataset
    else:
        # It's the full dataset
        indices = list(range(len(val_dataset)))
        dataset = val_dataset
    
    # Verify dataset has samples attribute
    if not hasattr(dataset, 'samples'):
        raise AttributeError("Dataset must have a 'samples' attribute with region information")
    
    idx_counter = 0
    
    with torch.inference_mode():
        for batch_idx, (x, y) in enumerate(tqdm(val_loader, desc="Evaluating per region")):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            # Get probabilities for class 1 (positive class)
            probs = torch.softmax(y_hat, dim=1)[:, 1]  # Probability of class 1
            preds = y_hat.argmax(dim=1)
            
            # Get regions for this batch
            batch_size_actual = x.size(0)
            for i in range(batch_size_actual):
                if idx_counter < len(indices):
                    sample_idx = indices[idx_counter]
                    region = dataset.samples.iloc[sample_idx]['region']
                    region_predictions[region]['y_true'].append(y[i].cpu().item())
                    region_predictions[region]['y_pred'].append(preds[i].cpu().item())
                    region_predictions[region]['y_proba'].append(probs[i].cpu().item())
                    idx_counter += 1
    
    # Convert lists to numpy arrays and compute confusion matrices, kappa, and AUC
    region_cms = {}
    for region, data in region_predictions.items():
        if len(data['y_true']) > 0:
            y_true = np.array(data['y_true'])
            y_pred = np.array(data['y_pred'])
            y_proba = np.array(data['y_proba'])
            cm = confusion_matrix(y_true, y_pred)
            kappa = cohen_kappa_score(y_true, y_pred)
            # Calculate AUC-ROC
            try:
                auc = roc_auc_score(y_true, y_proba)
            except ValueError:
                # Handle case where only one class is present
                auc = 0.0
            region_cms[region] = {'cm': cm, 'kappa': kappa, 'y_true': y_true, 'y_proba': y_proba, 'auc': auc}
    
    return region_cms

def plot_region_confusion_matrices(region_cms, save_path='plots'):
    """
    Plot confusion matrices for each region with Cohen's kappa.
    
    Args:
        region_cms (dict): Dictionary mapping region names to dicts with 'cm' and 'kappa'
        save_path (str): Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    if not region_cms:
        print("No region confusion matrices to plot.")
        return
    
    # Sort regions for consistent ordering
    regions = sorted(region_cms.keys())
    n_regions = len(regions)
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_regions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_regions == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    print("\nCohen's Kappa and AUC-ROC per Region:")
    print("-" * 50)
    
    for i, region in enumerate(regions):
        region_data = region_cms[region]
        cm = region_data['cm']
        kappa = region_data['kappa']
        auc = region_data.get('auc', 0.0)
        ax = axes[i] if n_regions > 1 else axes[0]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Class 0', 'Class 1'],
                   yticklabels=['Class 0', 'Class 1'],
                   ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title(f'{region}\n(n={cm.sum()}, κ={kappa:.3f}, AUC={auc:.3f})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        
        print(f"{region:12s}: Kappa={kappa:.4f}, AUC={auc:.4f}")
    
    print("-" * 50)
    
    # Hide unused subplots
    for i in range(n_regions, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrices_per_region.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nRegion confusion matrices saved to {save_path}/confusion_matrices_per_region.png")

def calculate_region_metrics(region_cms):
    """
    Calculate comprehensive metrics for each region.
    
    Args:
        region_cms (dict): Dictionary mapping region names to dicts with 'cm' and 'kappa'
        
    Returns:
        dict: Dictionary mapping region names to metrics dict
    """
    region_metrics = {}
    
    for region, region_data in region_cms.items():
        cm = region_data['cm']
        kappa = region_data['kappa']
        
        # Calculate metrics from confusion matrix
        # Ensure confusion matrix is 2x2 for binary classification
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 4:
            # Reshape if needed
            cm_2d = cm.reshape(2, 2) if cm.ndim == 1 else cm
            tn, fp, fn, tp = cm_2d.ravel()
        else:
            # Handle edge cases (shouldn't happen for binary classification)
            tn, fp, fn, tp = 0, 0, 0, 0
            if cm.size >= 1:
                # Try to extract values
                flat = cm.flatten()
                if len(flat) >= 4:
                    tn, fp, fn, tp = flat[0], flat[1], flat[2], flat[3]
                elif len(flat) == 1:
                    # Only one class predicted
                    tn = flat[0] if flat[0] > 0 else 0
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        # Precision (Class 1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall (Class 1)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1-score (Class 1)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Specificity (Class 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Get AUC from region_data if available
        auc = region_data.get('auc', 0.0)
        
        region_metrics[region] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'kappa': kappa,
            'auc': auc,
            'n_samples': int(cm.sum()),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    return region_metrics

def plot_region_metrics(region_metrics, save_path='plots'):
    """
    Plot comprehensive metrics for each region.
    
    Args:
        region_metrics (dict): Dictionary mapping region names to metrics dict
        save_path (str): Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    if not region_metrics:
        print("No region metrics to plot.")
        return
    
    regions = sorted(region_metrics.keys())
    
    # Extract metrics
    accuracies = [region_metrics[r]['accuracy'] for r in regions]
    precisions = [region_metrics[r]['precision'] for r in regions]
    recalls = [region_metrics[r]['recall'] for r in regions]
    f1_scores = [region_metrics[r]['f1_score'] for r in regions]
    kappas = [region_metrics[r]['kappa'] for r in regions]
    aucs = [region_metrics[r]['auc'] for r in regions]
    specificities = [region_metrics[r]['specificity'] for r in regions]
    
    # Create figure with subplots (2x4 to include AUC)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 1. Accuracy
    axes[0, 0].bar(regions, accuracies, color='steelblue', alpha=0.7)
    axes[0, 0].set_title('Accuracy per Region', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_xticklabels(regions, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1])
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Precision
    axes[0, 1].bar(regions, precisions, color='coral', alpha=0.7)
    axes[0, 1].set_title('Precision (Class 1) per Region', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Precision', fontsize=12)
    axes[0, 1].set_xticklabels(regions, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 1])
    for i, v in enumerate(precisions):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Recall
    axes[0, 2].bar(regions, recalls, color='mediumseagreen', alpha=0.7)
    axes[0, 2].set_title('Recall (Class 1) per Region', fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel('Recall', fontsize=12)
    axes[0, 2].set_xticklabels(regions, rotation=45, ha='right')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    axes[0, 2].set_ylim([0, 1])
    for i, v in enumerate(recalls):
        axes[0, 2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. F1-Score
    axes[1, 0].bar(regions, f1_scores, color='gold', alpha=0.7)
    axes[1, 0].set_title('F1-Score (Class 1) per Region', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('F1-Score', fontsize=12)
    axes[1, 0].set_xticklabels(regions, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylim([0, 1])
    for i, v in enumerate(f1_scores):
        axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Cohen's Kappa
    axes[1, 1].bar(regions, kappas, color='orchid', alpha=0.7)
    axes[1, 1].set_title("Cohen's Kappa per Region", fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel("Cohen's Kappa", fontsize=12)
    axes[1, 1].set_xticklabels(regions, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 1])
    for i, v in enumerate(kappas):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 6. AUC-ROC
    axes[1, 2].bar(regions, aucs, color='crimson', alpha=0.7)
    axes[1, 2].set_title('AUC-ROC per Region', fontsize=14, fontweight='bold')
    axes[1, 2].set_ylabel('AUC-ROC', fontsize=12)
    axes[1, 2].set_xticklabels(regions, rotation=45, ha='right')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    axes[1, 2].set_ylim([0, 1])
    for i, v in enumerate(aucs):
        axes[1, 2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 7. Combined metrics comparison
    x = np.arange(len(regions))
    width = 0.12
    axes[1, 3].bar(x - 2.5*width, accuracies, width, label='Accuracy', alpha=0.7)
    axes[1, 3].bar(x - 1.5*width, precisions, width, label='Precision', alpha=0.7)
    axes[1, 3].bar(x - 0.5*width, recalls, width, label='Recall', alpha=0.7)
    axes[1, 3].bar(x + 0.5*width, f1_scores, width, label='F1-Score', alpha=0.7)
    axes[1, 3].bar(x + 1.5*width, kappas, width, label="Kappa", alpha=0.7)
    axes[1, 3].bar(x + 2.5*width, aucs, width, label="AUC-ROC", alpha=0.7)
    axes[1, 3].set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
    axes[1, 3].set_ylabel('Score', fontsize=12)
    axes[1, 3].set_xticks(x)
    axes[1, 3].set_xticklabels(regions, rotation=45, ha='right')
    axes[1, 3].legend(loc='upper left', fontsize=8)
    axes[1, 3].grid(True, alpha=0.3, axis='y')
    axes[1, 3].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/region_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Region metrics comparison saved to {save_path}/region_metrics_comparison.png")

def plot_roc_curves(region_cms, save_path='plots'):
    """
    Plot ROC curves for each region and overall validation set.
    
    Args:
        region_cms (dict): Dictionary mapping region names to dicts with 'y_true', 'y_proba', and 'auc'
        save_path (str): Directory to save plots
    """
    os.makedirs(save_path, exist_ok=True)
    
    if not region_cms:
        print("No region data to plot ROC curves.")
        return
    
    regions = sorted(region_cms.keys())
    
    plt.figure(figsize=(10, 8))
    
    # Collect all data for overall ROC curve
    all_y_true = []
    all_y_proba = []
    
    # Plot ROC curve for each region
    for region in regions:
        region_data = region_cms[region]
        y_true = region_data['y_true']
        y_proba = region_data['y_proba']
        auc = region_data.get('auc', 0.0)
        
        # Collect for overall calculation
        all_y_true.extend(y_true)
        all_y_proba.extend(y_proba)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, linewidth=2, label=f'{region} (AUC = {auc:.3f})')
    
    # Calculate and plot overall ROC curve
    if len(all_y_true) > 0 and len(all_y_proba) > 0:
        all_y_true = np.array(all_y_true)
        all_y_proba = np.array(all_y_proba)
        try:
            overall_auc = roc_auc_score(all_y_true, all_y_proba)
            overall_fpr, overall_tpr, _ = roc_curve(all_y_true, all_y_proba)
            # Plot overall ROC curve with thicker line and different style
            plt.plot(overall_fpr, overall_tpr, linewidth=3, linestyle='-', 
                    color='black', label=f'Overall (AUC = {overall_auc:.3f})', zorder=10)
            print(f"\nOverall Validation Set AUC-ROC: {overall_auc:.4f}")
        except ValueError as e:
            print(f"Warning: Could not calculate overall AUC: {e}")
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves per Region and Overall', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/roc_curves_per_region.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"ROC curves saved to {save_path}/roc_curves_per_region.png")

def print_region_metrics_table(region_metrics):
    """
    Print a formatted table of metrics for each region.
    
    Args:
        region_metrics (dict): Dictionary mapping region names to metrics dict
    """
    regions = sorted(region_metrics.keys())
    
    print("\n" + "=" * 100)
    print("REGION-WISE METRICS SUMMARY")
    print("=" * 100)
    print(f"{'Region':<12} {'N':<8} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Kappa':<10} {'AUC-ROC':<10} {'Specificity':<10}")
    print("-" * 110)
    
    for region in regions:
        m = region_metrics[region]
        print(f"{region:<12} {m['n_samples']:<8} {m['accuracy']:<10.4f} {m['precision']:<10.4f} "
              f"{m['recall']:<10.4f} {m['f1_score']:<10.4f} {m['kappa']:<10.4f} {m['auc']:<10.4f} {m['specificity']:<10.4f}")
    
    print("-" * 110)
    
    # Calculate averages
    avg_acc = np.mean([region_metrics[r]['accuracy'] for r in regions])
    avg_prec = np.mean([region_metrics[r]['precision'] for r in regions])
    avg_rec = np.mean([region_metrics[r]['recall'] for r in regions])
    avg_f1 = np.mean([region_metrics[r]['f1_score'] for r in regions])
    avg_kappa = np.mean([region_metrics[r]['kappa'] for r in regions])
    avg_auc = np.mean([region_metrics[r]['auc'] for r in regions])
    avg_spec = np.mean([region_metrics[r]['specificity'] for r in regions])
    total_samples = sum([region_metrics[r]['n_samples'] for r in regions])
    
    print(f"{'AVERAGE':<12} {total_samples:<8} {avg_acc:<10.4f} {avg_prec:<10.4f} "
          f"{avg_rec:<10.4f} {avg_f1:<10.4f} {avg_kappa:<10.4f} {avg_auc:<10.4f} {avg_spec:<10.4f}")
    print("=" * 110)

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
        if 'train_kappas' in training_history and training_history['train_kappas']:
            print(f"  Training Cohen's Kappa: {training_history['train_kappas'][-1]:.4f}")
            print(f"  Validation Cohen's Kappa: {training_history['val_kappas'][-1]:.4f}")
        
        # Create plots
        plot_training_curves(training_history, save_path)
        
    except FileNotFoundError:
        print(f"Training history file '{file_path}' not found.")
    except Exception as e:
        print(f"Error loading training history: {e}")

if __name__ == "__main__":
    # Example usage
    load_and_plot_training_history()
