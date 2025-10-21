import torch

def initialize_training_history():
    """Create an empty training history dictionary for plotting and analysis."""
    return {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': [],
        'train_precisions': [],  # class 1
        'val_precisions': [],    # class 1
        'train_recalls': [],     # class 1
        'val_recalls': [],       # class 1
        'confusion_matrices': []
    }

def _extract_class1_metrics(metrics):
    """Helper to extract scalar values including class-1 precision/recall from a metrics dict."""
    loss = metrics['avg_loss']
    acc = metrics['accuracy']
    prec = metrics['precision'][1] if metrics.get('precision') is not None else 0.0
    rec = metrics['recall'][1] if metrics.get('recall') is not None else 0.0
    cm = metrics.get('confusion_matrix')
    return loss, acc, prec, rec, cm

def update_history(history, epoch, train_metrics, val_metrics):
    """Append current epoch's metrics to history and return convenient val scalars."""
    train_loss, train_acc, train_prec, train_rec, _ = _extract_class1_metrics(train_metrics)
    val_loss, val_acc, val_prec, val_rec, val_cm = _extract_class1_metrics(val_metrics)

    history['epochs'].append(epoch)
    history['train_losses'].append(train_loss)
    history['val_losses'].append(val_loss)
    history['train_accuracies'].append(train_acc)
    history['val_accuracies'].append(val_acc)
    history['train_precisions'].append(train_prec)
    history['val_precisions'].append(val_prec)
    history['train_recalls'].append(train_rec)
    history['val_recalls'].append(val_rec)
    if val_cm is not None:
        history['confusion_matrices'].append(val_cm.copy())

    return {'val_loss': val_loss, 'val_acc': val_acc, 'val_prec': val_prec, 'val_rec': val_rec}

def save_checkpoint_with_history(path, model, optimizer, epoch, best_recall, train_metrics, val_metrics, history):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_recall': best_recall,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'training_history': history
    }, path)
