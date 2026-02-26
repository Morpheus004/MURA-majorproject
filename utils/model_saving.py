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
        'train_kappas': [],
        'val_kappas': [],
        'confusion_matrices': [],
        'val_f2s': []
    }

def _extract_class1_metrics(metrics):
    """Helper to extract scalar values including class-1 precision/recall from a metrics dict."""
    loss = metrics['avg_loss']
    acc = metrics['accuracy']
    prec = metrics['precision'][1] if metrics.get('precision') is not None else 0.0
    rec = metrics['recall'][1] if metrics.get('recall') is not None else 0.0
    kappa = metrics.get('kappa', 0.0) if metrics.get('kappa') is not None else 0.0
    cm = metrics.get('confusion_matrix')
    f2 = metrics['f2_score'][1] if metrics.get('f2_score') is not None else 0.0
    return loss, acc, prec, rec, kappa, cm, f2

def update_history(history, epoch, train_metrics, val_metrics):
    """Append current epoch's metrics to history and return convenient val scalars."""
    train_loss, train_acc, train_prec, train_rec, train_kappa, _, train_f2 = _extract_class1_metrics(train_metrics)
    val_loss, val_acc, val_prec, val_rec, val_kappa, val_cm, val_f2 = _extract_class1_metrics(val_metrics)

    history['epochs'].append(epoch)
    history['train_losses'].append(train_loss)
    history['val_losses'].append(val_loss)
    history['train_accuracies'].append(train_acc)
    history['val_accuracies'].append(val_acc)
    history['train_precisions'].append(train_prec)
    history['val_precisions'].append(val_prec)
    history['train_recalls'].append(train_rec)
    history['val_recalls'].append(val_rec)
    history['val_f2s'].append(val_f2)
    history['train_kappas'].append(train_kappa)
    history['val_kappas'].append(val_kappa)
    if val_cm is not None:
        history['confusion_matrices'].append(val_cm.copy())

    return {'val_loss': val_loss, 'val_acc': val_acc, 'val_prec': val_prec, 'val_rec': val_rec, 'val_kappa': val_kappa, 'val_f2': val_f2}

def save_checkpoint_with_history(path, model, optimizer, epoch, best_metric, train_metrics, val_metrics, history, metric_name):
    s = f"best_{metric_name}"
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        s: best_metric,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'training_history': history
    }, path)
