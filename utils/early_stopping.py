import numpy as np
import torch
import copy

class EarlyStopping:
    """
    Early stopping utility to stop training when validation recall for class 1 stops improving
    and validation loss doesn't decrease.
    
    Args:
        patience (int): Number of epochs to wait after last improvement before stopping
        min_delta (float): Minimum change to qualify as an improvement
        restore_best_weights (bool): Whether to restore model weights from the best epoch
        verbose (bool): Whether to print early stopping messages
    """
    
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None
        self.best_epoch = None
        
    def __call__(self, val_recall, val_loss, model, epoch=None):
        """
        Check if early stopping criteria is met.
        
        Args:
            val_recall (float): Current validation recall for class 1
            val_loss (float): Current validation loss
            model: PyTorch model to potentially restore weights from
            epoch (int, optional): Current epoch index to record when improvement occurs
            
        Returns:
            bool: True if early stopping should be triggered
        """
        score = val_recall
        
        # Check if this is an improvement: recall improved AND loss didn't increase
        is_improvement = False
        if self.best_score is None:
            # First epoch - always save
            is_improvement = True
        elif score > self.best_score + self.min_delta and val_loss <= self.best_loss:
            # Recall improved significantly AND loss didn't increase
            is_improvement = True
        
        if is_improvement:
            self.best_score = score
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model, epoch)
            if self.verbose:
                print(f'EarlyStopping: Improvement! Recall: {score:.4f}, Loss: {val_loss:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience} (Best recall: {self.best_score:.4f}, Best loss: {self.best_loss:.4f})')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'Early stopping triggered! Best recall: {self.best_score:.4f}, Best loss: {self.best_loss:.4f}')
            
        return self.early_stop
    
    def save_checkpoint(self, model, epoch=None):
        """Save model weights when validation recall improves."""
        if self.restore_best_weights:
            self.best_weights = copy.deepcopy(model.state_dict())
            if epoch is not None:
                self.best_epoch = epoch
    
    def restore_weights(self, model):
        """Restore model to best weights."""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                if self.best_epoch is not None:
                    print(f"Model weights restored to best epoch {self.best_epoch}")
                else:
                    print("Model weights restored to best epoch")

