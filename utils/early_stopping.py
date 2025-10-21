import numpy as np
import torch
import copy

class EarlyStopping:
    """
    Early stopping utility to stop training when validation recall for class 1 stops improving.
    
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
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, val_recall, model):
        """
        Check if early stopping criteria is met.
        
        Args:
            val_recall (float): Current validation recall for class 1
            model: PyTorch model to potentially restore weights from
            
        Returns:
            bool: True if early stopping should be triggered
        """
        score = val_recall
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'Early stopping triggered! Best recall: {self.best_score:.4f}')
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
            
        return self.early_stop
    
    def save_checkpoint(self, model):
        """Save model weights when validation recall improves."""
        if self.restore_best_weights:
            self.best_weights = copy.deepcopy(model.state_dict())
    
    def restore_weights(self, model):
        """Restore model to best weights."""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print("Model weights restored to best epoch")

