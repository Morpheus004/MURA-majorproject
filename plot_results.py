#!/usr/bin/env python3
"""
Script to plot training results from saved training history.
Run this after training to generate comprehensive plots.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.plot_utils import (
    load_and_plot_training_history, 
    evaluate_model_per_region, 
    plot_region_confusion_matrices,
    calculate_region_metrics,
    plot_region_metrics,
    print_region_metrics_table
)
from utils.dataset import MuraDataset
from torch.utils.data import Subset
import torch
import timm
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def main():
    print("=" * 60)
    print("TRAINING RESULTS VISUALIZATION")
    print("=" * 60)
    
    # Check if training history exists
    history_file = 'best_model.pt'
    if not os.path.exists(history_file):
        print(f"❌ Training history file '{history_file}' not found!")
        print("Please run training first with: python main.py")
        return
    
    print(f"📊 Loading training history from '{history_file}'...")
    
    # Generate plots
    try:
        load_and_plot_training_history(history_file, save_path='plots')
        print("\n✅ Plots generated successfully!")
        print("📁 Check the 'plots' directory for:")
        print("   • training_curves.png - Loss, accuracy, precision, recall, kappa curves")
        print("   • precision_vs_recall.png - Precision vs Recall trade-off")
        print("   • confusion_matrices.png - Confusion matrices over time")
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate per-region confusion matrices and metrics
    print("\n" + "=" * 60)
    print("GENERATING PER-REGION ANALYSIS")
    print("=" * 60)
    
    try:
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(history_file, map_location=device, weights_only=False)
        
        # Create model
        model = timm.create_model('inception_resnet_v2', pretrained=False, num_classes=2)
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        model.eval()
        
        print("✅ Model loaded successfully")
        
        # Create validation dataset using the same split method as training
        # Match exactly the split from main.py (lines 61-67)
        dataset_path = '/kaggle/input/mura-v11/MURA-v1.1/'
        if not os.path.exists(dataset_path):
            # Fallback for local development
            dataset_path = './dataset/MURA-v1.1/'
        
        full_dataset = MuraDataset(is_training=True, dir_path=dataset_path)
        df = full_dataset.samples
        df['region_label'] = df['region'].astype(str) + "_" + df['label'].astype(str)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_indices, val_indices = next(sss.split(np.zeros(len(full_dataset)), df.region_label.values))
        val_dataset = Subset(full_dataset, val_indices)
        
        print("✅ Validation dataset created")
        print(f"   Validation set size: {len(val_dataset)}")
        
        # Evaluate per region
        print("\n📊 Evaluating model per region...")
        region_cms = evaluate_model_per_region(model, val_dataset, device, batch_size=16)
        
        # Calculate comprehensive metrics per region
        print("\n📈 Calculating metrics per region...")
        region_metrics = calculate_region_metrics(region_cms)
        
        # Print metrics table
        print_region_metrics_table(region_metrics)
        
        # Plot confusion matrices per region
        print("\n📊 Plotting confusion matrices per region...")
        plot_region_confusion_matrices(region_cms, save_path='plots')
        
        # Plot metrics comparison per region
        print("\n📊 Plotting metrics comparison per region...")
        plot_region_metrics(region_metrics, save_path='plots')
        
        print("\n✅ Per-region analysis generated successfully!")
        print("📁 Check the 'plots' directory for:")
        print("   • confusion_matrices_per_region.png - Confusion matrices for each region")
        print("   • region_metrics_comparison.png - Comprehensive metrics comparison per region")
        
    except Exception as e:
        print(f"❌ Error generating per-region analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("PLOTTING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
