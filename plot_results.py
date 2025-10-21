#!/usr/bin/env python3
"""
Script to plot training results from saved training history.
Run this after training to generate comprehensive plots.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.plot_utils import load_and_plot_training_history

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
        print("   • training_curves.png - Loss, accuracy, precision, recall curves")
        print("   • precision_vs_recall.png - Precision vs Recall trade-off")
        print("   • confusion_matrices.png - Confusion matrices over time")
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        return
    
    print("\n" + "=" * 60)
    print("PLOTTING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
