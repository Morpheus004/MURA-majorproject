from utils.dataset import MuraDataset
from utils.early_stopping import EarlyStopping
from utils.model_saving import initialize_training_history, save_checkpoint_with_history, update_history
from utils.train_test_utils import train_model, test_model
import torch
from torch.utils.data import DataLoader, random_split, Subset
import timm
import random
import numpy as np
from utils.plot_utils import load_and_plot_training_history
import wandb
import os
from sklearn.model_selection import StratifiedShuffleSplit

def reproducibility(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    return torch.Generator().manual_seed(seed)

def main():
    # Initialize wandb
    wandb.init(
        project="mura-classification",
        config={
            "model": "inception_resnet_v2",
            "batch_size": 64,
            "learning_rate": 1e-4,
            "epochs": 20,
            "freeze_backbone_epochs": 2,
            "optimizer": "RMSprop",
            "scheduler": "ReduceLROnPlateau",
            "early_stopping_patience": 5,
            "num_classes": 2
        }
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()
        print("Using GPU")
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        print("Using CPU")


    full_dataset = MuraDataset(is_training=True, dir_path='./dataset/MURA-v1.1/')

    # val_size = int(0.1 * len(full_dataset))
    # train_size = len(full_dataset) - val_size
    # gen = reproducibility(42)
    # train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],generator=gen)

    full_dataset = MuraDataset(is_training=True, dir_path='/kaggle/input/mura-v11/MURA-v1.1/')
    df = full_dataset.samples
    df['region_label'] = df['region'].astype(str) + "_" + df['label'].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, val_indices = next(sss.split(np.zeros(len(full_dataset)), df.region_label.values))
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, generator=gen)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, generator=gen)

    model = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=2)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # minimize validation loss
        factor=0.5,           # reduce LR by half
        patience=3,           # wait 3 epochs before reducing
        min_lr=1e-6           # minimum LR threshold
    )
    # Optional: warmup with frozen backbone for first few epochs
    freeze_backbone_epochs = 2
    for name, param in model.named_parameters():
        if 'classif' not in name:
            param.requires_grad = False

    epochs = 20
    best_recall = 0.0
    best_loss = float('inf')
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, verbose= True)
    history = initialize_training_history()
    print("Starting training") 
    for epoch in range(1, epochs + 1):
        # Unfreeze after warmup
        if epoch == freeze_backbone_epochs + 1:
            for param in model.parameters():
                param.requires_grad = True
            for g in optimizer.param_groups:
                g['lr'] = 5e-5

        train_metrics = train_model(model, train_loader, criterion, optimizer, epoch, device)
        val_metrics = test_model(model, val_loader, criterion, device)

        scalars = update_history(history, epoch, train_metrics, val_metrics)
        val_loss = scalars['val_loss']
        val_recall = scalars['val_rec']
        val_kappa = scalars.get('val_kappa', 0.0)

        scheduler.step(val_loss)
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": train_metrics['avg_loss'],
            "train/accuracy": train_metrics['accuracy'],
            "train/precision": train_metrics.get('precision', [0, 0])[1] if train_metrics.get('precision') else 0.0,
            "train/recall": train_metrics.get('recall', [0, 0])[1] if train_metrics.get('recall') else 0.0,
            "train/kappa": train_metrics.get('kappa', 0.0) if train_metrics.get('kappa') is not None else 0.0,
            "val/loss": val_loss,
            "val/accuracy": scalars['val_acc'],
            "val/precision": scalars['val_prec'],
            "val/recall": val_recall,
            "val/kappa": val_kappa,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        

        if early_stopping(val_recall, model):
            print("Early stopping triggered")
            break
        
        if val_recall > best_recall and val_loss <= best_loss:
            best_recall = val_recall
            best_loss = val_loss
            save_checkpoint_with_history('best_model.pt', model, optimizer, epoch, best_recall, train_metrics, val_metrics, history)
    
    # Restore best weights at the end
    early_stopping.restore_weights(model)
    
    # Save final training history
    print(f"Training completed! Best recall: {best_recall:.4f}")
    
    # Log best recall to wandb
    wandb.log({"best_recall": best_recall})
    
    # Log the best model artifact
    if os.path.exists('best_model.pt'):
        artifact = wandb.Artifact('best_model', type='model')
        artifact.add_file('best_model.pt')
        wandb.log_artifact(artifact)
        print("Model artifact logged to wandb")


if __name__ == '__main__':
    main()
    print("=" * 60)
    print("TRAINING RESULTS VISUALIZATION")
    print("=" * 60)

    # Check if training history exists
    history_file = 'best_model.pt'
    if not os.path.exists(history_file):
        print(f"❌ Training history file '{history_file}' not found!")
        print("Please run training first with: python main.py")
        # return 0

    print(f"📊 Loading training history from '{history_file}'...")

    # Generate plots
    try:
        load_and_plot_training_history(file_path=history_file, save_path='plots')
        print("\n✅ Plots generated successfully!")
        print("📁 Check the 'plots' directory for:")
        print("   • training_curves.png - Loss, accuracy, precision, recall curves")
        print("   • precision_vs_recall.png - Precision vs Recall trade-off")
        print("   • confusion_matrices.png - Confusion matrices over time")
        
        # Log plots to wandb
        if wandb.run is not None:
            plots_dir = 'plots'
            plot_files = ['training_curves.png', 'precision_vs_recall.png', 'confusion_matrices.png']
            for plot_file in plot_files:
                plot_path = os.path.join(plots_dir, plot_file)
                if os.path.exists(plot_path):
                    wandb.log({f"plots/{plot_file.replace('.png', '')}": wandb.Image(plot_path)})
                    print(f"   ✅ Logged {plot_file} to wandb")
        else:
            print("   ⚠️  Wandb run not active, skipping plot logging")
        
    except Exception as e:
        print(f"❌ Error generating plots: {e}")
        # return

    print("\n" + "=" * 60)
    print("PLOTTING COMPLETE!")
    print("=" * 60)
    
    # Finish wandb run
    if wandb.run is not None:
        wandb.finish()
        print("Wandb run completed!")


