from utils.dataset import MuraDataset
from utils.early_stopping import EarlyStopping
from utils.model_saving import initialize_training_history, save_checkpoint_with_history, update_history
from utils.train_test_utils import train_model, test_model
import torch
from torch.utils.data import DataLoader, random_split
import timm
import random
import numpy as np

def reproducibility(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    return torch.Generator().manual_seed(seed)

def main():
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

    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    gen = reproducibility(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],generator=gen)

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

        scheduler.step(val_loss)

        if early_stopping(val_recall, model):
            print("Early stopping triggered")
            break
        
        if val_recall > best_recall:
            best_recall = val_recall
            save_checkpoint_with_history('best_model.pt', model, optimizer, epoch, best_recall, train_metrics, val_metrics, history)
    
    # Restore best weights at the end
    early_stopping.restore_weights(model)
    
    # Save final training history
    print(f"Training completed! Best recall: {best_recall:.4f}")


if __name__ == '__main__':
    main()


