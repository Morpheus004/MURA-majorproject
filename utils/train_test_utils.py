import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

def train_model(model, train_loader, criterion, optimizer, epoch, device, batch_log_freq = 10):
    model.train()
    running_loss = 0
    total_correct = 0
    total_seen = 0
    all_preds = []
    all_targets = []

    for batch_idx, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = y_hat.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_seen += y.numel()
        all_preds.append(preds.detach().cpu())
        all_targets.append(y.detach().cpu())

    avg_loss = running_loss / len(train_loader)
    accuracy = total_correct / max(1, total_seen)
    print(f"Epoch {epoch} Average Loss = {avg_loss:.4f} | Accuracy = {accuracy:.4%}")

    # Detailed metrics for training
    if all_preds and all_targets:
        y_true = torch.cat(all_targets).numpy()
        y_pred = torch.cat(all_preds).numpy()
        print("\nTraining Classification report:")
        print(classification_report(y_true, y_pred, digits=4))
        cm = confusion_matrix(y_true, y_pred)
        print("Training Confusion matrix (rows=true, cols=pred):")
        print(cm)
        print("\n")
        
        # Calculate precision and recall for each class
        precision_scores = []
        recall_scores = []
        for i in range(cm.shape[0]):
            # Precision = TP / (TP + FP)
            if cm[:, i].sum() > 0:
                precision = cm[i, i] / cm[:, i].sum()
            else:
                precision = 0.0
            precision_scores.append(precision)
            
            # Recall = TP / (TP + FN)
            if cm[i, :].sum() > 0:
                recall = cm[i, i] / cm[i, :].sum()
            else:
                recall = 0.0
            recall_scores.append(recall)
        
        return {
            'avg_loss': avg_loss,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'precision': precision_scores,
            'recall': recall_scores
        }
    else:
        return {
            'avg_loss': avg_loss,
            'accuracy': accuracy,
            'confusion_matrix': None,
            'precision': None,
            'recall': None
        }

def test_model(model, test_loader, criterion, device, batch_log_freq = 10):
    model.eval()
    running_loss = 0
    total_correct = 0
    total_seen = 0
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for batch_idx, (x, y) in enumerate(tqdm(test_loader)):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            running_loss += loss.item()
            preds = y_hat.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_seen += y.numel()
            all_preds.append(preds.detach().cpu())
            all_targets.append(y.detach().cpu())

    avg_loss = running_loss / len(test_loader)
    accuracy = total_correct / max(1, total_seen)
    print(f"Average Loss = {avg_loss:.4f} | Accuracy = {accuracy:.4%}")

    # Detailed metrics
    if all_preds and all_targets:
        y_true = torch.cat(all_targets).numpy()
        y_pred = torch.cat(all_preds).numpy()
        print("\nValidation Classification report:")
        print(classification_report(y_true, y_pred, digits=4))
        cm = confusion_matrix(y_true, y_pred)
        print("Validation Confusion matrix (rows=true, cols=pred):")
        print(cm)
        print("\n")
        
        # Calculate precision and recall for each class
        precision_scores = []
        recall_scores = []
        for i in range(cm.shape[0]):
            # Precision = TP / (TP + FP)
            if cm[:, i].sum() > 0:
                precision = cm[i, i] / cm[:, i].sum()
            else:
                precision = 0.0
            precision_scores.append(precision)
            
            # Recall = TP / (TP + FN)
            if cm[i, :].sum() > 0:
                recall = cm[i, i] / cm[i, :].sum()
            else:
                recall = 0.0
            recall_scores.append(recall)
        
        return {
            'avg_loss': avg_loss,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'precision': precision_scores,
            'recall': recall_scores
        }
    else:
        return {
            'avg_loss': avg_loss,
            'accuracy': accuracy,
            'confusion_matrix': None,
            'precision': None,
            'recall': None
        }