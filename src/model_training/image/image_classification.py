import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
from torchvision.models import ResNet18_Weights

from tqdm import tqdm
import time

class BatteryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []

        # Load all images from each class
        for chemistry in self.classes:
            chemistry_path = os.path.join(root_dir, chemistry)
            if not os.path.isdir(chemistry_path):
                continue
            png_files = [f for f in os.listdir(chemistry_path) if f.endswith('.png')]
            for img_name in png_files:
                self.images.append(os.path.join(chemistry_path, img_name))
                self.labels.append(self.class_to_idx[chemistry])
        
        print(f"Loaded {len(self.images)} images from {os.path.basename(root_dir)}")
        for chemistry in self.classes:
            count = sum([1 for lbl in self.labels if lbl == self.class_to_idx[chemistry]])
            print(f"  {chemistry}: {count} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def train_model(model, train_loader, val_loader, criterion, optimizer, save_dir, num_epochs=10, device='cuda'):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_acc = 0.0
    
    # Lists to store metrics
    metrics = {
        'train_losses': [], 'train_accs': [],
        'val_losses': [], 'val_accs': [],
        'epochs': [], 'times': []
    }
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Add progress bar for training
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase with progress bar
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{val_loss/total:.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        epoch_time = time.time() - epoch_start
        
        # Store metrics
        metrics['train_losses'].append(train_loss)
        metrics['train_accs'].append(train_acc)
        metrics['val_losses'].append(val_loss)
        metrics['val_accs'].append(val_acc)
        metrics['epochs'].append(epoch + 1)
        metrics['times'].append(epoch_time)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{num_epochs} Summary:')
        print(f'Time: {epoch_time:.2f}s')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'metrics': metrics
            }, model_save_path)
            print(f'Saved best model to {model_save_path}')
    
    total_time = time.time() - start_time
    print(f'\nTraining completed in {total_time/60:.2f} minutes')
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(metrics['epochs'], metrics['train_losses'], label='Train')
    plt.plot(metrics['epochs'], metrics['val_losses'], label='Validation')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(metrics['epochs'], metrics['train_accs'], label='Train')
    plt.plot(metrics['epochs'], metrics['val_accs'], label='Validation')
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_progress.png'))
    plt.close()
    
    return metrics

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels

def main():
    start_time = time.time()  # Track runtime

    # Paths for train/val/test datasets
    base_path = os.path.join('model_prep')
    train_dir = os.path.join(base_path, 'train')
    val_dir   = os.path.join(base_path, 'val')
    test_dir  = os.path.join(base_path, 'test')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = BatteryDataset(train_dir, transform=transform)
    val_dataset   = BatteryDataset(val_dir, transform=transform)
    test_dataset  = BatteryDataset(test_dir, transform=transform)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    save_dir = os.path.join(os.path.dirname(base_path), 'stored_models', 'image_based_training')
    os.makedirs(save_dir, exist_ok=True)

    # Train
    metrics = train_model(model, train_loader, val_loader, criterion, optimizer,
                          save_dir=save_dir, num_epochs=10, device=device)

    # Load best model and evaluate
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    preds, labels = evaluate_model(model, test_loader, device)

    # Print classification metrics
    report = classification_report(labels, preds, target_names=train_dataset.classes, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=train_dataset.classes))

    # Print overall precision, recall, F1
    precision = np.mean([report[cls]['precision'] for cls in train_dataset.classes])
    recall = np.mean([report[cls]['recall'] for cls in train_dataset.classes])
    f1 = np.mean([report[cls]['f1-score'] for cls in train_dataset.classes])
    print(f"\nOverall Precision: {precision:.4f}")
    print(f"Overall Recall: {recall:.4f}")
    print(f"Overall F1-score: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Total runtime
    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time/60:.2f} minutes")


if __name__ == "__main__":
    main()
