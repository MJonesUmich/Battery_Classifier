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
    def __init__(self, root_dir, transform=None, max_samples_per_class=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        
        # Count images per class first
        class_counts = {}
        for chemistry in self.classes:
            chemistry_path = os.path.join(root_dir, chemistry)
            png_files = [f for f in os.listdir(chemistry_path) if f.endswith('.png')]
            class_counts[chemistry] = len(png_files)
        
        # Get the minimum count across classes if max_samples not specified
        if max_samples_per_class is None:
            max_samples_per_class = min(class_counts.values())
        
        print(f"Using {max_samples_per_class} samples per class in {os.path.basename(root_dir)}")
        
        # Load balanced number of images per class
        for chemistry in self.classes:
            chemistry_path = os.path.join(root_dir, chemistry)
            png_files = [f for f in os.listdir(chemistry_path) if f.endswith('.png')]
            
            # Randomly sample if we have more images than max_samples_per_class
            if len(png_files) > max_samples_per_class:
                png_files = np.random.choice(png_files, max_samples_per_class, replace=False)
            
            for img_name in png_files:
                self.images.append(os.path.join(chemistry_path, img_name))
                self.labels.append(self.class_to_idx[chemistry])

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

def main(max_train_samples_per_chemistry = 50):
    # Set paths
    data_dir = r'C:\Users\MJone\Documents\SIADS699\processed_images\model_prep'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets with sample limit for training
    train_dataset = BatteryDataset(train_dir, transform=transform, max_samples_per_class=max_train_samples_per_chemistry)
    val_dataset = BatteryDataset(val_dir, transform=transform)  # No limit for validation
    test_dataset = BatteryDataset(test_dir, transform=transform)  # No limit for testing
    
    # Print dataset sizes
    print(f"\nDataset sizes:")
    print(f"Training: {len(train_dataset)} images")
    print(f"Validation: {len(val_dataset)} images")
    print(f"Testing: {len(test_dataset)} images")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load pre-trained ResNet model
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Define model save directory
    save_dir = os.path.join(os.path.dirname(data_dir), 'stored_models', 'image_based_training')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Model will be saved to: {save_dir}")
    
    # Train the model with save directory
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        save_dir=save_dir, num_epochs=10, device=device
    )
    
    # Load best model for testing (update path)
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    all_preds, all_labels = evaluate_model(model, test_loader, device)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                              target_names=test_dataset.classes))
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    main(max_train_samples_per_chemistry = 10)
