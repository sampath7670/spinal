import os
import argparse
import shutil
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from preprocessing import get_dataloaders
import matplotlib.pyplot as plt

def train_model(data_dir, model_save_path, epochs=10, batch_size=32, lr=0.001, limit_batches=None, limit_yolo_files=None):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataloaders
    print("Loading data...")
    train_loader, val_loader, _, classes = get_dataloaders(data_dir, batch_size=batch_size)
    num_classes = len(classes)
    print(f"Dataset classes: {classes} (Count: {num_classes})")
    
    # Initialize pre-trained model (DenseNet121)
    print("Initializing DenseNet121 model...")
    # Use weights parameter for newer torchvision versions
    try:
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    except AttributeError:
        # Fallback for older torchvision versions
        model = models.densenet121(pretrained=True)
        
    # Replace the classification head
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Tracking training metrics
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    
    print("Starting training loop...")
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0
        
        batch_count = 0
        for inputs, labels in train_loader:
            if limit_batches is not None and batch_count >= limit_batches:
                break
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_train += inputs.size(0)
            batch_count += 1
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = running_corrects.double() / total_train
        
        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        running_val_corrects = 0
        total_val = 0
        
        val_batch_count = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                if limit_batches is not None and val_batch_count >= limit_batches:
                    break
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                running_val_loss += loss.item() * inputs.size(0)
                running_val_corrects += torch.sum(preds == labels.data)
                total_val += inputs.size(0)
                val_batch_count += 1
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = running_val_corrects.double() / total_val
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
              
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc.item())
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc.item())
        
        # Save best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            # Save state dict and class mapping
            checkpoint = {
                'state_dict': model.state_dict(),
                'classes': classes
            }
            torch.save(checkpoint, model_save_path)
            print(f"Saved new best model checkpoint to {model_save_path}")
            
    print("Training complete!")
    plot_training_curves(history)
    
    # --- YOLO Object Detection Phase ---
    print("\n==================================================")
    print("Preparing YOLOv8 dataset and training detector...")
    print("==================================================")
    try:
        yaml_path = prepare_yolo_dataset(data_dir, limit_files=limit_yolo_files)
        from ultralytics import YOLO
        print("Initializing YOLOv8n detector...")
        yolo_model = YOLO('yolov8n.pt')
        print("Training YOLOv8n detector...")
        # Train for a few epochs since the dataset is very small
        yolo_model.train(
            data=yaml_path,
            epochs=3,
            imgsz=224,
            batch=4,
            device='cpu' if device.type == 'cpu' else 0,
            workers=0,
            verbose=True
        )
        
        # Save best weights to models/best_yolo.pt
        best_weights_path = os.path.join(yolo_model.trainer.save_dir, 'weights', 'best.pt')
        project_root = os.path.dirname(os.path.dirname(model_save_path))
        yolo_save_path = os.path.join(project_root, 'models', 'best_yolo.pt')
        shutil.copy(best_weights_path, yolo_save_path)
        print(f"Successfully saved trained YOLO weights to {yolo_save_path}")
    except Exception as e:
        print(f"Error training YOLOv8: {e}")

def prepare_yolo_dataset(data_dir, limit_files=None):
    """
    Prepares a YOLO-compatible dataset structure using classification images.
    Generates synthetic bounding box annotations for the 'fracture' class.
    """
    yolo_dir = os.path.join(data_dir, 'yolo')
    if os.path.exists(yolo_dir):
        shutil.rmtree(yolo_dir)
        
    os.makedirs(os.path.join(yolo_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, 'val', 'labels'), exist_ok=True)
    
    # Process train and validation splits
    for split in ['train', 'validation', 'val']:
        yolo_split = 'val' if split in ['validation', 'val'] else 'train'
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
            
        for cls in ['fracture', 'no_fracture', 'normal']:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.exists(cls_dir):
                continue
                
            file_count = 0
            for fname in os.listdir(cls_dir):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue
                    
                if limit_files is not None and file_count >= limit_files:
                    break
                
                src_img_path = os.path.join(cls_dir, fname)
                dst_img_path = os.path.join(yolo_dir, yolo_split, 'images', f"{cls}_{fname}")
                shutil.copy(src_img_path, dst_img_path)
                
                # Bounding box label path
                label_name = f"{cls}_{os.path.splitext(fname)[0]}.txt"
                dst_lbl_path = os.path.join(yolo_dir, yolo_split, 'labels', label_name)
                
                if cls == 'fracture':
                    # Create a synthetic bounding box centered with slight offset/noise
                    # Format: class x_center y_center width height (normalized 0-1)
                    random.seed(hash(fname))
                    xc = round(random.uniform(0.45, 0.55), 4)
                    yc = round(random.uniform(0.4, 0.6), 4)
                    w = round(random.uniform(0.2, 0.35), 4)
                    h = round(random.uniform(0.15, 0.3), 4)
                    
                    with open(dst_lbl_path, 'w') as lf:
                        lf.write(f"0 {xc} {yc} {w} {h}\n")
                else:
                    # Empty label file for normal (no fracture) cases
                    with open(dst_lbl_path, 'w') as lf:
                        pass
                        
                file_count += 1
                        
    # Write dataset.yaml
    yaml_content = f"path: {os.path.abspath(yolo_dir)}\ntrain: train/images\nval: val/images\n\nnames:\n  0: fracture\n"
    yaml_path = os.path.join(yolo_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as yf:
        yf.write(yaml_content)
        
    return yaml_path

def plot_training_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'training_curves.png'))
    plt.savefig(plot_path)
    print(f"Saved training curves plot to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cervical Fracture AI Model")
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--model_save_path', type=str, default='models/best_model.pth', help='Path to save best model checkpoint')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--limit_batches', type=int, default=None, help='Limit batches per epoch')
    parser.add_argument('--limit_yolo_files', type=int, default=None, help='Limit files for YOLO training')
    
    args = parser.parse_args()
    
    # Resolve relative paths relative to repository root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_dir = os.path.join(project_root, args.dataset_dir)
    model_save_path = os.path.join(project_root, args.model_save_path)
    
    train_model(
        data_dir=dataset_dir,
        model_save_path=model_save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        limit_batches=args.limit_batches,
        limit_yolo_files=args.limit_yolo_files
    )
