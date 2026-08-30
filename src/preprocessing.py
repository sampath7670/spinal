import os
import random
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (FileNotFoundError, OSError, Exception) as e:
            print(f"\nWARNING: Skipping corrupted/missing file at index {index} (path: {self.samples[index][0]}). Error: {e}")
            # Try to return a random valid sample
            for _ in range(10):
                alt_idx = random.randint(0, len(self.samples) - 1)
                try:
                    return super().__getitem__(alt_idx)
                except Exception:
                    continue
            raise e

def get_transforms(img_size=224):
    """
    Returns train and val/test transforms.
    Includes data augmentation for the training split.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform

def get_dataloaders(data_dir, batch_size=32, img_size=224):
    """
    Creates and returns DataLoaders for train, validation, and test splits.
    
    Parameters:
        data_dir (str): Base path to the dataset folder containing train, validation, test splits.
        batch_size (int): Batch size for DataLoader.
        img_size (int): Target dimensions to resize images.
        
    Returns:
        train_loader (DataLoader): PyTorch DataLoader for training data.
        val_loader (DataLoader): PyTorch DataLoader for validation data.
        test_loader (DataLoader): PyTorch DataLoader for test data.
        classes (list): List of class names (e.g. ['fracture', 'no_fracture']).
    """
    train_transform, val_test_transform = get_transforms(img_size)
    
    # Search for train directory
    train_dir = None
    for name in ['train', 'training']:
        path = os.path.join(data_dir, name)
        if os.path.exists(path) and os.path.isdir(path):
            train_dir = path
            break
            
    # Search for validation directory
    val_dir = None
    for name in ['validation', 'val']:
        path = os.path.join(data_dir, name)
        if os.path.exists(path) and os.path.isdir(path):
            val_dir = path
            break
            
    # Search for test directory (optional fallback to val)
    test_dir = None
    for name in ['test', 'testing']:
        path = os.path.join(data_dir, name)
        if os.path.exists(path) and os.path.isdir(path):
            test_dir = path
            break

    if train_dir is None:
        raise FileNotFoundError(f"Could not find a train directory under {data_dir}")
    if val_dir is None:
        raise FileNotFoundError(f"Could not find a validation directory under {data_dir}")
        
    if test_dir is None:
        print("WARNING: Test directory not found. Using validation directory for test split.")
        test_dir = val_dir

    # Load datasets
    train_dataset = SafeImageFolder(train_dir, transform=train_transform)
    val_dataset = SafeImageFolder(val_dir, transform=val_test_transform)
    test_dataset = SafeImageFolder(test_dir, transform=val_test_transform)
    
    classes = train_dataset.classes
    
    # Create DataLoaders
    # num_workers=0 is safer across OS platforms (especially Windows)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, classes

if __name__ == "__main__":
    # Test script execution
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    print(f"Testing dataloader generation using dataset path: {dataset_path}")
    try:
        train_loader, val_loader, test_loader, classes = get_dataloaders(dataset_path, batch_size=4)
        print("DataLoaders loaded successfully!")
        print(f"Classes: {classes}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"Error loading dataloaders: {e}")
