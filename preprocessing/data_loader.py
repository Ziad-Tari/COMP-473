import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

def get_dataloader(dataset_name='celeba', dataroot='./data', image_size=64, 
                   batch_size=128, workers=2):
    """
    Dataloader for DCGAN training
    
    Args:
        dataset_name: 'celeba' or 'cifar10' datasets
        dataroot: Root directory for dataset
        image_size: Resize images (default: 64)
        batch_size: Batch size (default: 128)
        workers: Number of data loading worker (default: 2)
    
    Returns:
        dataloader: DataLoader Pytorch object ready for training
        dataset: Dataset Pytorch object (for checking sizes)
    """
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # Normalize to [-1, 1] since we use tanh on the last layer
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'celeba':
        print("Loading CelebA dataset (Download size: 1.4GB)")
        
        # Create ./data/ folder if not already there
        os.makedirs(dataroot, exist_ok=True)
        
        dataset = dset.CelebA(
            root=dataroot,
            split='train', # 80% train, 10% test, 10% val
            transform=transform,
            download=True
        )
        
        print(f"CelebA loaded ({len(dataset)} images)")
    
    elif dataset_name == 'cifar10':
        print("Loading CelebA dataset (Download size: 170MB)")
        
        os.makedirs(dataroot, exist_ok=True)
        
        dataset = dset.CIFAR10(
            root=dataroot,
            train=True,
            transform=transform,
            download=True
        )

        print(f"CIFAR-10 loaded ({len(dataset)} images)")
    else:
        print('Enter a valid dataset name (celeba or cifar10)')
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    
    print(f"Dataloader created: {len(dataloader)} batches per epoch")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size} by {image_size}")
    print(f"Total batches: {len(dataloader)}")
    
    return dataloader, dataset

def verify_dataloader(dataloader):
    """
    Loads one batch and prints info for quick sanity check
    """
    images, labels = next(iter(dataloader))
    
    print(f"Batch shape: {images.shape}")
    print(f"Batch type: {images.dtype}")
    print(f"Value range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Check if values are in correctly normalized
    if images.min() >= -1.1 and images.max() <= 1.1:
        print("Correct normalization (-1, 1)")
    else:
        print(f"Values not correctly normalized. Range: ({images.min()}, {images.max()})")
    
    if images.shape[1:] == (3, 64, 64):
        print("Dimensions are correct (3, 64, 64)")
    else:
        print(f"Expected (3, 64, 64), got {images.shape[1:]}")
        
    return images

# Example usage
if __name__ == "__main__":

    print("DCGAN Data Preprocessing Test\n")
    
    print("Testing with CIFAR-10")
    dataloader, dataset = get_dataloader(
        dataset_name='cifar10',
        dataroot='./data',
        image_size=64,
        batch_size=4,
        workers=2
    )
    
    images = verify_dataloader(dataloader)