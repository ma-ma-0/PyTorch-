import torch
from torchvision import datasets, transforms

def cifar_dataset():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_data = datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )

    return train_data, test_data

if __name__ == "__main__":
    train_data, test_data = cifar_dataset()
    print(f"Train data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")