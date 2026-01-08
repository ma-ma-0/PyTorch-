import torch
from torchvision import datasets, transforms

def cifar_dataset():
    # 講座5: 画像の前処理 (ToTensorとNormalize)
    # CIFAR-10の平均と標準偏差で正規化を行うのが一般的です
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 講座6: データセットのダウンロードと読み込み
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