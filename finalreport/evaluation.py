import torch
from torch.utils.data import DataLoader
from dataset import cifar_dataset
from model import CNN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. テストデータの読み込み
    _, test_data = cifar_dataset()
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # 2. モデルの定義と読み込み
    model = CNN().to(device)
    model_path = 'cifar_cnn.pth'
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please run train.py first.")
        return

    # 3. 推論と評価
    model.eval() # 評価モードに設定
    correct = 0
    total = 0
    
    with torch.no_grad(): # 勾配計算を行わない
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy on test images: {accuracy:.4f}')
    
    if accuracy >= 0.55:
        print("精度は55%以上です！")
    else:
        print("精度は55%未満です。")

if __name__ == "__main__":
    main()