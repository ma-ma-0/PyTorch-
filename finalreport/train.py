import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import cifar_dataset
from model import CNN

def main():
    # デバイスの設定 (GPUが使えるならGPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ハイパーパラメータ
    epochs = 20
    batch_size = 64
    learning_rate = 0.01

    # 1. データセットの読み込み
    train_data, _ = cifar_dataset() # テストデータはここでは使わない
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # 2. モデル、損失関数、最適化関数の定義
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # 3. 学習ループ
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 勾配の初期化
            optimizer.zero_grad()

            # 順伝播 (Forward)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 逆伝播 (Backward) とパラメータ更新
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # 精度の計算（参考用）
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        avg_acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

    # 4. モデルの保存
    # 課題PDFの指定に従い、state_dictを含めて保存
    save_path = 'cifar_cnn.pth'
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()