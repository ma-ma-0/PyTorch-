import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 講座3, 6: Conv2d, ReLU, MaxPool2d, Linearの定義
        # 入力画像: 3チャネル, 32x32
        
        # 畳み込み層1: 3ch -> 16ch, 3x3カーネル
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 畳み込み層2: 16ch -> 32ch, 3x3カーネル
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        
        # プーリング層: 2x2でサイズを半分にする
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全結合層
        # 画像サイズの変化: 32x32 ->(pool)-> 16x16 ->(pool)-> 8x8
        # 最終的な特徴量サイズ: 32ch * 8 * 8
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10) # 10クラス分類
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Conv -> ReLU -> Pool
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        # 1列に展開 (Flatten)
        x = x.view(-1, 32 * 8 * 8)
        
        # 全結合層
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = CNN()
    print(model)