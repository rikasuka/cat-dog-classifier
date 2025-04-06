import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, models
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ====================================
# 猫と犬を分類する画像分類モデル（ResNet18）
# モデル訓練・評価・可視化・保存まで含む
# ====================================

# ------------------------
# データ前処理とデータローダー
# ------------------------

# データ拡張と正規化（ImageNetの事前学習モデルに合わせた設定）
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),         # ランダムリサイズ＆クロップ
    transforms.RandomHorizontalFlip(),         # ランダム左右反転
    transforms.ToTensor(),                     # Tensor形式に変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 平均と標準偏差で正規化
                         std=[0.229, 0.224, 0.225])
])

# 画像フォルダのパス（'cats', 'dogs' のフォルダを含む）
train_dir = 'path_to_data'  # ご自身の画像フォルダパスに変更してください

# データ読み込み（ImageFolderを使用）
train_data = datasets.ImageFolder(train_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# ------------------------
# モデルの定義（ResNet18）
# ------------------------

# 事前学習済みのResNet18を使用し、出力層だけ2クラスに変更
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 猫と犬の2クラス分類

# GPUが使用可能ならGPUを使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ------------------------
# 損失関数と最適化手法
# ------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------
# モデルの学習
# ------------------------

num_epochs = 10
train_loss_history = []
train_acc_history = []

all_preds = []   # 予測ラベルの記録（混同行列用）
all_labels = []  # 正解ラベルの記録（混同行列用）

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc * 100:.2f}%")

# ------------------------
# 学習過程の可視化
# ------------------------

plt.figure(figsize=(12, 5))

# 損失の推移
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_loss_history, label='Training Loss')
plt.xlabel('エポック数')
plt.ylabel('損失')
plt.title('損失の推移')

# 精度の推移
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_acc_history, label='Training Accuracy')
plt.xlabel('エポック数')
plt.ylabel('精度')
plt.title('精度の推移')

plt.tight_layout()
plt.show()

# ------------------------
# 混同行列の可視化
# ------------------------

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['猫', '犬'], yticklabels=['猫', '犬'])
plt.xlabel('予測ラベル')
plt.ylabel('正解ラベル')
plt.title('混同行列')
plt.show()

# ------------------------
# 学習済みモデルの保存
# ------------------------

torch.save(model.state_dict(), 'cat_dog_model.pth')
