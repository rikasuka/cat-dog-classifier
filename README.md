# cat-dog-classifier
猫・犬の画像を分類するPyQt5アプリです。

# 猫・犬 画像分類アプリ（PyQt5）

## 概要
このアプリは、PyTorch で学習した ResNet18 モデルを使って、猫 or 犬の画像を分類するデスクトップアプリです。

## 使用技術
- PyTorch（ResNet18）
- torchvision / PIL
- PyQt5（GUI）
- 学習済みモデル：`cat_dog_model.pth`

## 実行方法
```bash
pip install -r requirements.txt
python app.py
