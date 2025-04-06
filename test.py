import os
import matplotlib.pyplot as plt
import numpy as np

# ============================
# 猫と犬の画像データの枚数を可視化するスクリプト
# ============================

# 【前提】以下のような構造の画像データフォルダが存在すること
# path_to_data/
# ├── cats/
# │   ├── cat1.jpg
# │   ├── cat2.jpg
# │   └── ...
# └── dogs/
#     ├── dog1.jpg
#     ├── dog2.jpg
#     └── ...

# データセットのディレクトリパスを指定
train_dir = 'path_to_data'  # ← 使用環境に応じてパスを変更してください

# 猫画像の枚数をカウント
cat_count = len([
    name for name in os.listdir(os.path.join(train_dir, 'cats'))
    if os.path.isfile(os.path.join(train_dir, 'cats', name))
])

# 犬画像の枚数をカウント
dog_count = len([
    name for name in os.listdir(os.path.join(train_dir, 'dogs'))
    if os.path.isfile(os.path.join(train_dir, 'dogs', name))
])

# カテゴリ名と対応する画像枚数をリストにまとめる
categories = ['猫', '犬']
counts = [cat_count, dog_count]

# ========================
# 1. 棒グラフによる可視化
# ========================
plt.figure(figsize=(6, 4))
plt.bar(categories, counts, width=0.5)
plt.xlabel('カテゴリ')
plt.ylabel('画像枚数')
plt.title('猫と犬の画像数（棒グラフ）')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ========================
# 2. 円グラフによる可視化
# ========================
plt.figure(figsize=(5, 5))
plt.pie(
    counts,
    labels=categories,
    autopct='%1.1f%%',     # パーセンテージを1桁小数で表示
    startangle=90,         # 円グラフの開始角度
    shadow=True            # 立体感のある表示
)
plt.title('猫と犬の割合（円グラフ）')
plt.axis('equal')          # 円がつぶれないように調整
plt.tight_layout()
plt.show()
