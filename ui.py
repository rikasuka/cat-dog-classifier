import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QPalette, QColor, QFont
from PyQt5.QtCore import Qt

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('猫・犬 画像分類アプリ')
        self.setGeometry(200, 200, 700, 500)

        # 背景色を白に設定
        palette = QPalette()
        palette.setColor(QPalette.Background, QColor(255, 255, 255))
        self.setPalette(palette)

        # レイアウト設定
        layout = QVBoxLayout()
        top_layout = QHBoxLayout()

        # 画像表示用ラベル
        self.label = QLabel('画像をアップロードして分類してください', self)
        self.label.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(self.label)

        # 分類結果表示ラベル
        self.result_label = QLabel('')
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 16, QFont.Bold))
        top_layout.addWidget(self.result_label)

        layout.addLayout(top_layout)

        # アップロードボタン
        self.upload_btn = QPushButton('画像をアップロード', self)
        self.upload_btn.setFont(QFont("Arial", 12))
        self.upload_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; border-radius: 10px; padding: 10px;"
        )
        self.upload_btn.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_btn)

        self.setLayout(layout)
        self.show()

    def upload_image(self):
        # ファイル選択ダイアログを表示
        file_name, _ = QFileDialog.getOpenFileName(
            self, '画像を選択', '', '画像ファイル (*.png *.jpg *.jpeg *.bmp)'
        )
        if file_name:
            self.classify_image(file_name)

    def classify_image(self, img_path):
        # 学習済みモデルの読み込み（ResNet18）
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load('cat_dog_model.pth', map_location='cpu'))
        model.eval()

        # 画像の前処理
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(img_path).convert('RGB')
        input_img = transform(image).unsqueeze(0)

        # 予測
        with torch.no_grad():
            outputs = model(input_img)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        # 結果の表示内容を生成
        predicted_label = '猫' if predicted.item() == 0 else '犬'
        confidence_percentage = confidence.item() * 100
        result_text = f"予測結果: {predicted_label}（信頼度: {confidence_percentage:.2f}%）"

        # アップロード画像を表示（リサイズ付き）
        pixmap = QPixmap(img_path)
        original_width = pixmap.width()
        original_height = pixmap.height()
        max_width, max_height = 400, 300
        scale = min(max_width / pixmap.width(), max_height / pixmap.height())
        new_size = pixmap.size() * scale
        self.label.setPixmap(pixmap.scaled(new_size, Qt.KeepAspectRatio))
        self.label.setFixedSize(new_size.width(), new_size.height())

        # 分類結果を表示
        self.result_label.setText(result_text)
        self.result_label.adjustSize()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
