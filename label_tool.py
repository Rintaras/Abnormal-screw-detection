import sys
import cv2
import os
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QPushButton, QListWidget, 
                             QFileDialog, QMessageBox, QSlider, QSpinBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt, QRect
import yaml

class LabelingTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('YOLOラベル付けツール')
        self.setGeometry(100, 100, 1400, 800)
        
        # メインウィジェット
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # 左側：画像表示
        self.image_widget = QWidget()
        image_layout = QVBoxLayout(self.image_widget)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        self.image_label.setStyleSheet('border: 2px solid gray;')
        image_layout.addWidget(self.image_label)
        
        # 画像操作ボタン
        button_layout = QHBoxLayout()
        self.prev_button = QPushButton('前の画像')
        self.prev_button.clicked.connect(self.prev_image)
        button_layout.addWidget(self.prev_button)
        
        self.next_button = QPushButton('次の画像')
        self.next_button.clicked.connect(self.next_image)
        button_layout.addWidget(self.next_button)
        
        self.save_button = QPushButton('ラベル保存')
        self.save_button.clicked.connect(self.save_labels)
        button_layout.addWidget(self.save_button)
        
        image_layout.addLayout(button_layout)
        
        # 右側：コントロールパネル
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # 画像リスト
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.load_image)
        control_layout.addWidget(QLabel('画像リスト:'))
        control_layout.addWidget(self.image_list)
        
        # ラベル情報
        self.label_info = QLabel('ラベル情報')
        self.label_info.setStyleSheet('background-color: #f0f0f0; padding: 10px;')
        control_layout.addWidget(self.label_info)
        
        # ラベル操作ボタン
        self.add_label_button = QPushButton('ラベル追加')
        self.add_label_button.clicked.connect(self.add_label)
        control_layout.addWidget(self.add_label_button)
        
        self.delete_label_button = QPushButton('ラベル削除')
        self.delete_label_button.clicked.connect(self.delete_label)
        control_layout.addWidget(self.delete_label_button)
        
        layout.addWidget(self.image_widget, 2)
        layout.addWidget(control_widget, 1)
        
        # データ
        self.images_dir = 'images'
        self.labels_dir = 'labels'
        self.current_image_path = None
        self.current_labels = []
        self.image_paths = []
        self.current_index = 0
        
        # マウス操作
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.image_label.mousePressEvent = self.mouse_press
        self.image_label.mouseMoveEvent = self.mouse_move
        self.image_label.mouseReleaseEvent = self.mouse_release
        
        # 画像読み込み
        self.load_image_list()
        
    def load_image_list(self):
        """画像リストを読み込み"""
        if not os.path.exists(self.images_dir):
            return
            
        self.image_paths = []
        for file in os.listdir(self.images_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(self.images_dir, file))
        
        self.image_list.clear()
        for path in self.image_paths:
            self.image_list.addItem(os.path.basename(path))
            
        if self.image_paths:
            self.load_image_by_index(0)
    
    def load_image_by_index(self, index):
        """インデックスで画像を読み込み"""
        if 0 <= index < len(self.image_paths):
            self.current_index = index
            self.load_image_from_path(self.image_paths[index])
    
    def load_image_from_path(self, image_path):
        """パスから画像を読み込み"""
        self.current_image_path = image_path
        image = cv2.imread(image_path)
        if image is not None:
            # ラベルファイルを読み込み
            label_path = self.get_label_path(image_path)
            self.current_labels = self.load_labels(label_path)
            
            # 画像を表示
            self.display_image(image)
            self.update_label_info()
    
    def get_label_path(self, image_path):
        """画像パスからラベルパスを取得"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(self.labels_dir, f'{base_name}.txt')
    
    def load_labels(self, label_path):
        """ラベルファイルを読み込み"""
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        labels.append({
                            'class': int(parts[0]),
                            'x_center': float(parts[1]),
                            'y_center': float(parts[2]),
                            'width': float(parts[3]),
                            'height': float(parts[4])
                        })
        return labels
    
    def display_image(self, image):
        """画像を表示"""
        h, w, _ = image.shape
        bytes_per_line = 3 * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # ラベルを描画
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor('red'), 3))
        painter.setFont(QFont('Arial', 12))
        
        for label in self.current_labels:
            x_center = label['x_center'] * w
            y_center = label['y_center'] * h
            width = label['width'] * w
            height = label['height'] * h
            
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.drawText(x1, y1 - 5, f'Neji (class {label["class"]})')
        
        painter.end()
        
        # スケールして表示
        scaled_pixmap = pixmap.scaled(
            self.image_label.width(), self.image_label.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
    
    def mouse_press(self, event):
        """マウス押下"""
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
    
    def mouse_move(self, event):
        """マウス移動"""
        if self.drawing:
            self.end_point = event.pos()
            self.update_image_with_rectangle()
    
    def mouse_release(self, event):
        """マウスリリース"""
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.end_point = event.pos()
            self.add_rectangle_label()
    
    def update_image_with_rectangle(self):
        """矩形を描画して画像を更新"""
        if not self.current_image_path:
            return
            
        image = cv2.imread(self.current_image_path)
        if image is None:
            return
            
        h, w, _ = image.shape
        bytes_per_line = 3 * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        painter = QPainter(pixmap)
        
        # 既存のラベルを描画
        painter.setPen(QPen(QColor('red'), 3))
        painter.setFont(QFont('Arial', 12))
        
        for label in self.current_labels:
            x_center = label['x_center'] * w
            y_center = label['y_center'] * h
            width = label['width'] * w
            height = label['height'] * h
            
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.drawText(x1, y1 - 5, f'Neji (class {label["class"]})')
        
        # 新しい矩形を描画
        if self.start_point and self.end_point:
            painter.setPen(QPen(QColor('blue'), 2))
            painter.drawRect(
                self.start_point.x(), self.start_point.y(),
                self.end_point.x() - self.start_point.x(),
                self.end_point.y() - self.start_point.y()
            )
        
        painter.end()
        
        scaled_pixmap = pixmap.scaled(
            self.image_label.width(), self.image_label.height(), 
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
    
    def add_rectangle_label(self):
        """矩形からラベルを追加"""
        if not self.start_point or not self.end_point:
            return
            
        # 画像の実際のサイズを取得
        image = cv2.imread(self.current_image_path)
        if image is None:
            return
            
        h, w, _ = image.shape
        
        # スケール係数を計算
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return
            
        scale_x = w / pixmap.width()
        scale_y = h / pixmap.height()
        
        # 座標を変換
        x1 = min(self.start_point.x(), self.end_point.x()) * scale_x
        y1 = min(self.start_point.y(), self.end_point.y()) * scale_y
        x2 = max(self.start_point.x(), self.end_point.x()) * scale_x
        y2 = max(self.start_point.y(), self.end_point.y()) * scale_y
        
        # YOLO形式に変換
        x_center = (x1 + x2) / 2 / w
        y_center = (y1 + y2) / 2 / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h
        
        # ラベルを追加
        self.current_labels.append({
            'class': 0,  # neji class
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height
        })
        
        self.update_label_info()
        self.display_image(image)
    
    def add_label(self):
        """ラベルを追加"""
        if self.current_image_path:
            self.add_rectangle_label()
    
    def delete_label(self):
        """選択されたラベルを削除"""
        if self.current_labels:
            self.current_labels.pop()
            self.update_label_info()
            if self.current_image_path:
                image = cv2.imread(self.current_image_path)
                if image is not None:
                    self.display_image(image)
    
    def save_labels(self):
        """ラベルを保存"""
        if not self.current_image_path:
            return
            
        label_path = self.get_label_path(self.current_image_path)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        
        with open(label_path, 'w') as f:
            for label in self.current_labels:
                f.write(f"{label['class']} {label['x_center']:.6f} {label['y_center']:.6f} {label['width']:.6f} {label['height']:.6f}\n")
        
        QMessageBox.information(self, '保存完了', f'ラベルを保存しました: {label_path}')
    
    def update_label_info(self):
        """ラベル情報を更新"""
        info_text = f"""
現在の画像: {os.path.basename(self.current_image_path) if self.current_image_path else 'なし'}

ラベル数: {len(self.current_labels)}
画像インデックス: {self.current_index + 1}/{len(self.image_paths)}

操作方法:
- マウスドラッグ: ラベル領域を選択
- ラベル追加: ボタンクリック
- ラベル削除: ボタンクリック
- ラベル保存: ボタンクリック
        """
        self.label_info.setText(info_text)
    
    def prev_image(self):
        """前の画像"""
        if self.current_index > 0:
            self.load_image_by_index(self.current_index - 1)
    
    def next_image(self):
        """次の画像"""
        if self.current_index < len(self.image_paths) - 1:
            self.load_image_by_index(self.current_index + 1)
    
    def load_image(self, item):
        """リストから画像を読み込み"""
        index = self.image_list.row(item)
        self.load_image_by_index(index)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LabelingTool()
    window.show()
    sys.exit(app.exec_()) 