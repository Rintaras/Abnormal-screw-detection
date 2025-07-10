import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen
from PyQt5.QtCore import QTimer, Qt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ネジ締め検知システム')
        self.setGeometry(100, 100, 800, 600)
        self.label = QLabel(self)
        self.label.setGeometry(10, 10, 780, 580)
        self.label.setAlignment(Qt.AlignCenter)
        self.setStyleSheet('border: 8px solid red;')

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # 機械学習モデル
        self.model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, alpha=0.001)
        self.scaler = StandardScaler()
        self.X_train = []
        self.y_train = []
        self.is_fitted = False
        self.last_features = None
        self.last_frame = None
        self.last_prob = 0.5
        self.class_labels = np.array([0, 1])  # 0:外れ, 1:締め
        self.learn_count = 0  # 学習回数
        self.label_set = set()
        self.prob_history = []  # 確率の移動平均用
        self.prob_window = 10
        self.last_bbox = None  # バウンディングボックス情報

    def extract_features(self, frame):
        h, w, _ = frame.shape
        size = min(h, w) // 3
        cx, cy = w // 2, h // 2
        crop = frame[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # ヒストグラム
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
        hist = hist / (np.sum(hist) + 1e-6)
        # エッジ量
        edges = cv2.Canny(gray, 100, 200)
        edge_score = np.sum(edges) / (edges.shape[0] * edges.shape[1] * 255)
        # 円検出
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=size//4,
                                   param1=50, param2=30, minRadius=size//6, maxRadius=size//2)
        circle_score = 1.0 if circles is not None else 0.0
        # HOG特徴量
        winSize = (gray.shape[1]//2*2, gray.shape[0]//2*2)
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        resized = cv2.resize(gray, winSize)
        hog_feat = hog.compute(resized).flatten()
        # バウンディングボックス情報を保存
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # 一番大きい円を採用
            main_circle = max(circles[0, :], key=lambda c: c[2])
            # 元画像座標に変換
            x = main_circle[0] + (cx - size//2)
            y = main_circle[1] + (cy - size//2)
            r = main_circle[2]
            self.last_bbox = ('circle', (x, y, r))
        else:
            # クロップ領域を矩形として保存
            self.last_bbox = ('rect', (cx-size//2, cy-size//2, size, size))
        # 特徴量ベクトル
        features = np.concatenate([hist, [edge_score, circle_score], hog_feat])
        return features

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            features = self.extract_features(frame)
            self.last_features = features
            self.last_frame = frame
            # 判定は両ラベルが揃ってから有効化
            if self.is_fitted and len(self.label_set) == 2:
                X = self.scaler.transform([features])
                prob = self.model.predict_proba(X)[0][1]
                # 移動平均で平滑化
                self.prob_history.append(prob)
                if len(self.prob_history) > self.prob_window:
                    self.prob_history.pop(0)
                avg_prob = np.mean(self.prob_history)
                self.last_prob = avg_prob
            else:
                avg_prob = 0.5
                self.last_prob = avg_prob
            percent = int(avg_prob * 100)
            if percent >= 80:
                self.setStyleSheet('border: 8px solid green;')
            else:
                self.setStyleSheet('border: 8px solid red;')
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image).scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
            # バウンディングボックス描画
            painter = QPainter(pixmap)
            pen = QPen(QColor('yellow'))
            pen.setWidth(4)
            painter.setPen(pen)
            if self.last_bbox is not None:
                if self.last_bbox[0] == 'circle':
                    x, y, r = self.last_bbox[1]
                    scale_x = pixmap.width() / w
                    scale_y = pixmap.height() / h
                    painter.drawEllipse(int(x*scale_x - r*scale_x), int(y*scale_y - r*scale_y), int(2*r*scale_x), int(2*r*scale_y))
                elif self.last_bbox[0] == 'rect':
                    x, y, size, size2 = self.last_bbox[1]
                    scale_x = pixmap.width() / w
                    scale_y = pixmap.height() / h
                    painter.drawRect(int(x*scale_x), int(y*scale_y), int(size*scale_x), int(size2*scale_y))
            # テキスト描画
            painter.setPen(QColor('white'))
            painter.setFont(QFont('Arial', 32))
            painter.drawText(20, 50, f'締め付け確率: {percent}%')
            painter.setFont(QFont('Arial', 24))
            painter.drawText(20, 100, f'学習回数: {self.learn_count}')
            painter.end()
            self.label.setPixmap(pixmap)

    def keyPressEvent(self, event):
        if self.last_features is None:
            return
        if event.key() == Qt.Key_Y:
            label = 1
        elif event.key() == Qt.Key_N:
            label = 0
        else:
            return
        self.X_train.append(self.last_features)
        self.y_train.append(label)
        self.learn_count += 1
        self.label_set.add(label)
        self.fit_model()

    def fit_model(self):
        # バッチ学習: 全データで再fit
        if len(self.label_set) < 2:
            return
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, alpha=0.001)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 